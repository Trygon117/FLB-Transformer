import torch
import triton
import triton.language as tl

def fetch_mapped_context(x, output_grids, cell_idx, routing_map, config):
    # This reads the routing map and fetches the exact tensors needed
    num_deps = len(config.dependencies)
    ingredients = []
    
    map_start = cell_idx * num_deps
    
    for i in range(num_deps):
        target_idx = routing_map[map_start + i].item()
        
        # We need to unpack the tuple to find which port to pull from
        _, port_idx = config.dependencies[i]
        
        if target_idx == -1:
            # The coordinate fell off the map so we return a blank tensor
            ingredients.append(torch.zeros_like(output_grids[0][0]))
            
        elif target_idx == -2:
            # The coordinate is exactly one layer below our grid so we pull from the raw input x
            spatial_size = output_grids[0].shape[0] // config.grid_shape[0]
            spatial_idx = cell_idx % spatial_size
            
            x_flat = x.view(config.batch_size, spatial_size, config.dim) 
            ingredients.append(x_flat[:, spatial_idx, :])
            
        else:
            # Standard Case we pull from the specific port grid
            ingredients.append(output_grids[port_idx][target_idx])
            
    return ingredients

@triton.jit
def fetch_mapped_context_kernel(
    x_ptr, out_grids_ptr, routing_map_ptr, port_map_ptr, gathered_out_ptr,
    active_cells_ptr,
    num_active_cells, num_deps, seq_len,
    
    # We now pass the specific memory strides for x
    stride_x_batch, stride_x_seq, stride_x_dim,
    stride_port, stride_cell, stride_batch, stride_dim,
    stride_gathered_cell, stride_gathered_dep, stride_gathered_batch, stride_gathered_dim,
    
    BLOCK_SIZE: tl.constexpr, BLOCK_DIM: tl.constexpr,
):
    pid_cell = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1) # Grab the specific batch index for this worker
    
    active_offsets = pid_cell * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cell_mask = active_offsets < num_active_cells
    
    cell_offsets = tl.load(active_cells_ptr + active_offsets, mask=cell_mask, other=0)
    
    map_offsets = cell_offsets * num_deps
    dim_offsets = tl.arange(0, BLOCK_DIM)
    
    cell_offsets_2d = cell_offsets[:, None]
    active_offsets_2d = active_offsets[:, None] 
    dim_offsets_2d = dim_offsets[None, :]
    
    spatial_idx_2d = (cell_offsets_2d % seq_len)
    
    # Calculate x addresses using the correct batch and sequence strides
    x_ptrs = x_ptr + (pid_batch * stride_x_batch) + (spatial_idx_2d * stride_x_seq) + (dim_offsets_2d * stride_x_dim)

    for i in range(num_deps):
        dep_offsets = map_offsets + i
        target_idx = tl.load(routing_map_ptr + dep_offsets, mask=cell_mask, other=-1)
        port_idx = tl.load(port_map_ptr + i)
        target_idx_2d = target_idx[:, None]
        
        # Include the batch stride to pull from the correct slice of the grid
        grid_ptrs = out_grids_ptr + (port_idx * stride_port) + (target_idx_2d * stride_cell) + (pid_batch * stride_batch) + (dim_offsets_2d * stride_dim)
        
        valid_mask = cell_mask[:, None] & (target_idx_2d >= 0)
        grid_data = tl.load(grid_ptrs, mask=valid_mask, other=0.0)
        
        x_mask = cell_mask[:, None] & (target_idx_2d == -2)
        x_data = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        final_data = tl.where(target_idx_2d == -2, x_data, grid_data)
        
        # Save using the specific batch stride
        out_ptrs = gathered_out_ptr + (active_offsets_2d * stride_gathered_cell) + (i * stride_gathered_dep) + (pid_batch * stride_gathered_batch) + (dim_offsets_2d * stride_gathered_dim)
        
        tl.store(out_ptrs, final_data, mask=cell_mask[:, None])


def run_fetch_kernel(x, output_grids, routing_map, config, active_cells):
    num_deps = len(config.dependencies)
    seq_len = config.grid_shape[1]
    num_active_cells = len(active_cells)
    
    active_cells_tensor = torch.tensor(active_cells, dtype=torch.int32, device=x.device)
    stacked_grids = torch.stack(output_grids, dim=0)
    port_list = [dep[1] for dep in config.dependencies]
    port_map = torch.tensor(port_list, dtype=torch.int32, device=x.device)
    
    gathered_out = torch.empty((num_active_cells, num_deps, config.batch_size, config.dim), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE = 32
    BLOCK_DIM = triton.next_power_of_2(config.dim)
    
    # Launch a 2D grid: (cell_blocks, batch_size)
    grid = (triton.cdiv(num_active_cells, BLOCK_SIZE), config.batch_size)
    
    fetch_mapped_context_kernel[grid](
        x, stacked_grids, routing_map, port_map, gathered_out,
        active_cells_tensor, 
        num_active_cells, num_deps, seq_len,
        
        # Strides for x
        x.stride(0), x.stride(1), x.stride(2),
        
        # Strides for the stacked grids
        stacked_grids.stride(0), stacked_grids.stride(1), stacked_grids.stride(2), stacked_grids.stride(3),
        
        # Strides for the gathered output (cell, dep, batch, dim)
        gathered_out.stride(0), gathered_out.stride(1), gathered_out.stride(2), gathered_out.stride(3),
        
        BLOCK_SIZE=BLOCK_SIZE, BLOCK_DIM=BLOCK_DIM,
    )
    
    return gathered_out