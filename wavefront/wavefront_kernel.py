import torch
import triton
import triton.language as tl
import math

def fetch_mapped_context(x, output_grids, cell_idx, routing_map, config):
    num_deps = len(config.dependencies)
    ingredients = []
    map_start = cell_idx * num_deps
    
    for i in range(num_deps):
        target_idx = routing_map[map_start + i].item()
        _, port_idx = config.dependencies[i]
        
        if target_idx == -1:
            ingredients.append(torch.zeros_like(output_grids[0][0]))
        elif target_idx == -2:
            spatial_size = output_grids[0].shape[0] // config.grid_shape[0]
            spatial_idx = cell_idx % spatial_size
            x_flat = x.view(config.batch_size, spatial_size, config.dim) 
            ingredients.append(x_flat[:, spatial_idx, :])
        else:
            ingredients.append(output_grids[port_idx][target_idx])
            
    return ingredients

@triton.jit
def fetch_mapped_context_kernel(
    x_ptr, out_grids_ptr, routing_map_ptr, port_map_ptr, gathered_out_ptr,
    active_cells_ptr, spatial_map_ptr,
    num_active_cells, seq_len, num_total_cells, actual_dim,
    
    stride_x_batch, stride_x_seq, stride_x_dim,
    stride_port, stride_cell, stride_batch, stride_dim,
    stride_gathered_cell, stride_gathered_dep, stride_gathered_batch, stride_gathered_dim,
    
    BLOCK_SIZE: tl.constexpr, BLOCK_DIM: tl.constexpr, NUM_DEPS: tl.constexpr,
):
    pid_cell = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    
    active_offsets = pid_cell * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cell_mask = active_offsets < num_active_cells
    
    cell_offsets = tl.load(active_cells_ptr + active_offsets, mask=cell_mask, other=num_total_cells)
    
    # Mask out padded dummy cells
    is_real_cell = (cell_offsets < num_total_cells) & cell_mask
    
    map_offsets = cell_offsets * NUM_DEPS
    dim_offsets = tl.arange(0, BLOCK_DIM)
    
    cell_offsets_2d = cell_offsets[:, None]
    active_offsets_2d = active_offsets[:, None] 
    dim_offsets_2d = dim_offsets[None, :]
    
    dim_mask_2d = dim_offsets_2d < actual_dim
    
    spatial_idx_2d = tl.load(spatial_map_ptr + cell_offsets_2d)
    x_ptrs = x_ptr + (pid_batch * stride_x_batch) + (spatial_idx_2d * stride_x_seq) + (dim_offsets_2d * stride_x_dim)

    for i in tl.static_range(NUM_DEPS): 
        dep_offsets = map_offsets + i
        target_idx = tl.load(routing_map_ptr + dep_offsets, mask=is_real_cell, other=-1)
        port_idx = tl.load(port_map_ptr + i)
        target_idx_2d = target_idx[:, None]
        
        grid_ptrs = out_grids_ptr + (port_idx * stride_port) + (target_idx_2d * stride_cell) + (pid_batch * stride_batch) + (dim_offsets_2d * stride_dim)
        
        # Apply the dim_mask to your memory loads
        valid_mask = is_real_cell[:, None] & (target_idx_2d >= 0) & dim_mask_2d 
        grid_data = tl.load(grid_ptrs, mask=valid_mask, other=0.0)
        
        x_mask = is_real_cell[:, None] & (target_idx_2d == -2) & dim_mask_2d
        x_data = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        final_data = tl.where(target_idx_2d == -2, x_data, grid_data)
        
        out_ptrs = gathered_out_ptr + (active_offsets_2d * stride_gathered_cell) + (i * stride_gathered_dep) + (pid_batch * stride_gathered_batch) + (dim_offsets_2d * stride_gathered_dim)
        
        # Apply the dim_mask to your memory store
        tl.store(out_ptrs, final_data, mask=cell_mask[:, None] & dim_mask_2d)

def run_fetch_kernel(x, stacked_grids, routing_map, port_map, spatial_map, config, active_cells_tensor, gathered_out):
    num_deps = len(config.dependencies)
    seq_len = config.grid_shape[1]
    num_active_cells = active_cells_tensor.size(0)
    num_total_cells = math.prod(config.grid_shape)
    
    actual_batch_size = x.shape[0] # <--- THE FIX
    
    BLOCK_SIZE = 32
    BLOCK_DIM = triton.next_power_of_2(config.dim)
    
    # Launch exactly actual_batch_size grids in the Y-dimension
    grid = (triton.cdiv(num_active_cells, BLOCK_SIZE), actual_batch_size)
    
    fetch_mapped_context_kernel[grid](
        x, stacked_grids, routing_map, port_map, gathered_out,
        active_cells_tensor, spatial_map,
        num_active_cells, seq_len, num_total_cells, config.dim,
        x.stride(0), x.stride(1), x.stride(2),
        stacked_grids.stride(0), stacked_grids.stride(1), stacked_grids.stride(2), stacked_grids.stride(3),
        gathered_out.stride(1), gathered_out.stride(0), gathered_out.stride(2), gathered_out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE, BLOCK_DIM=BLOCK_DIM, NUM_DEPS=num_deps,
    )
    return gathered_out