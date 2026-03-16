import math
import itertools
import torch
import torch.nn as nn
from .wavefront_api import WavefrontConfig, generate_wavefront_schedule
from .wavefront_kernel import run_fetch_kernel

class WavefrontEngineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layers, schedule, config, routing_map):
        num_cells = math.prod(config.grid_shape)
        
        output_grids = [
            torch.zeros((num_cells, config.batch_size, config.dim), device=x.device, dtype=x.dtype)
            for _ in range(config.num_ports)
        ]
        
        saved_inputs = [None for _ in range(num_cells)]
        
        # Explicitly disable gradient tracking for the forward pass 
        # since we manually replay it in the backward pass
        with torch.no_grad():
            for tick_group in schedule:
                # Step 1 Group cells by their specific layer depth
                layer_groups = {}
                for coord in tick_group:
                    layer_depth = coord[0]
                    if layer_depth not in layer_groups:
                        layer_groups[layer_depth] = []
                    layer_groups[layer_depth].append(coord)

                # Step 2 Process each layer group as one massive batch
                for layer_depth, coords in layer_groups.items():
                    cell_indices = []

                    # Calculate flat indices for this active batch
                    for coord in coords:
                        cell_idx = 0
                        stride = 1
                        for i in reversed(range(len(config.grid_shape))):
                            cell_idx += coord[i] * stride
                            stride *= config.grid_shape[i]
                        cell_indices.append(cell_idx)

                    gathered_tensors = run_fetch_kernel(x, output_grids, routing_map, config, cell_indices)

                    # Split the gathered tensors into separate lists for the block input
                    stacked_ingredients = []
                    for i in range(len(config.dependencies)):
                        # The kernel outputs (num_active, num_deps, batch, dim)
                        # We slice out the dependency and swap the first two dimensions 
                        # so the FLB_Block receives its expected (batch, num_cells, dim) shape
                        dep_chunk = gathered_tensors[:, i, :, :]
                        stacked_ingredients.append(dep_chunk.transpose(0, 1))

                    # Run the user layer exactly once for all cells
                    out_tuple = layers[layer_depth](*stacked_ingredients)

                    # Step 5 Scatter the computed results back into their specific port lockers
                    for idx, cell_idx in enumerate(cell_indices):
                        for port_idx in range(config.num_ports):
                            output_grids[port_idx][cell_idx] = out_tuple[port_idx][:, idx, :]
                    
        ctx.save_for_backward(x)
        ctx.layers = layers
        ctx.config = config
        ctx.routing_map = routing_map
        ctx.schedule = schedule
        ctx.save_for_backward(x, *output_grids)
        
        return tuple(output_grids)
    
    @staticmethod
    def backward(ctx, *grad_output_grids):
        # Retrieve the saved tensors
        x = ctx.saved_tensors[0]
        output_grids = ctx.saved_tensors[1:]
        
        layers = ctx.layers
        config = ctx.config
        routing_map = ctx.routing_map
        schedule = ctx.schedule 
        
        num_cells = output_grids[0].shape[0]
        num_deps = len(config.dependencies)
        
        grad_x = torch.zeros_like(x)
        
        current_grad_grids = []
        for g in grad_output_grids:
            if g is not None:
                current_grad_grids.append(g.clone())
            else:
                current_grad_grids.append(torch.zeros((num_cells, config.batch_size, config.dim), device=x.device))
                
        for tick_group in reversed(schedule):
            # Group by layer
            layer_groups = {}
            for coord in tick_group:
                layer_depth = coord[0]
                if layer_depth not in layer_groups:
                    layer_groups[layer_depth] = []
                layer_groups[layer_depth].append(coord)
                
            for layer_depth, coords in layer_groups.items():
                cell_indices = []
                for coord in coords:
                    cell_idx = 0
                    stride = 1
                    for i in reversed(range(len(config.grid_shape))):
                        cell_idx += coord[i] * stride
                        stride *= config.grid_shape[i]
                    cell_indices.append(cell_idx)
                    
                # Use Triton to instantly re-fetch the forward memory state!
                gathered_tensors = run_fetch_kernel(x, output_grids, routing_map, config, cell_indices)
                
                tracked_stacked = []
                for i in range(num_deps):
                    # Same reshaping logic as the forward pass
                    dep_chunk = gathered_tensors[:, i, :, :].transpose(0, 1)
                    # Detach and require grad so the autograd engine tracks it
                    stacked = dep_chunk.detach().requires_grad_(True)
                    tracked_stacked.append(stacked)
                    
                with torch.enable_grad():
                    # Replay the forward pass for this block
                    out_tuple = layers[layer_depth](*tracked_stacked)
                    
                    # Gather the correct error signals for each port
                    batched_grads = []
                    for port_idx in range(config.num_ports):
                        gathered = [current_grad_grids[port_idx][cell_idx] for cell_idx in cell_indices]
                        batched_grads.append(torch.stack(gathered, dim=1))
                    
                    # Push the errors backward
                    torch.autograd.backward(tensors=list(out_tuple), grad_tensors=batched_grads)
                    
                # Scatter the calculated gradients back to their origins
                for idx, cell_idx in enumerate(cell_indices):
                    map_start = cell_idx * num_deps
                    
                    for i in range(num_deps):
                        target_idx = routing_map[map_start + i].item()
                        
                        # We look up which port this ingredient originally came from
                        # config.dependencies[i] looks like ((spatial_offset), port_idx)
                        _, source_port = config.dependencies[i]
                        
                        ing_grad = tracked_stacked[i].grad[:, idx, :]
                        
                        if target_idx == -1:
                            pass
                        elif target_idx == -2:
                            spatial_size = num_cells // config.grid_shape[0]
                            spatial_idx = cell_idx % spatial_size
                            
                            grad_x_flat = grad_x.view(config.batch_size, spatial_size, config.dim)
                            grad_x_flat[:, spatial_idx, :] += ing_grad
                        else:
                            # Add the gradient back to the specific cell and specific port it came from
                            current_grad_grids[source_port][target_idx] += ing_grad
                            
        # Return grad_x and Nones for the other non-tensor arguments in forward()
        return grad_x, None, None, None, None

class WavefrontEngine(nn.Module):
    def __init__(self, config: WavefrontConfig, layers: nn.ModuleList):
        super().__init__()
        self.config = config
        self.layers = layers
        self.schedule = generate_wavefront_schedule(config)
        
        # We pre-compute the routing map once during initialization.
        # register_buffer ensures PyTorch automatically moves this map to the GPU 
        # alongside your model weights, and saves it in your checkpoint files.
        self.register_buffer("routing_map", self._build_routing_map())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We hand everything off to our custom autograd function to save memory
        return WavefrontEngineFunction.apply(
            x, self.layers, self.schedule, self.config, self.routing_map
        )

    def _build_routing_map(self) -> torch.Tensor:
        """
        Creates a master 1D cheat sheet for the Triton kernel.
        Calculates the flattened logical index of every cell's dependencies.
        """
        shape = self.config.grid_shape
        deps = self.config.dependencies
        num_cells = math.prod(shape)
        num_deps = len(deps)
        
        # The cheat sheet is a flat array initialized with -1 (out of bounds)
        route_map = torch.full((num_cells * num_deps,), -1, dtype=torch.int32)
        
        # Generate every coordinate in the N-dimensional grid
        ranges = [range(size) for size in shape]
        coord_to_idx = {coord: i for i, coord in enumerate(itertools.product(*ranges))}
        
        for coord, cell_idx in coord_to_idx.items():
            map_start = cell_idx * num_deps
            
            for dep_idx, dep in enumerate(deps):
                # Unpack the new format
                spatial_offset, port_idx = dep
                
                # Calculate the exact N-dimensional address using the spatial_offset
                target_coord = tuple(c + d for c, d in zip(coord, spatial_offset))
                
                # Check if the target is safely inside the main grid
                is_valid = all(0 <= target_coord[i] < shape[i] for i in range(len(shape)))
                if is_valid:
                    route_map[map_start + dep_idx] = coord_to_idx[target_coord]
                
                # If target falls exactly one layer below the grid, it pulls from raw input 'x'
                elif target_coord[0] == -1 and all(0 <= target_coord[i] < shape[i] for i in range(1, len(shape))):
                    # We use -2 to flag Triton to pull from the 'x' tensor instead of the 'lat_grid'
                    route_map[map_start + dep_idx] = -2 

        return route_map