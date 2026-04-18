import math
import itertools
import torch
import torch.nn as nn
from torch.func import functional_call, vjp, stack_module_state
from torch.amp import custom_fwd, custom_bwd
from torch.nn.attention import SDPBackend, sdpa_kernel
from .wavefront_api import WavefrontConfig, generate_wavefront_schedule
from .wavefront_kernel import run_fetch_kernel
import wavefront_backend

class WavefrontEngineFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x, layers, config, routing_map, port_map, active_cells_buffer, active_layers_buffer, gathered_out_buffer, stacked_params, bwd_cache, graph_workspace, spatial_map_buffer, batched_forward, *initial_states):
        ctx.has_initial_states = [state is not None for state in initial_states]
        ctx.batched_forward = batched_forward 
        ctx.active_cells_buffer = active_cells_buffer
        ctx.active_layers_buffer = active_layers_buffer
        ctx.gathered_out_buffer = gathered_out_buffer
        ctx.layers = layers
        ctx.config = config
        ctx.routing_map = routing_map
        ctx.port_map = port_map
        ctx.stacked_params = stacked_params
        ctx.bwd_cache = bwd_cache
        ctx.graph_workspace = graph_workspace

        num_cells = math.prod(config.grid_shape)
        num_deps = len(config.dependencies)
        num_ticks = active_cells_buffer.shape[0]

        workspace = ctx.graph_workspace
        static_x = workspace['static_x']
        static_stacked_grids = workspace['static_stacked_grids']

        with torch.no_grad():
            # Step 1 Grab all live weights across all layers simultaneously
            live_params, _ = stack_module_state(list(layers))
            
            # Step 2 Extract the raw tensors into flat lists
            live_tensors = list(live_params.values())
            stacked_tensors = list(stacked_params.values())
            
            # Step 3 Fire the C++ bulk copy to handle everything at once without a Python loop
            torch._foreach_copy_(stacked_tensors, live_tensors)

        # Drop the new input data off at the loading dock
        static_x.copy_(x)

        with torch.no_grad():
            # --- 2. Run the C++ Orchestration Loop ---
            wavefront_backend.execute_forward(
                num_ticks, config.num_ports,
                run_fetch_kernel, batched_forward,
                active_cells_buffer, static_x, static_stacked_grids,
                routing_map, port_map, spatial_map_buffer, gathered_out_buffer,
                bwd_cache, stacked_params, config
            )

        # 3. Pick up the final answers from the exit dock
        ctx.save_for_backward(static_x, static_stacked_grids)
        
        final_grids = [static_stacked_grids[p][:num_cells] for p in range(config.num_ports)]
        return tuple(final_grids)
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, *grad_output_grids):
        x, stacked_grids = ctx.saved_tensors
        
        # Pull the essentials
        layers = ctx.layers
        config = ctx.config

        batched_forward = ctx.batched_forward
        
        # Pull the flattened memory and scheduling buffers
        active_cells_buffer = ctx.active_cells_buffer
        active_layers_buffer = ctx.active_layers_buffer
        gathered_out_buffer = ctx.gathered_out_buffer
        
        # Pull the specific optimized tools we built
        stacked_params = ctx.stacked_params
        bwd_cache = ctx.bwd_cache
        
        # Calculate dynamic shapes
        num_cells = math.prod(config.grid_shape)
        num_deps = len(config.dependencies)
        num_ticks = active_cells_buffer.shape[0]
        
        workspace = ctx.graph_workspace
        static_grad_x = workspace['static_grad_x']
        static_current_grad_grids = workspace['static_current_grad_grids']
        static_grad_outputs = workspace['static_grad_outputs']
        
        # Build the static weight accumulator dock exactly once
        if 'static_cell_grad_accumulators' not in workspace:
            # We allocate [num_cells, *weight_shape] so every cell writes without atomic locks
            workspace['static_cell_grad_accumulators'] = {
                k: torch.zeros((num_cells,) + v.shape[1:], dtype=v.dtype, device=x.device) 
                for k, v in stacked_params.items()
            }
        static_cell_grad_accumulators = workspace['static_cell_grad_accumulators']

        # 1. Drop the incoming errors off at the loading dock
        for i, g in enumerate(grad_output_grids):
            if g is not None:
                static_grad_outputs[i][:num_cells].copy_(g)

        # --- 2. Build a tiny Python bridge for the C++ code to hit ---
        def batched_backward_step(cell_params, tracked_stacked_tuple, batched_grads_tuple):
            _, backward_machine = vjp(batched_forward, cell_params, *tracked_stacked_tuple)
            return backward_machine(batched_grads_tuple)

        # --- 3. Fire the C++ Backend ---
        import wavefront_backend
        wavefront_backend.execute_backward(
            num_ticks, config.num_ports, num_deps, num_cells,
            batched_backward_step,
            active_cells_buffer, gathered_out_buffer,
            static_grad_x, static_current_grad_grids, static_grad_outputs,
            static_cell_grad_accumulators, bwd_cache, stacked_params,
            config.dependencies
        )

        num_layers = config.grid_shape[0]
        cells_per_layer = num_cells // num_layers

        with torch.no_grad():
            # 1. Reduce the gradients EXACTLY ONCE for all layers
            reduced_grads = {}
            for name, param in layers[0].named_parameters():
                reduced_grads[name] = static_cell_grad_accumulators[name].view(
                    num_layers, cells_per_layer, *param.shape
                ).sum(dim=1)

            # 2. Distribute the pre-calculated gradients
            for i, layer in enumerate(layers):
                for name, param in layer.named_parameters():
                    layer_grad = reduced_grads[name][i].to(param.dtype)
                    
                    if param.grad is None:
                        param.grad = layer_grad.clone()
                    else:
                        param.grad.add_(layer_grad)

        grad_initial_states = []
        if hasattr(ctx, 'has_initial_states'):
            for p, has_state in enumerate(ctx.has_initial_states):
                if has_state:
                    grad_initial_states.append(static_current_grad_grids[p][num_cells : num_cells + num_layers].clone())
                else:
                    grad_initial_states.append(None)
        else:
            grad_initial_states = [None] * config.num_ports
        
        standard_returns = [static_grad_x] + [None] * 12
        
        return tuple(standard_returns + grad_initial_states)

class WavefrontEngine(nn.Module):
    def __init__(self, config: WavefrontConfig, layers: nn.ModuleList):
        super().__init__()
        self.config = config
        self.layers = layers
        self.schedule = generate_wavefront_schedule(config)
        
        r_map_tensor = self._build_routing_map()
        self.register_buffer("routing_map", r_map_tensor)
        self.routing_map_list = r_map_tensor.tolist()
        
        port_list = [dep[1] for dep in config.dependencies]
        self.register_buffer("port_map", torch.tensor(port_list, dtype=torch.int32))

        self.compiled_cells, self.compiled_layers, self.max_cells_per_tick = self._compile_schedule()
        
        self.active_cells_buffer = None
        self.active_layers_buffer = None
        self.gathered_out_buffer = None

        def pure_layer_forward(params, *inputs):
            # 1. Restore the thread-local autocast state stripped by the C++/vmap boundary.
            # We dynamically use inputs[0].dtype so it perfectly matches the memory dock!
            with torch.amp.autocast('cuda', dtype=inputs[0].dtype):
                reshaped_inputs = tuple(inp.unsqueeze(1) for inp in inputs)
                out_tuple = torch.func.functional_call(self.layers[0], params, reshaped_inputs)
                
                # 2. Hard-cast the outputs back to the precise dock dtype just to be safe
                return tuple(out.squeeze(1).to(inputs[0].dtype) for out in out_tuple)

        self.batched_forward = torch.vmap(pure_layer_forward, randomness="different")

    @torch.compiler.disable
    def forward(self, x: torch.Tensor, initial_states: dict = None) -> torch.Tensor:
        if initial_states is None:
            initial_states = {}
            
        self._init_buffers(x)
        
        num_cells = math.prod(self.config.grid_shape)
        # We assume the boundary dock size is equal to the grid's first dimension 
        boundary_dock_size = self.config.grid_shape[0]
        
        # 1. Zero out the old boundary memory
        self.graph_workspace['static_stacked_grids'][:, num_cells : num_cells + boundary_dock_size].zero_()
        
        # 2. Dynamically inject any provided contexts into their respective ports
        states_list = []
        for p in range(self.config.num_ports):
            if p in initial_states:
                state_tensor = initial_states[p]
                self.graph_workspace['static_stacked_grids'][p][num_cells : num_cells + boundary_dock_size].copy_(state_tensor)
                states_list.append(state_tensor)
            else:
                states_list.append(None)

        # 3. Fire the Autograd engine, dynamically unpacking the states at the very end
        return WavefrontEngineFunction.apply(
            x, self.layers, self.config, 
            self.routing_map, self.port_map, 
            self.active_cells_buffer, self.active_layers_buffer,
            self.gathered_out_buffer, self.stacked_params,
            self.bwd_cache, self.graph_workspace,
            self.spatial_map_buffer,
            self.batched_forward,
            *states_list
        )

    def _init_buffers(self, x):
        # Ask PyTorch what precision it is currently using
        compute_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled('cuda') else x.dtype

        # Rebuild buffers if the device OR the precision changes
        if self.active_cells_buffer is not None and self.active_cells_buffer.device == x.device and self.gathered_out_buffer.dtype == compute_dtype:
            return

        if self.port_map.device != x.device:
            self.routing_map = self.routing_map.to(x.device)
            self.port_map = self.port_map.to(x.device)

        num_ticks = len(self.compiled_cells)
        num_deps = len(self.config.dependencies)
        max_cells = self.max_cells_per_tick

        # Pre-compute the modulo math for the Triton kernel to avoid % on the GPU
        if not hasattr(self, 'spatial_map_buffer') or self.spatial_map_buffer.device != x.device:
            num_cells = math.prod(self.config.grid_shape)
            seq_len = self.config.grid_shape[1]
            # Create the map and pad it with one extra '0' for the dummy cell to prevent out-of-bounds
            spatial_map = [(i % seq_len) for i in range(num_cells)] + [0]
            self.spatial_map_buffer = torch.tensor(spatial_map, dtype=torch.int32, device=x.device)

        # Move the flat arrays directly to the GPU
        self.active_cells_buffer = torch.tensor(self.compiled_cells, dtype=torch.int32, device=x.device)
        self.active_layers_buffer = torch.tensor(self.compiled_layers, dtype=torch.int32, device=x.device)

        # Shape is now natively: (Ticks, Deps, Max_Cells_Per_Tick, Batch, Dim)
        self.gathered_out_buffer = torch.zeros(
            (num_ticks, num_deps, max_cells, x.shape[0], self.config.dim), 
            device=x.device, dtype=compute_dtype
        )

        # Phase 1: Stack the physical weights from your custom modules exactly once
        # Detach the initial snapshot so it becomes a safe memory dock
        if not hasattr(self, 'stacked_params'):
            params, _ = stack_module_state(list(self.layers))
            self.stacked_params = {k: v.detach().clone() for k, v in params.items()}

        if not hasattr(self, 'bwd_cache'):
            self.bwd_cache = {}
            num_cells = math.prod(self.config.grid_shape)
            
            # The cache is now perfectly 1D, just like the active cells
            for t_idx in range(num_ticks):
                cell_indices = self.active_cells_buffer[t_idx]
                layer_indices = self.active_layers_buffer[t_idx]
                
                valid_mask = layer_indices >= 0
                safe_layer_indices = torch.where(valid_mask, layer_indices, 0)
                
                # Turn the boolean mask into a strict integer array!
                valid_idx = torch.where(valid_mask)[0] 
                
                valid_cells = cell_indices[valid_idx]
                
                tick_deps = []
                for i in range(num_deps):
                    target_indices = self.routing_map[valid_cells * num_deps + i]
                    
                    # Convert to strict integer indices!
                    x_idx = torch.where(target_indices == -2)[0]
                    x_targets = (valid_cells[x_idx] % (num_cells // self.config.grid_shape[0])).long()
                    
                    # Convert to strict integer indices!
                    grid_idx = torch.where(target_indices >= 0)[0]
                    grid_targets = target_indices[grid_idx].long()
                    
                    tick_deps.append({
                        'x_idx': x_idx,
                        'x_targets': x_targets,
                        'grid_idx': grid_idx,
                        'grid_targets': grid_targets
                    })
                    
                self.bwd_cache[t_idx] = {
                    'safe_layer_indices': safe_layer_indices,
                    'valid_cells': valid_cells,
                    'valid_idx': valid_idx,
                    'deps': tick_deps
                }

        if not hasattr(self, 'graph_workspace') or self.graph_workspace['static_x'].dtype != compute_dtype:
            batch = self.config.batch_size
            dim = self.config.dim
            seq_len = self.config.grid_shape[1]
            num_cells = math.prod(self.config.grid_shape)
            
            # Dynamically calculate the boundary dock size
            boundary_dock_size = self.config.grid_shape[0]
            
            self.graph_workspace = {                
                'static_x': torch.zeros((batch, seq_len, dim), device=x.device, dtype=compute_dtype),
                
                # Replace num_cells + 1 with num_cells + boundary_dock_size
                'static_stacked_grids': torch.zeros((self.config.num_ports, num_cells + boundary_dock_size, batch, dim), device=x.device, dtype=compute_dtype),
                
                'static_grad_x': torch.zeros((batch, seq_len, dim), device=x.device, dtype=compute_dtype),
                
                # Do the same for the backward memory docks!
                'static_grad_outputs': [torch.zeros((num_cells + boundary_dock_size, batch, dim), device=x.device, dtype=compute_dtype) for _ in range(self.config.num_ports)],
                'static_current_grad_grids': [torch.zeros((num_cells + boundary_dock_size, batch, dim), device=x.device, dtype=compute_dtype) for _ in range(self.config.num_ports)]
            }

    def _compile_schedule(self):
        num_cells = math.prod(self.config.grid_shape)
        max_cells_per_tick = 0
        
        # 1. Find the maximum number of cells active in ANY single tick
        for tick_group in self.schedule:
            max_cells_per_tick = max(max_cells_per_tick, len(tick_group))
            
        compiled_cells = []
        compiled_layers = []
        
        for tick_group in self.schedule:
            tick_cells = []
            tick_layers = []
            for coord in tick_group:
                l = coord[0] # Layer depth
                idx = 0
                stride = 1
                # Calculate the flat spatial index
                for d in reversed(range(len(self.config.grid_shape))):
                    idx += coord[d] * stride
                    stride *= self.config.grid_shape[d]
                
                tick_cells.append(idx)
                tick_layers.append(l)
            
            # 2. Pad the tick to a uniform size with dummy variables
            while len(tick_cells) < max_cells_per_tick:
                tick_cells.append(num_cells) 
                tick_layers.append(-1)
                
            compiled_cells.append(tick_cells)
            compiled_layers.append(tick_layers)
            
        return compiled_cells, compiled_layers, max_cells_per_tick

    def _build_routing_map(self) -> torch.Tensor:
        num_cells = math.prod(self.config.grid_shape)
        num_deps = len(self.config.dependencies)
        routing_map = torch.full((num_cells * num_deps,), -1, dtype=torch.int32)
        def get_flat_idx(coord):
            idx = 0
            stride = 1
            for i in reversed(range(len(self.config.grid_shape))):
                idx += coord[i] * stride
                stride *= self.config.grid_shape[i]
            return idx
        ranges = [range(size) for size in self.config.grid_shape]
        for coord in itertools.product(*ranges):
            current_idx = get_flat_idx(coord)
            for dep_idx, dep in enumerate(self.config.dependencies):
                spatial_offset, _ = dep
                target_coord = tuple(c + d for c, d in zip(coord, spatial_offset))
                is_valid = all(0 <= target_coord[i] < self.config.grid_shape[i] for i in range(len(coord)))
                map_pos = current_idx * num_deps + dep_idx
                if is_valid:
                    routing_map[map_pos] = get_flat_idx(target_coord)
                elif target_coord[0] == -1 and all(0 <= target_coord[i] < self.config.grid_shape[i] for i in range(1, len(coord))):
                    routing_map[map_pos] = -2
        return routing_map