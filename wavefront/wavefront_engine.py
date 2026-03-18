import math
import itertools
import torch
import torch.nn as nn
from torch.func import functional_call, vjp, stack_module_state
from torch.amp import custom_fwd, custom_bwd
from torch.nn.attention import SDPBackend, sdpa_kernel
from .wavefront_api import WavefrontConfig, generate_wavefront_schedule
from .wavefront_kernel import run_fetch_kernel

class WavefrontEngineFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x, layers, config, routing_map, port_map, active_cells_buffer, active_layers_buffer, gathered_out_buffer, stacked_params, bwd_cache, graph_workspace):
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
            def pure_layer_forward(params, *inputs):
                reshaped_inputs = tuple(inp.unsqueeze(1) for inp in inputs)
                out_tuple = functional_call(layers[0], params, reshaped_inputs)
                return tuple(out.squeeze(1) for out in out_tuple)

            batched_forward = torch.vmap(pure_layer_forward, randomness="different")

            # 2. Package the entire execution loop into a single reusable function
            def execute_static_forward():
                with sdpa_kernel(SDPBackend.MATH):
                    for tick_idx in range(num_ticks):
                        cell_indices_tensor = active_cells_buffer[tick_idx]
                        
                        # Use static_x and static_stacked_grids instead of the dynamic ones
                        run_fetch_kernel(
                            static_x, static_stacked_grids, routing_map, port_map, config, 
                            cell_indices_tensor, 
                            gathered_out_buffer[tick_idx]
                        )

                        tick_cache = bwd_cache[tick_idx]
                        safe_layer_indices = tick_cache['safe_layer_indices']
                        valid_cells = tick_cache['valid_cells']
                        valid_idx = tick_cache['valid_idx']

                        cell_params = {k: v[safe_layer_indices] for k, v in stacked_params.items()}

                        stacked_ingredients = []
                        for d_idx in range(num_deps):
                            stacked_ingredients.append(gathered_out_buffer[tick_idx, d_idx])

                        out_tuple = batched_forward(cell_params, *stacked_ingredients)

                        for port_idx in range(config.num_ports):
                            port_output = out_tuple[port_idx] 
                            static_stacked_grids[port_idx][valid_cells] = port_output[valid_idx]

            # 3. The Graph Capture Logic
            if not workspace['is_recorded']:
                # PyTorch requires us to warm up the engine once before recording
                torch.cuda.synchronize()
                execute_static_forward()
                torch.cuda.synchronize()

                # Turn the cameras on and record the static math track
                with torch.cuda.graph(workspace['fwd_graph']):
                    execute_static_forward()
                
            # 4. Hit play. The GPU runs the entire track instantly without the CPU.
            workspace['fwd_graph'].replay()

        # 5. Pick up the final answers from the exit dock
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
        
        # Pull the flattened memory and scheduling buffers
        active_cells_buffer = ctx.active_cells_buffer
        active_layers_buffer = ctx.active_layers_buffer
        gathered_out_buffer = ctx.gathered_out_buffer
        
        # Pull the specific optimized tools we built
        stacked_params = ctx.stacked_params
        bwd_cache = ctx.bwd_cache
        
        # Calculate dynamic shapes
        num_cells = stacked_grids.shape[1] - 1
        num_deps = len(config.dependencies)
        num_ticks = active_cells_buffer.shape[0]
        
        workspace = ctx.graph_workspace
        static_grad_x = workspace['static_grad_x']
        static_current_grad_grids = workspace['static_current_grad_grids']
        static_grad_outputs = workspace['static_grad_outputs']
        
        # Build the static weight accumulator dock exactly once
        if 'static_grad_accumulators' not in workspace:
            workspace['static_grad_accumulators'] = {k: torch.zeros_like(v) for k, v in stacked_params.items()}
        static_grad_accumulators = workspace['static_grad_accumulators']

        # 1. Drop the incoming errors off at the loading dock
        for i, g in enumerate(grad_output_grids):
            if g is not None:
                static_grad_outputs[i][:num_cells].copy_(g)

        def pure_layer_forward(params, *inputs):
            reshaped_inputs = tuple(inp.unsqueeze(1) for inp in inputs)
            out_tuple = functional_call(layers[0], params, reshaped_inputs)
            return tuple(out.squeeze(1) for out in out_tuple)

        batched_forward = torch.vmap(pure_layer_forward, randomness="different")

        # 2. Package the backward loop into the static recorder
        def execute_static_backward():
            # Clean the slate. Zero the loading docks before the math begins.
            static_grad_x.zero_()
            for g in static_current_grad_grids:
                g.zero_()
            for g in static_grad_accumulators.values():
                g.zero_()
                
            # Move the starting errors into the active grid
            for i in range(config.num_ports):
                static_current_grad_grids[i][:num_cells].copy_(static_grad_outputs[i][:num_cells])

            with sdpa_kernel(SDPBackend.MATH):
                for tick_idx in reversed(range(num_ticks)):
                    cell_indices_tensor = active_cells_buffer[tick_idx]
                    gathered_tensors = gathered_out_buffer[tick_idx]

                    tick_cache = bwd_cache[tick_idx]
                    safe_layer_indices = tick_cache['safe_layer_indices']
                    valid_idx = tick_cache['valid_idx']
                    valid_layer_targets = tick_cache['valid_layer_targets']

                    cell_params = {k: v[safe_layer_indices] for k, v in stacked_params.items()}

                    tracked_stacked = []
                    for i in range(num_deps):
                        tracked_stacked.append(gathered_tensors[i])

                    out_tuple, backward_machine = vjp(batched_forward, cell_params, *tracked_stacked)

                    batched_grads = []
                    for port_idx in range(config.num_ports):
                        # Use the static grids!
                        port_grads = static_current_grad_grids[port_idx][cell_indices_tensor]
                        batched_grads.append(port_grads)

                    grad_returns = backward_machine(tuple(batched_grads))
                    grad_params = grad_returns[0]  
                    grad_inputs = grad_returns[1:]

                    for name, g in grad_params.items():
                        if g is not None:
                            safe_g = g[valid_idx].to(static_grad_accumulators[name].dtype)
                            static_grad_accumulators[name].index_add_(0, valid_layer_targets, safe_g)

                    for i in range(num_deps):
                        _, source_port = config.dependencies[i]
                        dep_cache = tick_cache['deps'][i]

                        dep_grads = grad_inputs[i] 
                        valid_grads = dep_grads[valid_idx] 

                        if dep_cache['x_targets'].numel() > 0:
                            static_grad_x.index_add_(1, dep_cache['x_targets'], valid_grads[dep_cache['x_idx']].transpose(0, 1))

                        if dep_cache['grid_targets'].numel() > 0:
                            static_current_grad_grids[source_port].index_add_(0, dep_cache['grid_targets'], valid_grads[dep_cache['grid_idx']])

        # 3. The Graph Capture Logic
        if not workspace['is_recorded']:
            # PyTorch requires us to warm up the engine once before recording
            torch.cuda.synchronize()
            execute_static_backward()
            torch.cuda.synchronize()

            # Turn the cameras on and record the static math track
            with torch.cuda.graph(workspace['bwd_graph']):
                execute_static_backward()
                
            # Lock the cameras. The engine is permanently built.
            workspace['is_recorded'] = True

        # 4. Hit play. The GPU runs the backward pass completely independent of Python.
        workspace['bwd_graph'].replay()

        # 5. Pick the final errors up off the exit dock and apply them to the physical model
        for i, layer in enumerate(layers):
            for name, param in layer.named_parameters():
                # Manually cast the BFloat16 gradient back to FP32
                grad_to_apply = static_grad_accumulators[name][i].to(param.dtype)
                
                if param.grad is None:
                    param.grad = grad_to_apply
                else:
                    param.grad += grad_to_apply
                    
        return static_grad_x, None, None, None, None, None, None, None, None, None, None

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

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._init_buffers(x)
        return WavefrontEngineFunction.apply(
            x, self.layers, self.config, 
            self.routing_map, self.port_map, 
            self.active_cells_buffer, self.active_layers_buffer,
            self.gathered_out_buffer, self.stacked_params,
            self.bwd_cache, self.graph_workspace
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
                
                # Pre-calculate the layer targets for the backward pass
                valid_layer_targets = layer_indices[valid_idx].long()
                
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
                    'valid_layer_targets': valid_layer_targets,
                    'deps': tick_deps
                }

        if not hasattr(self, 'graph_workspace') or self.graph_workspace['static_x'].dtype != compute_dtype:
            batch = self.config.batch_size
            dim = self.config.dim
            seq_len = self.config.grid_shape[1]
            num_cells = math.prod(self.config.grid_shape)
            
            self.graph_workspace = {
                'is_recorded': False,
                'fwd_graph': torch.cuda.CUDAGraph(),
                'bwd_graph': torch.cuda.CUDAGraph(),
                
                'static_x': torch.zeros((batch, seq_len, dim), device=x.device, dtype=compute_dtype),
                'static_stacked_grids': torch.zeros((self.config.num_ports, num_cells + 1, batch, dim), device=x.device, dtype=compute_dtype),
                
                'static_grad_x': torch.zeros((batch, seq_len, dim), device=x.device, dtype=compute_dtype),
                'static_grad_outputs': [torch.zeros((num_cells + 1, batch, dim), device=x.device, dtype=compute_dtype) for _ in range(self.config.num_ports)],
                'static_current_grad_grids': [torch.zeros((num_cells + 1, batch, dim), device=x.device, dtype=compute_dtype) for _ in range(self.config.num_ports)]
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