import math
import traceback
import torch
import torch.nn as nn
from .wavefront_api import WavefrontConfig, generate_wavefront_schedule
from .wavefront_engine import WavefrontEngine
from .wavefront_kernel import fetch_mapped_context
import copy

def verify_schedule(config: WavefrontConfig, schedule: list) -> bool:
    """
    Universal verifier to ensure the schedule mathematically respects all dependencies in any dimension.
    """
    execution_log = {}

    # Map each coordinate to the exact tick it was executed
    for tick_idx, tick_group in enumerate(schedule):
        for coord in tick_group:
            if coord in execution_log:
                print(f"  [FAIL] Duplicate execution found for coordinate {coord}")
                return False
            execution_log[coord] = tick_idx

    # Check Completeness by multiplying all dimensions together
    expected_blocks = math.prod(config.grid_shape)
    if len(execution_log) != expected_blocks:
        print(f"  [FAIL] Missing blocks. Expected {expected_blocks} got {len(execution_log)}")
        return False

    # Check Dependencies dynamically
    for coord, current_tick in execution_log.items():
        for dep in config.dependencies:
            # Unpack the spatial offset from the port index
            spatial_offset, port_idx = dep
            
            # Calculate the exact grid address using the spatial offset
            dep_coord = tuple(c + d for c, d in zip(coord, spatial_offset))

            # Check if that address is actually inside the boundaries
            is_valid = all(0 <= dep_coord[i] < config.grid_shape[i] for i in range(len(coord)))

            if is_valid:
                dep_tick = execution_log[dep_coord]
                if current_tick <= dep_tick:
                    print(f"  [FAIL] Dependency violation. {coord} ran before or with {dep_coord}")
                    return False

    return True

def test_standard_2d_grid():
    print("Running Test Standard 2D Grid...", end=" ")
    config = WavefrontConfig(
        grid_shape=(4, 4), 
        batch_size=1,
        dependencies=[((-1, 0), 0), ((0, -1), 1)],
        num_ports=2
    )
    schedule = generate_wavefront_schedule(config)
    assert verify_schedule(config, schedule), "Standard 2D grid failed validation."
    print("PASS")

def test_custom_3d_video_grid():
    print("Running Test Custom 3D Video Grid...", end=" ")
    config = WavefrontConfig(
        grid_shape=(3, 8, 8),
        batch_size=1,
        dependencies=[((-1, 0, 0), 0), ((0, -1, 0), 1), ((0, 0, -1), 1)],
        num_ports=2
    )
    schedule = generate_wavefront_schedule(config)
    assert verify_schedule(config, schedule), "Custom 3D grid failed validation."
    print("PASS")

class DummyLayer(nn.Module):
    def __init__(self, dim, num_dependencies, num_ports=2):
        super().__init__()
        self.num_ports = num_ports
        self.dim = dim
        # We multiply the output size by the number of ports so we can split it
        self.linear = nn.Linear(dim * num_dependencies, dim * num_ports)

    def forward(self, *inputs):
        context = torch.cat(inputs, dim=-1)
        out = self.linear(context)
        
        # Split the wide output tensor into a distinct tuple for each port
        # out shape is (batch, cells, dim * num_ports)
        split_outputs = []
        for i in range(self.num_ports):
            split_outputs.append(out[..., i * self.dim : (i + 1) * self.dim])
            
        return tuple(split_outputs)

def _test_fetch(grid_shape, x_input, output_grids, coord, dep):
    """A helper function to simulate the engine fetching logic."""
    l, t = coord
    spatial_offset, port_idx = dep
    dl, dt = spatial_offset
    target_l, target_t = l + dl, t + dt
    
    if 0 <= target_l < grid_shape[0] and 0 <= target_t < grid_shape[1]:
        # We target the specific port grid
        grid_to_use = output_grids[port_idx]
        if torch.is_tensor(grid_to_use):
            return grid_to_use[target_l, :, target_t, :]
        return grid_to_use[target_l][target_t]
        
    elif target_l == -1 and 0 <= target_t < grid_shape[1]:
        return x_input[:, target_t, :]
    else:
        return torch.zeros_like(x_input[:, 0, :])

def test_backprop_flow():
    print("Running Test Autograd Gradient Flow...", end=" ")
    config = WavefrontConfig(
        grid_shape=(3, 4), batch_size=1, dim=8, 
        dependencies=[((-1, 0), 0), ((0, -1), 1)], num_ports=2
    )
    schedule = generate_wavefront_schedule(config)
    
    num_layers, seq_len = config.grid_shape
    layers = nn.ModuleList([DummyLayer(config.dim, len(config.dependencies), config.num_ports) for _ in range(num_layers)])
    x_input = torch.randn(config.batch_size, seq_len, config.dim, requires_grad=True)
    
    # Initialize a distinct 2D grid for each port
    output_grids = [[[None for _ in range(seq_len)] for _ in range(num_layers)] for _ in range(config.num_ports)]
    
    for tick_group in schedule:
        for coord in tick_group:
            l, t = coord
            fetched = [_test_fetch(config.grid_shape, x_input, output_grids, coord, dep) for dep in config.dependencies]
            
            # The dummy layer now returns a tuple
            out_tuple = layers[l](*fetched)
            
            # Scatter outputs to their respective port grids
            for port_idx in range(config.num_ports):
                output_grids[port_idx][l][t] = out_tuple[port_idx]
            
    # Calculate loss from Port 0 of the very last cell
    loss = output_grids[0][-1][-1].sum()
    loss.backward()
    
    assert layers[0].linear.weight.grad is not None, "Gradient tape snapped."
    print("PASS")

def test_mapped_fetcher():
    print("Running Test Mapped Fetcher Logic...", end=" ")
    
    # 1. Setup a standard 2D configuration with the new multi-port routing
    config = WavefrontConfig(
        grid_shape=(3, 4), 
        batch_size=2, 
        dim=8, 
        dependencies=[((-1, 0), 0), ((0, -1), 1)],
        num_ports=2
    )
    layers = nn.ModuleList([DummyLayer(config.dim, len(config.dependencies), config.num_ports) for _ in range(config.grid_shape[0])])
    engine = WavefrontEngine(config, layers)
    
    x_input = torch.randn(config.batch_size, config.grid_shape[1], config.dim)
    num_cells = math.prod(config.grid_shape)
    
    # Create a distinct list of grids for each port
    output_grids = [torch.randn(num_cells, config.batch_size, config.dim) for _ in range(config.num_ports)]
    
    # 2. Test coordinate (1, 2) which translates to flat index 6
    cell_idx = 6
    ingredients = fetch_mapped_context(x_input, output_grids, cell_idx, engine.routing_map, config)
    
    assert len(ingredients) == 2, "Did not fetch the correct number of dependencies"
    
    # 3. Rigorous Mathematical Assertions
    # Dependency 0 looks at ((-1, 0), Port 0). 
    # From coord (1, 2) this targets coord (0, 2) which is flat index 2.
    expected_tensor_0 = output_grids[0][2]
    assert torch.equal(ingredients[0], expected_tensor_0), "Fetcher pulled from the wrong spatial coordinate or port for dependency 0"
    
    # Dependency 1 looks at ((0, -1), Port 1).
    # From coord (1, 2) this targets coord (1, 1) which is flat index 5.
    expected_tensor_1 = output_grids[1][5]
    assert torch.equal(ingredients[1], expected_tensor_1), "Fetcher pulled from the wrong spatial coordinate or port for dependency 1"
    
    print("PASS")

def test_custom_autograd_equivalence():
    print("Running Test Custom Engine Autograd vs Pure PyTorch...", end=" ")
    
    config = WavefrontConfig(
        grid_shape=(2, 3), batch_size=2, dim=4, 
        dependencies=[((-1, 0), 0), ((0, -1), 1)], num_ports=2
    )
    
    # 1. Setup the Ground Truth and move everything to the GPU
    torch.manual_seed(42)
    device = 'cuda'
    
    x_base = torch.randn(config.batch_size, config.grid_shape[1], config.dim, device=device)
    layers_base = nn.ModuleList([DummyLayer(config.dim, len(config.dependencies), config.num_ports) for _ in range(config.grid_shape[0])]).to(device)
    
    x_pure = x_base.clone().requires_grad_(True)
    layers_pure = copy.deepcopy(layers_base)
    output_grids_pure = [[[None for _ in range(config.grid_shape[1])] for _ in range(config.grid_shape[0])] for _ in range(config.num_ports)]
    schedule = generate_wavefront_schedule(config)
    
    for tick_group in schedule:
        for coord in tick_group:
            l, t = coord
            fetched = [_test_fetch(config.grid_shape, x_pure, output_grids_pure, coord, dep) for dep in config.dependencies]
            out_tuple = layers_pure[l](*fetched)
            for port_idx in range(config.num_ports):
                output_grids_pure[port_idx][l][t] = out_tuple[port_idx]
            
    loss_pure = output_grids_pure[0][-1][-1].sum()
    loss_pure.backward()
    
    # 2. Setup Our Custom Engine
    x_custom = x_base.clone().requires_grad_(True)
    layers_custom = copy.deepcopy(layers_base)
    # Push the entire engine (including all buffers) to the GPU at once
    engine = WavefrontEngine(config, layers_custom).to(device)
    
    output_grids_custom = engine(x_custom)
    # The custom engine returns a list of flattened grids so we grab Port 0 and sum the final cell
    loss_custom = output_grids_custom[0][-1].sum() 
    loss_custom.backward()
    
    # 3. Compare the math
    assert torch.allclose(x_pure.grad, x_custom.grad, atol=1e-5), "Input gradients do not match"
    assert torch.allclose(layers_pure[0].linear.weight.grad, layers_custom[0].linear.weight.grad, atol=1e-5), "Weight gradients do not match"
    
    print("PASS")

def run_all_tests():
    print("=== Starting N-Dimensional Wavefront Test Suite ===")
    tests = [
        test_standard_2d_grid,
        test_custom_3d_video_grid,
        test_backprop_flow,
        test_mapped_fetcher,
        test_custom_autograd_equivalence
    ]
    
    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL\n  Assertion: {e}")
        except Exception as e:
            print(f"ERROR\n  Exception: {e}")
            import traceback
            traceback.print_exc()
            
    print("=============================================")
    print(f"Results: {passed}/{len(tests)} tests passed.")

if __name__ == "__main__":
    run_all_tests()