import dataclasses
from typing import List, Tuple
import itertools

@dataclasses.dataclass
class WavefrontConfig:
    grid_shape: Tuple[int, ...] 
    batch_size: int
    dim: int = 256
    
    # Dependencies now expect a tuple of (spatial_offset, port_index)
    dependencies: List[Tuple[Tuple[int, ...], int]] = dataclasses.field(
        default_factory=lambda: [((-1, 0), 0), ((0, -1), 1)]
    )
    
    # Tell the engine how many distinct output grids to build
    num_ports: int = 2
    requires_grad: bool = True

def generate_wavefront_schedule(config: WavefrontConfig) -> List[List[Tuple[int, ...]]]:
    """
    Calculates the execution order for any N-dimensional grid using recursive Depth-First Search.
    This safely handles complex dependencies like downward biological feedback.
    """
    tick_map = {}
    
    def get_tick(coord):
        # If we already calculated this cell's tick, just return it
        if coord in tick_map:
            return tick_map[coord]
            
        max_dep_tick = -1
        
        # Look at every ingredient this cell needs
        for dep in config.dependencies:
            spatial_offset, _ = dep
            dep_coord = tuple(c + d for c, d in zip(coord, spatial_offset))
            
            # Check if that address is actually inside the boundaries of our map
            is_valid = all(0 <= dep_coord[i] < config.grid_shape[i] for i in range(len(coord)))
            
            if is_valid:
                # Recursively fetch the tick of the dependency
                # If it hasn't been calculated yet, this will go calculate it on the fly
                max_dep_tick = max(max_dep_tick, get_tick(dep_coord))
                
        # This cell runs exactly one tick after its slowest dependency finishes
        my_tick = max_dep_tick + 1
        tick_map[coord] = my_tick
        return my_tick

    # Generate every coordinate in our grid
    ranges = [range(size) for size in config.grid_shape]
    all_coords = itertools.product(*ranges)
    
    # Calculate the tick for every coordinate using our smart recursive function
    for coord in all_coords:
        get_tick(coord)
        
    # Group the coordinates by their tick
    schedule = {}
    for coord, tick in tick_map.items():
        if tick not in schedule:
            schedule[tick] = []
        schedule[tick].append(coord)
        
    # Sort the schedule so tick 0 comes first and return the grouped coordinates
    sorted_ticks = sorted(schedule.keys())
    return [schedule[tick] for tick in sorted_ticks]