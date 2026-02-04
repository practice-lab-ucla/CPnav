import math
from typing import Tuple

def world_to_grid_00(pos: Tuple[float, float], grid_spacing_m: float) -> Tuple[int, int]:
    """
    Map world (x, z) in meters -> 0-based grid cell (row, col).
    Simple tiling: floor(coord / grid_spacing_m).
    """
    x, z = float(pos[0]), float(pos[1])
    if grid_spacing_m <= 0:
        raise ValueError("grid_spacing_m must be > 0")
    col = math.floor(x / grid_spacing_m)
    row = math.floor(z / grid_spacing_m)
    return int(row), int(col)