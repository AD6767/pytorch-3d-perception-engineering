from numpy import long
import torch
import math

"""
Coordinate convention

LiDAR point:
    (x, y, z, intensity)

Assumed physical coordinates:
    +x -> forward
    +y -> left
    +z -> up

Pillar indexing:
    pillar_x comes from physical x
    pillar_y comes from physical y

BEV tensor indexing:
    bev[row, column]
    row    = pillar_y
    column = pillar_x

Important:
    Tensor row/column coordinates are not physical x/y coordinates.

    When visualizing with imshow:
        origin="upper" -> row index increases downward
        origin="lower" -> row index increases upward

    Use origin="lower" if we want the visualization to preserve
    the Cartesian +y-up convention.
"""


def filter_lidar_range(points: torch.Tensor,
                       x_range: tuple[float, float],
                       y_range: tuple[float, float],
                       z_range: tuple[float, float]) -> torch.Tensor:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    mask = ((x >= x_range[0]) & (x < x_range[1]) &
           (y >= y_range[0]) & (y < y_range[1]) &
           (z >= z_range[0]) & (z < z_range[1]))
    return points[mask]

def get_bev_grid_size(x_range: tuple[float, float],
                      y_range: tuple[float, float],
                      pillar_size: tuple[float, float]) -> tuple[long, long]:
    if pillar_size[0] <= 0 or pillar_size[1] <= 0:
        raise ValueError("pillar dimensions must be positive")
    x_length = x_range[1] - x_range[0]
    y_length = y_range[1] - y_range[0]
    bev_x_size = long(math.floor(x_length / pillar_size[0]))
    bev_y_size = long(math.floor(y_length / pillar_size[1]))
    return bev_x_size, bev_y_size

def points_to_pillar_indices(points: torch.Tensor,
                             x_range: tuple[float, float],
                             y_range: tuple[float, float],
                             pillar_size: tuple[float, float]) -> torch.Tensor:
    """Convert XY coordinates to discrete pillar indices.
    Args:
        points: LiDAR points with shape [N, 4+].
        x_range: Min/max x coordinates.
        y_range: Min/max y coordinates.
        pillar_size: Size of one pillar in x and y.
    Returns:
        Pillar indices with shape [N, 2]: [pillar_x, pillar_y].
    """
    # We are assuming the points are already passed through `filter_lidar_range()`
    if pillar_size[0] <= 0 or pillar_size[1] <= 0:
        raise ValueError("pillar dimensions must be positive")
    # points: (N, 4)
    x = points[:, 0] - x_range[0]
    y = points[:, 1] - y_range[0]
    pillar_x = torch.floor(x / pillar_size[0]).long() # (N,)
    pillar_y = torch.floor(y / pillar_size[1]).long() # (N,)
    pillar = torch.stack([pillar_x, pillar_y], dim=1) # (N, 2)
    return pillar