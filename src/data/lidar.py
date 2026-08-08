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
    pillar_x = long(math.floor(x_length / pillar_size[0]))
    pillar_y = long(math.floor(y_length / pillar_size[1]))
    return pillar_x, pillar_y

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

"""
Flow for understanding:
70,000 LiDAR points

       ↓ filtering

50,000 relevant points

       ↓ pillar indexing/grouping

8,000 occupied pillars

       ↓ cap at T=32

pillar_features:
[8000, 32, 9]
"""
def create_pillars(points: torch.Tensor,
                   pillar_indices: torch.Tensor,
                   x_range: tuple[float, float],
                   y_range: tuple[float, float],
                   pillar_size: tuple[float, float],
                   max_points_per_pillar: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert LiDAR points into fixed-size pillar tensors.
    Args:
        points:
            [N, 4] containing x, y, z, intensity.
        pillar_indices:
            [N, 2] containing [pillar_x, pillar_y].
    Returns:
        pillar_features:
            [P, T, 9]
        occupied_pillars:
            [P, 2]
        point_mask:
            [P, T]
    """
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError("points must have shape [N, 4+]")
    if pillar_indices.shape != (points.shape[0], 2):
        raise ValueError("pillar_indices must have shape [N, 2]")
    # Example
    # Input: tensor([[0, 0], [0, 0], [1, 0], [1, 1]])
    # occupied_pillars, inverse_indices: (tensor([[0, 0], [1, 0], [1, 1]]), tensor([0, 0, 1, 2]))
    occupied_pillars, inverse_indices = torch.unique(pillar_indices,
                                                     dim=0,
                                                     return_inverse=True) # (P, 2)
    P = occupied_pillars.shape[0]
    T = max_points_per_pillar
    pillar_features = torch.zeros(size=(P, T, 9), dtype=points.dtype, device=points.device)
    point_mask = torch.zeros(size=(P, T), dtype=torch.bool, device=points.device)

    x_size, y_size = pillar_size
    for pillar_id in range(P):
        # which point belongs to which pillar
        mask = pillar_id == inverse_indices # (N,)
        pillar_points = points[mask] # subset of the original N belonging to the current pillar
        # Limit the number of points
        pillar_points = pillar_points[:, :T]
        M = pillar_points.shape[0] # (M can be max T)
        xyz_points = pillar_points[:, :3] # (M, 3)
        # compute mean
        xyz_mean = torch.mean(xyz_points, dim=0, keepdim=True) # collapse rows (1, 3)
        cluster_offset = xyz_points - xyz_mean # (M, 3)
        pillar_x = occupied_pillars[pillar_id, 0]
        pillar_y = occupied_pillars[pillar_id, 1]
        pillar_center_x = x_range[0] + (pillar_x.float() + 0.5) * x_size
        pillar_center_y = y_range[0] + (pillar_y.float() + 0.5) * y_size
        center_offset = torch.stack(
            [pillar_points[:, 0] - pillar_center_x, 
             pillar_points[:, 1] - pillar_center_y], dim=1) # (M, 2)
        features = torch.cat([
            pillar_points[:, :4], # (M, 4)
            cluster_offset, # (M, 3)
            center_offset, # (M, 2)
        ], dim=1) # (M, 9) max (T, 9)
        pillar_features[pillar_id, :M] = features
        point_mask[pillar_id, :M] = True

    return pillar_features, occupied_pillars, point_mask
