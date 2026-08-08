import torch
from src.data.lidar import (filter_lidar_range, points_to_pillar_indices,
                            get_bev_grid_size)

def test_filter_lidar_range():
    x_range=(0.0, 70.0)
    y_range=(-20.0, 20.0)
    z_range=(-3.0, 3.0)
    points = torch.tensor([
        [10.0,  2.0,  0.5, 0.8],   # valid
        [15.0, -5.0,  1.0, 0.6],   # valid
        [-5.0,  1.0,  0.2, 0.9],   # behind ego
        [80.0,  0.0,  1.0, 0.4],   # too far
        [12.0, 30.0,  0.0, 0.7],   # outside lateral range
        [0.0, 19.0,  2.9, 0.7],    # valid 
    ])
    filtered_points = filter_lidar_range(points=points,
                                         x_range=x_range,
                                         y_range=y_range,
                                         z_range=z_range)
    assert filtered_points.shape == torch.Size([3, 4])
    assert torch.equal(filtered_points, torch.stack([points[0], points[1], points[5]], dim=0))

def test_points_to_pillar_indices() -> None:
    # filtered points
    points = torch.tensor([
        [12.3, -4.7, 0.5, 0.8],
        [0.1, -19.9, 1.0, 0.6],
        [69.9, 19.9, 0.0, 0.5],
    ])
    indices = points_to_pillar_indices(
        points=points,
        x_range=(0.0, 70.0),
        y_range=(-20.0, 20.0),
        pillar_size=(0.5, 0.5),
    )
    expected = torch.tensor([
        [24, 30],
        [0, 0],
        [139, 79],
    ])
    assert torch.equal(indices, expected)

def test_get_bev_grid_size():
    bev_x_size, bev_y_size = get_bev_grid_size(x_range=(0.0, 70.0),
                                               y_range=(-20.0, 20.0),
                                               pillar_size=(0.5, 0.5))
    assert bev_x_size == 140
    assert bev_y_size == 80


if __name__ == '__main__':
    test_filter_lidar_range()
    test_points_to_pillar_indices()
    test_get_bev_grid_size()
    print(f"All tests for LiDAR passed")
