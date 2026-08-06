import torch

from src.models.pointnet2_ops import (
    index_points,
    pairwise_squared_distance,
)
from src.models.pointnet2_ops import (
    farthest_point_sample,
    index_points,
    pairwise_squared_distance,
)


def test_pairwise_squared_distance_values() -> None:
    source = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ]
        ]
    )
    distances = pairwise_squared_distance(
        source,
        target,
    )
    expected = torch.tensor(
        [
            [
                [0.0, 4.0],
                [1.0, 5.0],
            ]
        ]
    )
    assert distances.shape == (1, 2, 2)
    assert torch.allclose(distances, expected)


def test_pairwise_distance_to_self() -> None:
    points = torch.randn(2, 8, 3)
    distances = pairwise_squared_distance(
        points,
        points,
    )
    assert distances.shape == (2, 8, 8)
    assert torch.allclose(
        distances,
        distances.transpose(1, 2),
        atol=1e-6,
    )
    diagonal = distances.diagonal(
        dim1=1,
        dim2=2,
    )
    assert torch.allclose(
        diagonal,
        torch.zeros_like(diagonal),
        atol=1e-6,
    )


def test_index_points_with_2d_indices() -> None:
    points = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
            ],
            [
                [10.0, 10.0],
                [11.0, 11.0],
                [12.0, 12.0],
                [13.0, 13.0],
            ],
        ]
    )
    indices = torch.tensor(
        [
            [0, 2],
            [1, 3],
        ],
        dtype=torch.long,
    )
    selected = index_points(points, indices)
    expected = torch.tensor(
        [
            [
                [0.0, 0.0],
                [2.0, 2.0],
            ],
            [
                [11.0, 11.0],
                [13.0, 13.0],
            ],
        ]
    )
    assert selected.shape == (2, 2, 2)
    assert torch.equal(selected, expected)


def test_index_points_with_neighborhood_indices() -> None:
    points = torch.randn(2, 10, 3)
    indices = torch.tensor(
        [
            [
                [0, 1, 2],
                [3, 4, 5],
            ],
            [
                [1, 3, 5],
                [2, 4, 6],
            ],
        ],
        dtype=torch.long,
    )
    selected = index_points(points, indices)
    assert selected.shape == (2, 2, 3, 3)
    assert torch.equal(
        selected[0, 0],
        points[0, torch.tensor([0, 1, 2])],
    )
    assert torch.equal(
        selected[1, 1],
        points[1, torch.tensor([2, 4, 6])],
    )


def test_pairwise_squared_distance_backward() -> None:
    source = torch.randn(
        2,
        8,
        3,
        requires_grad=True,
    )
    target = torch.randn(
        2,
        4,
        3,
        requires_grad=True,
    )
    distances = pairwise_squared_distance(
        source,
        target,
    )
    loss = distances.mean()
    loss.backward()
    assert source.grad is not None
    assert target.grad is not None
    assert torch.isfinite(source.grad).all()
    assert torch.isfinite(target.grad).all()

def test_farthest_point_sample_shape() -> None:
    points = torch.randn(2, 16, 3) # (B, N, C)
    indices = farthest_point_sample(points, num_samples=4)
    assert indices.shape == (2, 4)
    assert indices.dtype == torch.long
    assert indices.min() >= 0
    assert indices.max() < 16

def test_farthest_point_sample_on_line() -> None:
    points = torch.tensor(
        [
            [
                [0.0],
                [1.0],
                [2.0],
                [3.0],
                [4.0],
            ]
        ]
    )
    indices = farthest_point_sample(points, num_samples=3)
    expected = torch.tensor(
        [[0, 4, 2]],
        dtype=torch.long,
    )
    assert torch.equal(indices, expected)


def test_farthest_point_sample_has_unique_indices() -> None:
    points = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
            ]
        ]
    )
    indices = farthest_point_sample(
        points,
        num_samples=5,
    )
    assert torch.unique(indices[0]).numel() == 5


def test_index_farthest_sampled_points() -> None:
    points = torch.randn(2, 16, 3)
    indices = farthest_point_sample(
        points,
        num_samples=4,
    )
    sampled_points = index_points(
        points,
        indices,
    )
    assert sampled_points.shape == (2, 4, 3)

if __name__ == "__main__":
    test_pairwise_squared_distance_values()
    test_pairwise_distance_to_self()
    test_index_points_with_2d_indices()
    test_index_points_with_neighborhood_indices()
    test_pairwise_squared_distance_backward()
    test_farthest_point_sample_shape()
    test_farthest_point_sample_on_line()
    test_farthest_point_sample_has_unique_indices()
    test_index_farthest_sampled_points()
