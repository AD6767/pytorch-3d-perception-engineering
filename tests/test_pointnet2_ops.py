import torch

from src.models.pointnet2_ops import (
    farthest_point_sample,
    index_points,
    pairwise_squared_distance,
    query_ball_point,
    sample_and_group
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

def test_query_ball_point_on_line() -> None:
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
    centers = torch.tensor(
        [
            [
                [2.0],
            ]
        ]
    )
    indices = query_ball_point(
        points=points,
        centers=centers,
        radius=1.1,
        max_neighbors=3,
    )
    assert indices.shape == (1, 1, 3)
    selected_indices = set(indices[0, 0].tolist())
    assert selected_indices == {1, 2, 3}

def test_query_ball_point_pads_missing_neighbors() -> None:
    points = torch.tensor(
        [
            [
                [0.0],
                [2.0],
                [5.0],
            ]
        ]
    )
    centers = torch.tensor(
        [
            [
                [0.0],
            ]
        ]
    )
    indices = query_ball_point(
        points=points,
        centers=centers,
        radius=0.5,
        max_neighbors=3,
    )
    expected = torch.tensor(
        [
            [
                [0, 0, 0],
            ]
        ],
        dtype=torch.long,
    )
    assert torch.equal(indices, expected)

def test_fps_and_ball_query_integration() -> None:
    points = torch.randn(2, 32, 3)

    center_indices = farthest_point_sample(
        points,
        num_samples=4,
    )

    centers = index_points(
        points,
        center_indices,
    )

    neighbor_indices = query_ball_point(
        points=points,
        centers=centers,
        radius=0.5,
        max_neighbors=8,
    )

    grouped_points = index_points(
        points,
        neighbor_indices,
    )

    assert center_indices.shape == (2, 4)
    assert centers.shape == (2, 4, 3)
    assert neighbor_indices.shape == (2, 4, 8)
    assert grouped_points.shape == (2, 4, 8, 3)

def test_grouped_points_can_be_centered_locally() -> None:
    points = torch.randn(2, 32, 3)

    center_indices = farthest_point_sample(
        points,
        num_samples=4,
    )

    centers = index_points(
        points,
        center_indices,
    )

    neighbor_indices = query_ball_point(
        points=points,
        centers=centers,
        radius=0.5,
        max_neighbors=8,
    )

    grouped_points = index_points(
        points,
        neighbor_indices,
    )

    local_points = (
        grouped_points
        - centers.unsqueeze(dim=2)
    )

    assert local_points.shape == (2, 4, 8, 3)
    assert torch.isfinite(local_points).all()

def test_sample_and_group_coordinates_only() -> None:
    points = torch.randn(2, 32, 3)
    centers, grouped_features = sample_and_group(
        points=points,
        num_centers=4,
        radius=0.5,
        max_neighbors=8,
    )
    assert centers.shape == (2, 4, 3)
    assert grouped_features.shape == (2, 4, 8, 3)
    assert torch.isfinite(centers).all()
    assert torch.isfinite(grouped_features).all()

def test_sample_and_group_with_point_features() -> None:
    points = torch.randn(2, 32, 3)
    point_features = torch.randn(2, 32, 16)
    centers, grouped_features = sample_and_group(
        points=points,
        num_centers=4,
        radius=0.5,
        max_neighbors=8,
        point_features=point_features,
    )
    assert centers.shape == (2, 4, 3)
    assert grouped_features.shape == (2, 4, 8, 19)

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
    test_query_ball_point_on_line()
    test_query_ball_point_pads_missing_neighbors()
    test_fps_and_ball_query_integration()
    test_grouped_points_can_be_centered_locally()
    test_sample_and_group_coordinates_only()
    test_sample_and_group_with_point_features()
    print("All pointnet2_ops tests passed.")
