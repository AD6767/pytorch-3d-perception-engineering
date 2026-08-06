import torch

from src.models.pointnet2 import (
    PointNet2Classifier,
    PointNetSetAbstraction,
)


def test_set_abstraction_without_features() -> None:
    layer = PointNetSetAbstraction(
        num_centers=8,
        radius=0.5,
        max_neighbors=16,
        input_feature_dim=0,
        mlp_channels=[32, 64],
    )
    points = torch.randn(2, 64, 3)
    centers, features = layer(points)
    assert centers.shape == (2, 8, 3)
    assert features.shape == (2, 8, 64)


def test_set_abstraction_with_features() -> None:
    layer = PointNetSetAbstraction(
        num_centers=8,
        radius=0.5,
        max_neighbors=16,
        input_feature_dim=32,
        mlp_channels=[64, 128],
    )
    points = torch.randn(2, 64, 3)
    point_features = torch.randn(2, 64, 32)
    centers, features = layer(
        points,
        point_features,
    )
    assert centers.shape == (2, 8, 3)
    assert features.shape == (2, 8, 128)


def test_pointnet2_classifier_output() -> None:
    model = PointNet2Classifier(
        num_classes=10,
        dropout=0.0,
    )
    points = torch.randn(2, 512, 3)
    logits = model(points)
    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()


def test_pointnet2_backward() -> None:
    model = PointNet2Classifier(
        num_classes=10,
        dropout=0.0,
    )
    points = torch.randn(2, 512, 3)
    labels = torch.tensor([1, 4])
    logits = model(points)
    loss = torch.nn.functional.cross_entropy(
        logits,
        labels,
    )
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    assert all(
        gradient is not None
        for gradient in gradients
    )
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
    )

if __name__ == "__main__":
    test_set_abstraction_without_features()
    test_set_abstraction_with_features()
    test_pointnet2_classifier_output()
    test_pointnet2_backward()
    print("All PointNet2 tests passed.")
