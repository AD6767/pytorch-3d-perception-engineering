import torch

from src.models.pointnet import PointNetClassifier, PointNetEncoder


def test_encoder_output_shape():
    points = torch.randn(size=(10, 512, 3)) # [B, N, 3]
    encoder = PointNetEncoder(in_channels=3, feature_dim=256)
    global_features = encoder(points)
    assert global_features.shape == torch.Size([10, 256])
    assert torch.isfinite(global_features).all()

def test_classifier_output_shape():
    points = torch.randn(size=(10, 512, 3)) # [B, N, 3]
    model = PointNetClassifier(num_classes=3,
                               input_dim=3,
                               feature_dim=256,
                               dropout=0.3)
    logits = model(points)
    assert logits.shape == torch.Size([10, 3])
    assert torch.isfinite(logits).all()

def test_model_supports_batch_size_one():
    points = torch.randn(1, 512, 3)
    model = PointNetClassifier(num_classes=3)
    model.eval()
    with torch.inference_mode():
        logits = model(points)

    assert logits.shape == (1, 3)

def test_model_supports_different_point_counts():
    points_128 = torch.randn(4, 128, 3)
    points_1024 = torch.randn(4, 1024, 3)
    model = PointNetClassifier(num_classes=3)

    logits_128 = model(points_128)
    logits_1024 = model(points_1024)

    assert logits_128.shape == (4, 3)
    assert logits_1024.shape == (4, 3)

if __name__ == "__main__":
    test_encoder_output_shape()
    test_classifier_output_shape()
    test_model_supports_batch_size_one()
    test_model_supports_different_point_counts()
    print("All PointNet tests passed.")
