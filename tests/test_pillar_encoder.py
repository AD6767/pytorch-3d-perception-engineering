import torch

from src.models.pillar_encoder import PillarEncoder


def test_pillar_encoder_shape():
    features = torch.randn(
        5,   # pillars
        32,  # max points
        9,   # input features
    )
    mask = torch.ones(5, 32, dtype=torch.bool)
    model = PillarEncoder(input_dim=9, feature_dim=64)
    output = model(features, mask)
    assert output.shape == (5, 64)

def test_pillar_encoder_with_padding():
    features = torch.randn(2, 4, 9)
    mask = torch.tensor([
        [True, True, False, False],
        [True, True, True, False],
    ])
    model = PillarEncoder(input_dim=9, feature_dim=64)
    output = model(features, mask)
    assert output.shape == (2, 64)
    assert torch.isfinite(output).all()

if __name__ == '__main__':
    test_pillar_encoder_shape()
    test_pillar_encoder_with_padding()
    print(f"All tests for Pillar Encoder passed")