import torch
from torch.utils.data import DataLoader

from src.data.modelnet import ModelNetPointCloudDataset


def test_modelnet_sample():
    dataset = ModelNetPointCloudDataset(root="data/modelnet10_points_512",
                                        train=True,
                                        num_points=512)
    points, label = dataset[0]
    assert points.shape == (512, 3)
    assert points.dtype == torch.float32
    assert label.shape == ()
    assert label.dtype == torch.long
    assert 0 <= label.item() < 10
    assert torch.isfinite(points).all()


def test_modelnet_normalization() -> None:
    dataset = ModelNetPointCloudDataset(root="data/modelnet10_points_512",
                                        train=True,
                                        num_points=512)
    points, _ = dataset[0]
    assert torch.allclose(points.mean(dim=0), torch.zeros(3), atol=1e-5)
    max_radius = torch.linalg.vector_norm(points, dim=1).max()
    assert torch.allclose(max_radius, torch.tensor(1.0), atol=1e-5)


def test_modelnet_dataloader() -> None:
    dataset = ModelNetPointCloudDataset(root="data/modelnet10_points_512",
                                        train=True,
                                        num_points=512)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    points, labels = next(iter(loader))
    assert points.shape == (32, 512, 3)
    assert labels.shape == (32,)