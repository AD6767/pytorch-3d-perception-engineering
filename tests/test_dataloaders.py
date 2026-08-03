import torch
from torch.utils.data import DataLoader

from src.data.splits import split
from src.data.synthetic_shapes import SyntheticPointCloudDataset


def test_dataloader_batch_shapes() -> None:
    dataset = SyntheticPointCloudDataset(
        num_samples=300,
        points_per_shape=512,
        seed=42,
    )

    train_dataset, _, _ = split(
        dataset=dataset,
        train_fraction=0.7,
        val_fraction=0.15,
        seed=42,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )

    points, labels = next(iter(train_loader))

    assert points.shape == (16, 512, 3)
    assert labels.shape == (16,)
    assert points.dtype == torch.float32
    assert labels.dtype == torch.long
    assert torch.isfinite(points).all()


if __name__ == "__main__":
    test_dataloader_batch_shapes()
    print("DataLoader test passed.")