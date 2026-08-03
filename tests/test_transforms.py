import torch

from src.data.synthetic_shapes import SyntheticPointCloudDataset, CLASS_IDX_TO_LABEL
from src.data.transforms import normalize_point_cloud

def test_dataset_transform():
    raw_dataset = SyntheticPointCloudDataset(num_samples=300,
                                          points_per_shape=512,
                                          seed=42,
                                          transform=None)
    raw_points, raw_label = raw_dataset[7]
    normalized_dataset = SyntheticPointCloudDataset(num_samples=300,
                                              points_per_shape=512,
                                              seed=42,
                                              transform=normalize_point_cloud)
    normalized_points, normalized_label = normalized_dataset[7]
    # print(f"Raw dataset centroid: {raw_points.mean(dim=0)}, raw_label: {CLASS_IDX_TO_LABEL[int(raw_label.item())]}")
    # print(f"Normalized centroid: {normalized_points.mean(dim=0)}, normalized_label: {CLASS_IDX_TO_LABEL[int(normalized_label.item())]}")
    assert raw_points.shape == normalized_points.shape
    assert raw_label == normalized_label

    assert torch.allclose(
        normalized_points.mean(dim=0),
        torch.zeros(3),
        atol=1e-6,
    )
    assert torch.isfinite(normalized_points).all()


if __name__ == '__main__':
    test_dataset_transform()
