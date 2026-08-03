import torch

from src.data.synthetic_shapes import SyntheticPointCloudDataset

dataset = SyntheticPointCloudDataset(num_samples=300,
                               points_per_shape=2048,
                               seed=42)

def test_dataset_length():
    assert len(dataset) == 300

def test_dataset_get_item_cube():
    points, label = dataset[4]
    assert label.item() == 1
    assert points.shape == torch.Size([2048, 3])
    points1, label1 = dataset[16]
    assert label1.item() == 1
    assert points1.shape == torch.Size([2048, 3])
    points2, label2 = dataset[4]
    assert label2.item() == 1
    assert points2.shape == torch.Size([2048, 3])
    # randomness working as expected
    assert points[0][0] == points2[0][0]
    assert points[0][1] == points2[0][1]
    assert points[0][2] == points2[0][2]

def test_label_is_stable() -> None:
    dataset = SyntheticPointCloudDataset(
        num_samples=300,
        points_per_shape=512,
        seed=42,
    )

    for index in range(len(dataset)):
        _, first_label = dataset[index]
        _, second_label = dataset[index]

        assert first_label.item() == second_label.item()

if __name__ == '__main__':
    test_dataset_length()
    test_dataset_get_item_cube()
    test_label_is_stable()
