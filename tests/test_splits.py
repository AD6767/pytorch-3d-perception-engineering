from src.data.splits import split
from src.data.synthetic_shapes import SyntheticPointCloudDataset
from torch.utils.data import Dataset

from collections import Counter

def count_labels(dataset: Dataset) -> Counter:
    return Counter(
        int(dataset[index][1].item())
        for index in range(len(dataset))
    )

def test_dataset_split() -> None:
    dataset = SyntheticPointCloudDataset(
        num_samples=300,
        points_per_shape=512,
        seed=42,
    )

    train_data, val_data, test_data = split(
        dataset=dataset,
        train_fraction=0.7,
        val_fraction=0.15,
        seed=42,
    )

    train_counter = count_labels(train_data)
    val_counter = count_labels(val_data)
    test_counter = count_labels(test_data)

    print("Train classes:", train_counter)
    print("Validation classes:", val_counter)
    print("Test classes:", test_counter)

    assert train_counter == Counter({
        0: 70,
        1: 70,
        2: 70,
    })

    assert val_counter == Counter({
        0: 15,
        1: 15,
        2: 15,
    })

    assert test_counter == Counter({
        0: 15,
        1: 15,
        2: 15,
    })


def test_dataset_is_balanced() -> None:
    dataset = SyntheticPointCloudDataset(
        num_samples=300,
        points_per_shape=512,
        seed=42,
    )

    labels = [
        dataset.get_label(index)
        for index in range(len(dataset))
    ]

    assert Counter(labels) == Counter({
        0: 100,
        1: 100,
        2: 100,
    })


if __name__ == '__main__':
    test_dataset_is_balanced()
    test_dataset_split()

    print("All split tests passed.")
