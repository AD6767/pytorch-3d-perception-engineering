from collections import defaultdict

import torch
from torch.utils.data import Dataset, Subset

# Dataset is an abstract base class used to define your data source
# Subset is a utility class used to slice or extract specific indices from an existing dataset

def split(dataset: Dataset,
          train_fraction: float = 0.7,
          val_fraction: float = 0.15,
          seed: int | None = None) -> tuple[Subset, Subset, Subset]:
    """
    Split a classification dataset while preserving class proportions.
    The dataset must return: points, label
    Args:
        dataset: Full PyTorch dataset.
        train_fraction: Fraction assigned to training.
        validation_fraction: Fraction assigned to validation.
        seed: Random seed used when shuffling indices within each class.
    Returns:
        train_dataset, validation_dataset, test_dataset
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    indices_by_class = defaultdict(list) # Initialize like this instead of {}
    # Group indices by class
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        indices_by_class[label.item()].append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for class_label in indices_by_class.keys():
        N = len(indices_by_class[class_label])
        permutation = torch.randperm(N, generator=generator) # [0, N-1]
        random_indices = [
            indices_by_class[class_label][position]
            for position in permutation
        ]

        N_train = int(train_fraction * N)
        N_val = int(val_fraction * N)
        # N_test = N - N_train - N_val
        train_indices.extend(random_indices[:N_train])
        val_indices.extend(random_indices[N_train: N_train + N_val])
        test_indices.extend(random_indices[N_train + N_val:])

    train_dataset = Subset(dataset=dataset, indices=train_indices)
    val_dataset = Subset(dataset=dataset, indices=val_indices)
    test_dataset = Subset(dataset=dataset, indices=test_indices)
    return train_dataset, val_dataset, test_dataset