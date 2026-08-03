import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from scripts.train import train_model
from src.data.synthetic_shapes import SyntheticPointCloudDataset
from src.data.transforms import normalize_point_cloud
from src.data.splits import split
from src.models.pointnet import PointNetClassifier

"""
Tiny dataset size: 24 samples
Classes:           8 sphere, 8 cube, 8 cylinder
Batch size:        24
Shuffle:           true
Epochs:            200-500
Optimizer:         Adam
Learning rate:     1e-3
Weight decay:      0
Augmentation:      none
Dropout:           0
"""

def main():
    device = torch.device("cpu")
    print(f"Using device {device}")
    # random seed
    seed = 42
    torch.manual_seed(seed)
    # create synthetic dataset
    dataset = SyntheticPointCloudDataset(num_samples=30,
                                         points_per_shape=512,
                                         seed=seed,
                                         transform=normalize_point_cloud)
    train_data, val_data, test_data = split(dataset=dataset,
                                            train_fraction=0.8,
                                            val_fraction=0.1,
                                            seed=seed)
    print(f"Train Data size: {len(train_data)}")
    print(f"Validation Data size: {len(val_data)}")
    print(f"Test Data size: {len(test_data)}")
    # create data loaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=24, shuffle=True)
    val_dataloader = DataLoader(dataset=val_data, batch_size=24, shuffle=False)
    test_dataloader = DataLoader(dataset=test_data, batch_size=24, shuffle=False)
    # create model
    model = PointNetClassifier(num_classes=3, dropout=0.)
    model.to(device)
    # Loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)
    train_loss, train_acc, test_loss, test_acc = train_model(model=model,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        train_dataloader=train_dataloader,
                                        device=device,
                                        model_path=None,
                                        epochs=50)


if __name__ == '__main__':
    main()
