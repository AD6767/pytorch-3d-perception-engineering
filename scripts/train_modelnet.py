import argparse

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from src.data.modelnet import ModelNetPointCloudDataset
from src.data.splits import train_val_split
from src.training.engine import train_model, evaluate
from src.models.pointnet import PointNetClassifier
from src.data.transforms import rotate_z_and_jitter, jitter_point_cloud

def get_transform(augmentation: str | None = None) -> Callable | None:
    if augmentation is None:
        return None
    if augmentation == "rotate-jitter":
        return rotate_z_and_jitter
    if augmentation == "random-jitter":
        return jitter_point_cloud
    raise ValueError(f"Unknown augmentation: {augmentation}")


def main(model_path: str,
         train_val_data_path: str,
         test_data_path: str,
         augmentation: str | None = None):
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cpu")
    print(f"Using device: {device}")
    transform = get_transform(augmentation=augmentation)
    print(f"Augmentation: {augmentation}")
    print(f"Transform: {transform}")
    # ------- dataset -------
    train_val_dataset = ModelNetPointCloudDataset(root=train_val_data_path,
                                                  train=True,
                                                  num_points=512,
                                                  transform=None)
    train_data, val_data = train_val_split(dataset=train_val_dataset,
                                           train_fraction=0.85,
                                           seed=seed)
    # get augmented train data (not for val)
    if transform is not None:
        print("Applying transformation to training data only")
        train_indices = train_data.indices
        augmented_train_data = ModelNetPointCloudDataset(root=train_val_data_path,
                                                         train=True,
                                                         num_points=512,
                                                         transform=transform)
        train_data = Subset(augmented_train_data, train_indices)
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=32,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_data,
                                batch_size=32,
                                shuffle=False)
    print("Train transform:", train_data.dataset.transform)
    print("Validation transform:", val_data.dataset.transform)
    # ----- model ------
    model = PointNetClassifier(num_classes=10,
                               input_dim=3,
                               feature_dim=256,
                               dropout=0.3)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Create model save path
    best_checkpoint_path = Path(model_path)
    best_checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    train_model(model=model, criterion=criterion, 
                optimizer=optimizer, 
                train_dataloader=train_dataloader, 
                device=device, 
                test_dataloader=val_dataloader, 
                epochs=20, 
                model_path=model_path)
    # --------- evaluate -----------
    model1 = PointNetClassifier(num_classes=10)
    model1.load_state_dict(torch.load(f=model_path, map_location=device, weights_only=True))
    model1.to(device)
    criterion = nn.CrossEntropyLoss()
    test_data = ModelNetPointCloudDataset(root=test_data_path,
                                          train=False,
                                          num_points=512,
                                          transform=transform)
    test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    test_loss, test_acc = evaluate(model=model1,
                                   criterion=criterion,
                                   dataloader=test_dataloader,
                                   device=device)
    print(f"ModelNet_10_512 | Test loss: {test_loss:.4f} | Final Test acc: {test_acc * 100:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # "data/modelnet10_points_512_test"
    parser.add_argument("--model_path", 
                        type=str, 
                        required=True,
                        help="Model Path: str")
    parser.add_argument("--train_val_data_path", 
                            type=str, 
                            required=True,
                            help="Train and Val data path: str")
    parser.add_argument("--test_data_path", 
                            type=str, 
                            required=True,
                            help="Test data path: str")
    parser.add_argument("--augmentation", 
                        type=str, 
                        required=False,
                        choices=["random-jitter", "rotate-jitter"],
                        default=None,
                        help="Select augmentation for training (default=None)")
    args = parser.parse_args()
    main(args.model_path, args.train_val_data_path, args.test_data_path, args.augmentation)
