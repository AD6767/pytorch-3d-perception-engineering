from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.synthetic_shapes import SyntheticPointCloudDataset
from src.data.splits import split
from src.data.transforms import normalize_point_cloud
from src.models.pointnet import PointNetClassifier
from src.training.engine import train_model, evaluate


def train_wrapper(device: torch.device,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  model_path: str):
    model = PointNetClassifier(num_classes=3)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_model(model=model,
                criterion=criterion,
                optimizer=optimizer,
                train_dataloader=train_loader,
                device=device,
                epochs=20,
                model_path=model_path,
                test_dataloader=val_loader)


def train_and_evaluate(model_path: str,
                       transform=None):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    dataset = SyntheticPointCloudDataset(num_samples=300,
                                            points_per_shape=512,
                                            seed=42,
                                            transform=transform)
    train_data, val_data, test_data = split(dataset=dataset,
                                            train_fraction=0.7,
                                            val_fraction=0.15,
                                            seed=42)
    train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

    train_wrapper(device=device, 
                  train_loader=train_loader, 
                  val_loader=val_loader, 
                  model_path=model_path)

    # Load the best checkpoint model and test
    model = PointNetClassifier(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    test_loss, test_acc = evaluate(model=model, 
                                   criterion=criterion, 
                                   dataloader=test_loader, 
                                   device=device)
    print(f"Final Test loss: {test_loss:.4f} | Final Test acc: {test_acc * 100:.2f}%")

def main():
    checkpoint_path = Path("outputs/best_pointnet.pt")
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    train_and_evaluate(model_path=checkpoint_path.__str__(),
                       transform=normalize_point_cloud)

if __name__ == '__main__':
    main()
