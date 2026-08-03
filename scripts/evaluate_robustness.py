import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.data.synthetic_shapes import SyntheticPointCloudDataset
from src.data.splits import split
from src.data.transforms import (normalize_and_rotate_z_and_jitter, normalize_and_rotate_z, 
                                 normalize_and_rotate_3d, normalize_and_jitter)
from scripts.train import evaluate, train_and_evaluate

from src.models.pointnet import PointNetClassifier

from pathlib import Path

# evaluates clean/corrupted test sets
def evaluate_robustness():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    # Load the best checkpoint model
    model = PointNetClassifier(num_classes=3)
    model.load_state_dict(torch.load("outputs/augmented_best_pointnet.pth", 
                                     map_location=device, 
                                     weights_only=True))
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    dataset = SyntheticPointCloudDataset(num_samples=300,
                                            points_per_shape=512,
                                            seed=42,
                                            transform=normalize_and_rotate_z)
    _, _, test_data = split(dataset=dataset,
                                            train_fraction=0.7,
                                            val_fraction=0.15,
                                            seed=42)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    torch.manual_seed(123)
    test_loss, test_acc = evaluate(model=model, 
                                    criterion=criterion, 
                                    dataloader=test_loader, 
                                    device=device)
    print(f"normalize_and_rotate_z: Final Test loss: {test_loss:.4f} | Final Test acc: {test_acc * 100:.2f}%")
    dataset = SyntheticPointCloudDataset(num_samples=300,
                                            points_per_shape=512,
                                            seed=42,
                                            transform=normalize_and_rotate_3d)
    _, _, test_data = split(dataset=dataset,
                                            train_fraction=0.7,
                                            val_fraction=0.15,
                                            seed=42)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    torch.manual_seed(123)
    test_loss, test_acc = evaluate(model=model, 
                                    criterion=criterion, 
                                    dataloader=test_loader, 
                                    device=device)
    print(f"normalize_and_rotate_3d: Final Test loss: {test_loss:.4f} | Final Test acc: {test_acc * 100:.2f}%")
    dataset = SyntheticPointCloudDataset(num_samples=300,
                                            points_per_shape=512,
                                            seed=42,
                                            transform=normalize_and_jitter)
    _, _, test_data = split(dataset=dataset,
                                            train_fraction=0.7,
                                            val_fraction=0.15,
                                            seed=42)
    test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    torch.manual_seed(123)
    test_loss, test_acc = evaluate(model=model, 
                                    criterion=criterion, 
                                    dataloader=test_loader, 
                                    device=device)
    print(f"normalize_and_jitter: Final Test loss: {test_loss:.4f} | Final Test acc: {test_acc * 100:.2f}%")


def main():
    # checkpoint_path = Path("outputs/augmented_best_pointnet.pth")
    # checkpoint_path.parent.mkdir(
    #     parents=True,
    #     exist_ok=True,
    # )
    # train_and_evaluate(model_path="outputs/augmented_best_pointnet.pth",
    #                    transform=normalize_and_rotate_z_and_jitter)   
    evaluate_robustness()

if __name__ == '__main__':
    main()