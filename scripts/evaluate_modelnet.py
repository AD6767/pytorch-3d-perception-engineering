import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.modelnet import ModelNetPointCloudDataset
from src.models.pointnet import PointNetClassifier
from scripts.train_modelnet import get_transform


CLASS_NAMES = [
    "bathtub",
    "bed",
    "chair",
    "desk",
    "dresser",
    "monitor",
    "night_stand",
    "sofa",
    "table",
    "toilet",
]

def evaluate_per_class(model: nn.Module,
                       dataloader: DataLoader,
                       num_classes: int,
                       device: torch.device) -> tuple[float, 
                                                      torch.Tensor, 
                                                      torch.Tensor]:
    model.eval()

    confusion_matrix = torch.zeros(size=(num_classes, num_classes),
                                   dtype=torch.long) # (10x10)
    with torch.inference_mode():
        for points, labels in dataloader:
            points = points.to(device)
            labels = labels.to(device) # (B,)
            logits = model(points) # (B, 10)
            preds = torch.argmax(logits, dim=1) # (B,)
            for idx in range(labels.shape[0]):
                gt_label = int(labels[idx].item())
                pred_label = int(preds[idx].item())
                confusion_matrix[gt_label][pred_label] += 1
    # along the diagonal gt == pred
    correct_per_class = torch.diag(confusion_matrix) # (10,)
    total_per_class = torch.sum(confusion_matrix, dim=1) # (10,)
    per_class_accuracy = correct_per_class / torch.clamp(total_per_class, min=1).float()
    # total accuracy
    total_correct = torch.sum(torch.diag(confusion_matrix)).item()
    total_samples = torch.sum(confusion_matrix).item()
    total_accuracy = (1.0 * total_correct) / total_samples
    return total_accuracy, per_class_accuracy, confusion_matrix

def evaluate_modelnet(model: nn.Module,
                      dataloader: DataLoader,
                      num_classes: int,
                      device: torch.device):
    overall_accuracy, per_class_accuracy, confusion_matrix = evaluate_per_class(model=model,
                                                                                dataloader=dataloader,
                                                                                device=device,
                                                                                num_classes=num_classes)
    print(f"Overall test accuracy: {overall_accuracy * 100:.2f}%")
    print("\nPer-class accuracy:")
    for class_index, class_name in enumerate(CLASS_NAMES):
        num_correct = confusion_matrix[class_index, class_index].item()
        num_samples = confusion_matrix[class_index].sum().item()
        print(f"{class_index:2d} {class_name:12s}: "
              f"{per_class_accuracy[class_index] * 100:6.2f}% "
              f"({num_correct}/{num_samples})")
    print("\nConfusion matrix:")
    print(confusion_matrix)

def main(model_path: str,
         data_path: str,
         augmentation: str | None = None):
    device = torch.device("cpu")
    print(f"Using device: {device}")
    transform = get_transform(augmentation=augmentation)
    print(f"Augmentation: {augmentation}")
    print(f"Transform: {transform}")
    seed = 42
    torch.manual_seed(seed)
    # --------- evaluate -----------
    model = PointNetClassifier(num_classes=10)
    model_path_ = Path(model_path)
    print(f"Loading checkpoint: {model_path_.resolve()}")
    model.load_state_dict(torch.load(f=model_path_, map_location=device, weights_only=True))
    model.to(device)
    test_data = ModelNetPointCloudDataset(root=data_path,
                                          train=False,
                                          num_points=512,
                                          transform=transform)
    test_dataloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)
    evaluate_modelnet(model=model,
                      dataloader=test_dataloader,
                      num_classes=10,
                      device=device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # "data/modelnet10_points_512_test"
    parser.add_argument("--model_path", 
                        type=str, 
                        required=True,
                        help="Model Path: str")
    parser.add_argument("--data_path", 
                            type=str, 
                            required=True,
                            help="Data path: str")
    parser.add_argument("--augmentation", 
                        type=str, 
                        required=False,
                        choices=["random-jitter", "rotate-jitter"],
                        default=None,
                        help="Select augmentation for training (default=None)")
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.augmentation)
