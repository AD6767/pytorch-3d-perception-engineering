from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader


def train_model(model: nn.Module,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                train_dataloader: DataLoader,
                device: torch.device,
                epochs: int,
                model_path: str | None = None,
                test_dataloader: DataLoader | None = None) -> tuple[List[float], List[float], List[float], List[float]]:
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_val_loss = float("inf")
    if test_dataloader is None:
        test_dataloader = train_dataloader
    for epoch in range(epochs):
        batch_loss, batch_acc = train_one_epoch(model=model,
                        criterion=criterion,
                        optimizer=optimizer,
                        dataloader=train_dataloader,
                        device=device)
        batch_val_loss, batch_val_acc = evaluate(model=model,
                                     criterion=criterion,
                                     dataloader=test_dataloader,
                                     device=device)
        if model_path is not None and batch_val_loss < best_val_loss:
            best_val_loss = batch_val_loss
            torch.save(model.state_dict(), model_path)
        train_loss.append(batch_loss)
        train_acc.append(batch_acc)
        val_loss.append(batch_val_loss)
        val_acc.append(batch_val_acc)
        print(f"Epoch: {epoch + 1} | train_loss: {batch_loss:.4f} | train_acc: {batch_acc * 100:.2f}%"
              f"| val_loss: {batch_val_loss:.4f} | val_acc: {batch_val_acc * 100:.2f}%")
    return train_loss, train_acc, val_loss, val_acc
    

def train_one_epoch(model: nn.Module,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    dataloader: DataLoader,
                    device: torch.device) -> tuple[float, float]:
    model.train()
    correct = 0
    total_samples = 0
    total_loss = 0.
    for batch_idx, (points, labels) in enumerate(dataloader):
        # labels (B,)
        optimizer.zero_grad()
        points = points.to(device)
        labels = labels.to(device)

        logits = model(points) # (B, 3)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_samples += labels.shape[0]
        probs = torch.softmax(logits, dim=1) # (B, 3)
        preds = torch.argmax(probs, dim=1) # (B,)
        correct += torch.sum(preds == labels).item()
        total_loss += loss.item() * labels.shape[0]

    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    return avg_loss, avg_acc

def evaluate(model: nn.Module,
             criterion: nn.Module,
             dataloader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    correct = 0
    total_samples = 0
    total_loss = 0.
    with torch.inference_mode():
        for points, labels in dataloader:
            # labels (B,)
            points = points.to(device)
            labels = labels.to(device)

            logits = model(points) # (B, 3)
            loss = criterion(logits, labels)

            total_samples += labels.shape[0]
            probs = torch.softmax(logits, dim=1) # (B, 3)
            preds = torch.argmax(probs, dim=1) # (B,)
            correct += torch.sum(preds == labels).item()
            total_loss += loss.item() * labels.shape[0]

    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    return avg_loss, avg_acc
