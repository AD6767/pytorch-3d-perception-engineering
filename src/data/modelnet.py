from collections.abc import Callable

import torch

import torch_geometric.transforms as transforms
from torch.utils.data import Dataset
from torch_geometric.datasets import ModelNet

from src.data.transforms import normalize_point_cloud


class ModelNetPointCloudDataset(Dataset):
    def __init__(self,
                 root: str,
                 train: bool,
                 num_points: int = 512,
                 transform: Callable[[torch.Tensor], torch.Tensor] | None = None) -> None:
        super().__init__()
        self.dataset = ModelNet(root=root,
                                name='10',
                                train=train,
                                pre_transform=transforms.SamplePoints(num=num_points)) # convert mesh to point cloud
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def get_label(self, index: int) -> int:
        return int(self.dataset[index].y.item())

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.dataset[index]
        points = data.pos.to(torch.float32) # (N, 3)
        label = data.y.squeeze(dim=0).to(torch.long) # scalar

        points = normalize_point_cloud(points=points)

        if self.transform is not None:
            points = self.transform(points)

        return points, label
