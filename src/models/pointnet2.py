import torch
import torch.nn as nn

from src.models.pointnet2_ops import sample_and_group

"""
[B, N, 3]
   ↓ sample and group
[B, S, K, 3 + D]
   ↓ shared MLP
[B, D_out, S, K]
   ↓ max over K neighbors
[B, S, D_out]
"""

class PointNetSetAbstraction(nn.Module):
    """Sample centers, group neighbors, and aggregate local features."""
    def __init__(self, 
                 num_centers: int,
                 radius: float,
                 max_neighbors: int,
                 input_feature_dim: int,
                 mlp_channels: list[int]):
        super().__init__()
        self.num_centers = num_centers
        self.radius = radius
        self.max_neighbors = max_neighbors

        # Each neighborhood contains 3(XYZ) + optional input features
        input_channels = 3 + input_feature_dim

        layers: list[nn.Module] = []
        for out_channels in mlp_channels:
            layers.extend([
                nn.Conv2d(in_channels=input_channels,
                          out_channels=out_channels,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ])
            input_channels = out_channels

        self.shared_mlp = nn.Sequential(*layers)

    def forward(self, 
                points: torch.Tensor,
                point_features: torch.Tensor | None = None) -> tuple[torch.Tensor, 
                                                                     torch.Tensor]:
        """
        Args:
            points: XYZ coordinates [B, N, 3].
            point_features: Optional features [B, N, D].
        Returns:
            centers: Sampled center coordinates [B, S, 3].
            center_features: Aggregated features [B, S, D_out].
        """
        # (B, N, 3), (B, N, D) -> (B, S, 3), (B, S, K, C)
        centers, grouped_features = sample_and_group(points=points,
                                                     num_centers=self.num_centers,
                                                     radius=self.radius,
                                                     max_neighbors=self.max_neighbors,
                                                     point_features=point_features)
        # (B, S, K, C) -> (B, C, S, K)
        grouped_features = grouped_features.permute(dims=(0, 3, 1, 2))
        local_features = self.shared_mlp(grouped_features) # (B, D_out, S, K)
        # Aggregate all K neighbors for all centers
        center_features = torch.max(local_features, dim=-1).values # (B, D_out, S)
        center_features = center_features.permute(dims=(0, 2, 1)) # (B, S, D_out)
        return centers, center_features


class PointNet2Classifier(nn.Module):
    """Small PointNet++ classifier for ModelNet10."""
    def __init__(self,
                 num_classes: int = 10,
                 dropout: float = 0.3):
        super().__init__()

        self.sa1 = PointNetSetAbstraction(num_centers=128,
                                          radius=0.2,
                                          max_neighbors=32,
                                          input_feature_dim=0,
                                          mlp_channels=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(num_centers=32,
                                          radius=0.4,
                                          max_neighbors=32,
                                          input_feature_dim=128,
                                          mlp_channels=[128, 128, 256])
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        centers1, features1 = self.sa1(points=points,
                                       point_features=None) # (B, 128, 128)
        _, features2 = self.sa2(points=centers1,
                                point_features=features1) # (B, 32, 256)
        # global aggregation over remaining features
        global_features = torch.max(features2, dim=1).values # (B, 256)
        out = self.classifier(global_features) # (B, num_classes)
        return out