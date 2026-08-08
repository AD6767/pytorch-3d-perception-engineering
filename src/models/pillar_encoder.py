import torch
import torch.nn as nn

""" Intuition:
pillar 0:
32 points
   ↓
shared MLP
   ↓
max pool
   ↓
64-D description

pillar 1:
18 points + padding
   ↓
shared MLP
   ↓
max pool
   ↓
64-D description
"""
class PillarEncoder(nn.Module):
    def __init__(self,
                 input_dim: int = 9,
                 feature_dim: int = 64):
        super().__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=feature_dim),
            nn.ReLU()
        )

    def forward(self,
                pillar_features: torch.Tensor,
                point_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pillar_features: [P, T, F]
            point_mask: [P, T]
        Returns:
            pillar_features: [P, C]
        """
        # (P, T, 9) -> (P, T, 64)
        x = self.shared_mlp(pillar_features)
        # point mask has `False` value for empty points
        x = x.masked_fill(~point_mask.unsqueeze(-1), # inserts a new dimension of size 1 at the very end
                          float("-inf"))
        x = torch.max(x, dim=1).values # (P, 64)
        return x
