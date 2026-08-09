import torch
import torch.nn as nn

""" Intuition:
irregular LiDAR points
        ↓
pillars
        ↓
one vector per pillar
        ↓
scatter to BEV
        ↓
[C, H, W]
"""
class PillarScatter(nn.Module):
    def __init__(self,
                 num_x: int,
                 num_y: int):
        super().__init__()
        self.W = num_x
        self.H = num_y

    def forward(self,
                pillar_features: torch.Tensor,
                pillar_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pillar_features: [P, C]
            pillar_indices: [P, 2] as [pillar_x, pillar_y]
        Returns:
            bev: [C, H, W]
                 where H = num_y and W = num_x
        """
        P, C = pillar_features.shape
        bev = torch.zeros(size=(C, self.H, self.W), dtype=pillar_features.dtype, device=pillar_features.device)
        # for pillar_idx in range(P):
        #     pillar_x, pillar_y = pillar_indices[pillar_idx] # (width, height)
        #     bev[:, pillar_y, pillar_x] = pillar_features[pillar_idx]
        pillar_x, pillar_y = pillar_indices[:, 0], pillar_indices[:, 1] # (P,)
        # LHS: selects P grid cells for every channel, producing shape (C, P)
        bev[:, pillar_y, pillar_x] = pillar_features.T # (RHS: transpose-> [C, P])
        return bev # (C, H, W)