import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """
    Encode a point cloud into one global feature vector.
    Input:
        points: [B, N, 3]
    Output:
        global_features: [B, feature_dim]
    """
    def __init__(self,
                 in_channels: int = 3,
                 feature_dim: int = 256):
        super().__init__()
        self.conv_mlp_block_1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                      out_channels=64,
                      kernel_size=1,
                      stride=1),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
        )
        self.conv_mlp_block_2 = nn.Sequential(
            nn.Conv1d(in_channels=64,
                        out_channels=128,
                        kernel_size=1,
                        stride=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU()
        )
        self.conv_mlp_block_3 = nn.Sequential(
            nn.Conv1d(in_channels=128,
                        out_channels=256,
                        kernel_size=1,
                        stride=1),
            nn.BatchNorm1d(num_features=feature_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected points with shape [B, N, 3], got {tuple(x.shape)}")

        if x.shape[-1] != 3:
            raise ValueError(f"Expected three coordinates per point, got shape {tuple(x.shape)}")
        # feature_dim = 256
        # # Conv1d expects [B, channels, sequence_length]
        # [B, N, 3] -> [B, 3, N]
        x = x.permute(0, 2, 1) # [B, 3, N]
        x = self.conv_mlp_block_3(self.conv_mlp_block_2(self.conv_mlp_block_1(x))) # [B, 256, N]
        global_features = torch.max(x, dim=2).values # (B, 256,)
        return global_features


class PointNetClassifier(nn.Module):
    """
    Simplified PointNet classifier.
    Input:
        points: [B, N, 3]
    Output:
        logits: [B, num_classes]
    """
    def __init__(self,
                 num_classes: int,
                 input_dim: int = 3,
                 feature_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels=input_dim,
                                       feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, 3]
        global_features = self.encoder(x) # extracted features [B, feature_dim]
        logits = self.classifier(global_features) # (B, 3)
        return logits
