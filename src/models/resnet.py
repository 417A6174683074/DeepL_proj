import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Single residual block with skip connection"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU()
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.block(x) + x


class ResNet(nn.Module):
    def __init__(
        self,
        in_feats: int = 85,
        num_classes: int = 42,
        hidden_dim: int = 600,
        num_blocks: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()

        input_proj = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
        )

        self.model = nn.Sequential(
            input_proj,
            res_blocks,
            classifier,
        )

    def forward(self, x):
        return self.model(x)

    def __str__(self):
        return "ResNet"
