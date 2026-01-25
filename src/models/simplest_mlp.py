import torch
from torch import Tensor
import torch.nn as nn


class SimplestMLP(nn.Module):
    def __init__(self, in_features: int = 85, num_classes: int = 42, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits

    def __str__(self):
        return "SimplestMLP"
