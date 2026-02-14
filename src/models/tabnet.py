import torch
import torch.nn as nn


class FeatureSelector(nn.Module):
    """Learns a mask to select which features to use"""

    def __init__(self, in_feats: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_feats),
        )

    def forward(self, x):
        # Returns soft mask (probabilities) over features
        mask = torch.sigmoid(self.net(x))
        return mask


class DecisionStep(nn.Module):
    """Single decision step: select features and process them"""

    def __init__(self, in_feats: int, hidden_dim: int = 600, dropout: float = 0.4):
        super().__init__()

        self.feature_selector = FeatureSelector(in_feats, hidden_dim=64)

        self.processor = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, prior_mask=None):
        # Feature selection
        mask = self.feature_selector(x)

        # Apply mask and process selected features
        if prior_mask is not None:
            mask = mask * (1 - prior_mask)  # Mask out previously selected features

        masked_x = x * mask
        processed = self.processor(masked_x)

        return processed, mask


class TabNet(nn.Module):
    def __init__(
        self,
        in_feats: int = 85,
        num_classes: int = 42,
        hidden_dim: int = 600,
        num_steps: int = 8,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.in_feats = in_feats

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # Decision steps (similar to residual blocks)
        self.decision_steps = nn.ModuleList([DecisionStep(in_feats, hidden_dim, dropout) for _ in range(num_steps)])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # Initial projection
        features = self.input_proj(x)

        # Aggregate decisions across steps
        prior_mask = torch.zeros(x.shape[0], self.in_feats, device=x.device)

        for step in self.decision_steps:
            step_output, mask = step(x, prior_mask)
            features = features + step_output  # Aggregate like residual connections
            prior_mask = prior_mask + mask

        # Final classification
        return self.classifier(features)

    def __str__(self):
        return "TabNet"
