import torch
import torch.nn as nn


class DecisionStep(nn.Module):
    def __init__(self, in_feats: int, hidden_dim: int = 800, dropout: float = 0.3, feat_selector_hidden_dim=64):
        super().__init__()

        self.feature_selector = nn.Sequential(
            nn.Linear(in_feats, feat_selector_hidden_dim),
            nn.ReLU(),
            nn.Linear(feat_selector_hidden_dim, in_feats),
            nn.Sigmoid(),
        )

        self.processor = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, prior_mask=None):
        mask = self.feature_selector(x)

        if prior_mask is not None:
            mask = mask * (1 - prior_mask)

        masked_x = x * mask
        processed = self.processor(masked_x)

        return processed, mask


class TabNet(nn.Module):
    def __init__(
        self,
        in_feats: int = 85,
        num_classes: int = 42,
        hidden_dim: int = 800,
        num_steps: int = 15,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.in_feats = in_feats

        self.input_proj = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
        )

        self.decision_steps = nn.ModuleList([DecisionStep(in_feats, hidden_dim, dropout) for _ in range(num_steps)])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        features = self.input_proj(x)

        prior_mask = torch.zeros(x.shape[0], self.in_feats, device=x.device)

        for step in self.decision_steps:
            step_output, mask = step(x, prior_mask)
            features = features + step_output
            prior_mask = prior_mask + mask

        return self.classifier(features)

    def __str__(self):
        return "TabNet"
