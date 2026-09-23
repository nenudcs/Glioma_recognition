from __future__ import annotations

import torch
from torch import nn


class Goal3Model(nn.Module):
    """Small 3-D binary classifier; replace internals with the trained model."""

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 8, 3, padding=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x).mean(dim=(2, 3, 4))
        return self.head(features).squeeze(1)
