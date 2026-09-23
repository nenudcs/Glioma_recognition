from __future__ import annotations

import torch
from torch import nn


class Goal4Model(nn.Module):
    """Small 3-D multi-head classifier for structured Goal 4 output."""

    HEAD_SIZES = {
        "location": 15,
        "morphology": 2,
        "who_grade": 4,
        "enhancement": 1,
        "enhancement_pattern": 8,
        "necrosis": 1,
        "cystic_change": 1,
        "hemorrhage": 1,
        "calcification": 1,
        "margin_clear": 1,
        "lobulation": 1,
        "signal_t2wi": 3,
        "signal_flair": 3,
    }

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
        self.heads = nn.ModuleDict(
            {name: nn.Linear(16, size) for name, size in self.HEAD_SIZES.items()}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(x).mean(dim=(2, 3, 4))
        return {name: head(features) for name, head in self.heads.items()}
