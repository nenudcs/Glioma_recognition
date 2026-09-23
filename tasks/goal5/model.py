from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Goal5Model(nn.Module):
    """Small two-channel 3-D segmenter for core and peripheral abnormality."""

    def __init__(self, in_channels: int = 4, out_channels: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 8, 3, padding=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.InstanceNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(16, 8, 3, padding=1),
            nn.InstanceNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decoder(self.encoder(x))
        return F.interpolate(logits, size=x.shape[2:], mode="trilinear", align_corners=False)
