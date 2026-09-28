from __future__ import annotations

import torch
from torch import nn
import os

from inference_backends.medicalnet import MedicalNet3DClassifier


class Goal3Model(nn.Module):
    """Small 3-D binary classifier; replace internals with the trained model."""

    def __init__(
        self,
        in_channels: int = 1,
        *,
        backend: str | None = None,
        checkpoint: str | None = None,
        medicalnet_depth: int = 50,
    ) -> None:
        super().__init__()
        selected = (backend or os.environ.get("GOAL3_BACKEND", "tiny")).lower()
        if selected == "medicalnet":
            self.backend = "medicalnet"
            self.medicalnet = MedicalNet3DClassifier(
                depth=int(os.environ.get("GOAL3_MEDICALNET_DEPTH", medicalnet_depth)),
                in_channels=in_channels,
                num_classes=1,
                checkpoint=checkpoint or os.environ.get("GOAL3_CHECKPOINT"),
                strict=os.environ.get("GOAL3_STRICT_CHECKPOINT", "0").lower() in {"1", "true", "yes"},
            )
            self.features = None
            self.head = None
            return
        if selected != "tiny":
            raise ValueError(f"unsupported Goal3 backend: {selected!r}; use tiny or medicalnet")
        self.backend = "tiny"
        self.medicalnet = None
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
        if self.backend == "medicalnet":
            return self.medicalnet(x).reshape(x.shape[0], -1)[:, 0]
        features = self.features(x).mean(dim=(2, 3, 4))
        return self.head(features).squeeze(1)
