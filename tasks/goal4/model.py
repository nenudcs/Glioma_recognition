from __future__ import annotations

import torch
from torch import nn
import os

from inference_backends.medicalnet import MedicalNet3DEncoder


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

    def __init__(
        self,
        in_channels: int = 1,
        *,
        backend: str | None = None,
        checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        selected = (backend or os.environ.get("GOAL4_BACKEND", "tiny")).lower()
        if selected == "medicalnet":
            self.backend = "medicalnet"
            self.encoder = MedicalNet3DEncoder(
                depth=int(os.environ.get("GOAL4_MEDICALNET_DEPTH", "50")),
                in_channels=in_channels,
                checkpoint=checkpoint or os.environ.get("GOAL4_CHECKPOINT"),
                strict=os.environ.get("GOAL4_STRICT_CHECKPOINT", "0").lower() in {"1", "true", "yes"},
            )
            feature_dim = self.encoder.feature_dim
            self.heads = nn.ModuleDict(
                {name: nn.Linear(feature_dim, size) for name, size in self.HEAD_SIZES.items()}
            )
            return
        if selected != "tiny":
            raise ValueError(f"unsupported Goal4 backend: {selected!r}; use tiny or medicalnet")
        self.backend = "tiny"
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
        if self.backend == "medicalnet":
            features = self.encoder(x)
        else:
            features = self.features(x).mean(dim=(2, 3, 4))
        return {name: head(features) for name, head in self.heads.items()}
