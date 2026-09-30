from __future__ import annotations

import os

import torch
from torch import nn

from tasks.goal_common import MedicalNet3DEncoder


class Goal4Model(nn.Module):
    HEAD_SIZES = {"location": 15, "morphology": 2, "who_grade": 4, "enhancement": 1, "enhancement_pattern": 8, "necrosis": 1, "cystic_change": 1, "hemorrhage": 1, "calcification": 1, "margin_clear": 1, "lobulation": 1, "signal_t2wi": 3, "signal_flair": 3}

    def __init__(self, in_channels: int = 4, *, backend: str | None = None) -> None:
        super().__init__(); selected = (backend or os.environ.get("GOAL4_BACKEND", "tiny")).lower(); self.backend = selected
        if selected == "medicalnet":
            # ``Goal4MedicalNet`` in the training repository registers the
            # encoder directly under this name. Reuse that adapter so the
            # inference checkpoint has the same state-dict keys.
            self.encoder = MedicalNet3DEncoder(
                depth=int(os.environ.get("GOAL4_MEDICALNET_DEPTH", "50")),
                in_channels=in_channels,
            )
            self.heads = nn.ModuleDict({name: nn.Linear(self.encoder.feature_dim, size) for name, size in self.HEAD_SIZES.items()})
        elif selected == "tiny":
            self.features = nn.Sequential(nn.Conv3d(in_channels, 8, 3, padding=1), nn.InstanceNorm3d(8), nn.ReLU(inplace=True), nn.MaxPool3d(2), nn.Conv3d(8, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(inplace=True))
            self.heads = nn.ModuleDict({name: nn.Linear(16, size) for name, size in self.HEAD_SIZES.items()})
        else: raise ValueError(f"unsupported Goal4 backend: {selected!r}")

    @property
    def feature_dim(self) -> int:
        return self.encoder.feature_dim if self.backend == "medicalnet" else 16

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x) if self.backend == "medicalnet" else self.features(x).mean(dim=(2, 3, 4))
        return {name: head(features) for name, head in self.heads.items()}
