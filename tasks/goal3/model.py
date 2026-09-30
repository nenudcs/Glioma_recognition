from __future__ import annotations

import os

import torch
from torch import nn

from tasks.goal_common import MedicalNet3DClassifier


class Goal3Model(nn.Module):
    """Case-level four-channel classifier matching the training checkpoint."""

    def __init__(self, in_channels: int = 4, *, backend: str | None = None) -> None:
        super().__init__()
        selected = (backend or os.environ.get("GOAL3_BACKEND", "tiny")).lower()
        self.backend = selected
        if selected == "medicalnet":
            # Keep the module layout identical to the training-side
            # ``MedicalNet3DClassifier`` so a training ``best.pt`` can be
            # loaded without key remapping.
            medicalnet = MedicalNet3DClassifier(
                depth=int(os.environ.get("GOAL3_MEDICALNET_DEPTH", "50")),
                in_channels=in_channels,
                num_classes=1,
            )
            self._medicalnet_api = medicalnet._api
            if self._medicalnet_api == "official_api":
                self.input_adapter = medicalnet.input_adapter
                self.encoder = medicalnet.encoder
                self.classifier = medicalnet.classifier
            else:
                self.backbone = medicalnet.backbone
        elif selected == "tiny":
            self.features = nn.Sequential(nn.Conv3d(in_channels, 8, 3, padding=1), nn.InstanceNorm3d(8), nn.ReLU(inplace=True), nn.MaxPool3d(2), nn.Conv3d(8, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(inplace=True))
            self.head = nn.Linear(16, 1)
        else:
            raise ValueError(f"unsupported Goal3 backend: {selected!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "medicalnet":
            if self._medicalnet_api == "official_api":
                output = self.classifier(self.encoder(self.input_adapter(x)))
            else:
                output = self.backbone(x)
            return output.reshape(x.shape[0], -1).squeeze(1)
        return self.head(self.features(x).mean(dim=(2, 3, 4))).squeeze(1)
