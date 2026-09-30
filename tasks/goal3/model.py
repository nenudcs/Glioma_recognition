from __future__ import annotations

import importlib
import os

import torch
from torch import nn


class _MedicalNetFeatures(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        block = backbone.layer4[-1]
        self.feature_dim = int(block.conv3.out_channels if hasattr(block, "conv3") else block.conv2.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.relu(model.bn1(model.conv1(x)))
        x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
        return torch.nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)


def _build_backbone(depth: int) -> nn.Module:
    errors = []
    for module_name in ("models.resnet", "medicalnet.models.resnet"):
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, f"resnet{depth}", None)
            if factory is not None:
                return factory(sample_input_W=32, sample_input_H=32, sample_input_D=16,
                               shortcut_type="B", no_cuda=True, num_seg_classes=1)
            generate_model = getattr(module, "generate_model", None)
            if generate_model is not None:
                return generate_model(model_depth=depth, n_classes=1, n_input_channels=1,
                                      shortcut_type="B", conv1_t_size=7, conv1_t_stride=1,
                                      no_max_pool=False)
            errors.append(f"{module_name}: no supported factory")
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError("MedicalNet is not importable: " + " | ".join(errors))


class Goal3Model(nn.Module):
    """Case-level four-channel classifier matching the training checkpoint."""

    def __init__(self, in_channels: int = 4, *, backend: str | None = None) -> None:
        super().__init__()
        selected = (backend or os.environ.get("GOAL3_BACKEND", "tiny")).lower()
        self.backend = selected
        if selected == "medicalnet":
            backbone = _build_backbone(int(os.environ.get("GOAL3_MEDICALNET_DEPTH", "50")))
            self.input_adapter = nn.Identity() if in_channels == 1 else nn.Conv3d(in_channels, 1, 1)
            self.encoder = _MedicalNetFeatures(backbone)
            self.classifier = nn.Linear(self.encoder.feature_dim, 1)
        elif selected == "tiny":
            self.features = nn.Sequential(nn.Conv3d(in_channels, 8, 3, padding=1), nn.InstanceNorm3d(8), nn.ReLU(inplace=True), nn.MaxPool3d(2), nn.Conv3d(8, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(inplace=True))
            self.head = nn.Linear(16, 1)
        else:
            raise ValueError(f"unsupported Goal3 backend: {selected!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "medicalnet":
            return self.classifier(self.encoder(self.input_adapter(x))).squeeze(1)
        return self.head(self.features(x).mean(dim=(2, 3, 4))).squeeze(1)
