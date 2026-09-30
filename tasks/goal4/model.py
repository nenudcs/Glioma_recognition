from __future__ import annotations

import importlib
import os

import torch
from torch import nn


class _MedicalNetFeatures(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__(); self.backbone = backbone
        block = backbone.layer4[-1]
        self.feature_dim = int(block.conv3.out_channels if hasattr(block, "conv3") else block.conv2.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone; x = m.relu(m.bn1(m.conv1(x))); x = m.maxpool(x); x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
        return torch.nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)


def _build_backbone(depth: int) -> nn.Module:
    errors = []
    for module_name in ("models.resnet", "medicalnet.models.resnet"):
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, f"resnet{depth}", None)
            if factory is not None:
                return factory(sample_input_W=32, sample_input_H=32, sample_input_D=16, shortcut_type="B", no_cuda=True, num_seg_classes=1)
            generate_model = getattr(module, "generate_model", None)
            if generate_model is not None:
                return generate_model(model_depth=depth, n_classes=1, n_input_channels=1, shortcut_type="B", conv1_t_size=7, conv1_t_stride=1, no_max_pool=False)
            errors.append(f"{module_name}: no supported factory")
        except (ImportError, AttributeError) as exc: errors.append(f"{module_name}: {exc}")
    raise ImportError("MedicalNet is not importable: " + " | ".join(errors))


class Goal4Model(nn.Module):
    HEAD_SIZES = {"location": 15, "morphology": 2, "who_grade": 4, "enhancement": 1, "enhancement_pattern": 8, "necrosis": 1, "cystic_change": 1, "hemorrhage": 1, "calcification": 1, "margin_clear": 1, "lobulation": 1, "signal_t2wi": 3, "signal_flair": 3}

    def __init__(self, in_channels: int = 4, *, backend: str | None = None) -> None:
        super().__init__(); selected = (backend or os.environ.get("GOAL4_BACKEND", "tiny")).lower(); self.backend = selected
        if selected == "medicalnet":
            backbone = _build_backbone(int(os.environ.get("GOAL4_MEDICALNET_DEPTH", "50")))
            self.encoder = nn.ModuleDict({"input_adapter": nn.Identity() if in_channels == 1 else nn.Conv3d(in_channels, 1, 1), "encoder": _MedicalNetFeatures(backbone)})
            self.heads = nn.ModuleDict({name: nn.Linear(self.encoder["encoder"].feature_dim, size) for name, size in self.HEAD_SIZES.items()})
        elif selected == "tiny":
            self.features = nn.Sequential(nn.Conv3d(in_channels, 8, 3, padding=1), nn.InstanceNorm3d(8), nn.ReLU(inplace=True), nn.MaxPool3d(2), nn.Conv3d(8, 16, 3, padding=1), nn.InstanceNorm3d(16), nn.ReLU(inplace=True))
            self.heads = nn.ModuleDict({name: nn.Linear(16, size) for name, size in self.HEAD_SIZES.items()})
        else: raise ValueError(f"unsupported Goal4 backend: {selected!r}")

    @property
    def feature_dim(self) -> int:
        return self.encoder["encoder"].feature_dim if self.backend == "medicalnet" else 16

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder["encoder"](self.encoder["input_adapter"](x)) if self.backend == "medicalnet" else self.features(x).mean(dim=(2, 3, 4))
        return {name: head(features) for name, head in self.heads.items()}
