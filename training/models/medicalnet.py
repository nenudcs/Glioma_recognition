"""Optional MedicalNet adapters used by training and competition tasks.

MedicalNet is kept as an optional dependency because the competition image and
weight packages are usually mounted separately.  The adapter imports the
MedicalNet ``models.resnet.generate_model`` factory only when requested and
accepts both a raw state dict and the common ``{"state_dict": ...}`` checkpoint
format.  It never downloads weights.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "net"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("MedicalNet checkpoint must be a state_dict or a checkpoint dict")
    return {
        str(key).removeprefix("module."): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def _medicalnet_factory() -> Any:
    errors: list[str] = []
    for module_name in ("models.resnet", "medicalnet.models.resnet"):
        try:
            module = importlib.import_module(module_name)
            return module.generate_model
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "MedicalNet is not importable. Add the MedicalNet repository to PYTHONPATH "
        "or install its package before selecting backend=medicalnet. "
        + " | ".join(errors)
    )


class MedicalNet3DClassifier(nn.Module):
    """MedicalNet ResNet classifier for ``(B, C, D, H, W)`` inputs."""

    def __init__(
        self,
        *,
        depth: int = 50,
        in_channels: int = 4,
        num_classes: int = 1,
        checkpoint: str | Path | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        generate_model = _medicalnet_factory()
        self.backbone = generate_model(
            model_depth=depth,
            n_classes=num_classes,
            n_input_channels=in_channels,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
        )
        self.checkpoint_report: dict[str, Any] = {}
        if checkpoint:
            state = _unwrap_state_dict(torch.load(checkpoint, map_location="cpu"))
            missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
            self.checkpoint_report = {
                "path": str(checkpoint),
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone(x)
        if output.ndim == 1:
            output = output.unsqueeze(0)
        return output


class MedicalNet3DEncoder(nn.Module):
    """MedicalNet backbone with its final classifier replaced by Identity."""

    def __init__(
        self,
        *,
        depth: int = 50,
        in_channels: int = 4,
        checkpoint: str | Path | None = None,
        strict: bool = False,
    ) -> None:
        super().__init__()
        generate_model = _medicalnet_factory()
        self.backbone = generate_model(
            model_depth=depth,
            n_classes=1,
            n_input_channels=in_channels,
            shortcut_type="B",
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
        )
        classifier = getattr(self.backbone, "fc", None)
        if classifier is None or not hasattr(classifier, "in_features"):
            raise RuntimeError("MedicalNet backbone does not expose a final fc layer")
        self.feature_dim = int(classifier.in_features)
        self.backbone.fc = nn.Identity()
        self.checkpoint_report: dict[str, Any] = {}
        if checkpoint:
            state = _unwrap_state_dict(torch.load(checkpoint, map_location="cpu"))
            missing, unexpected = self.backbone.load_state_dict(state, strict=strict)
            self.checkpoint_report = {
                "path": str(checkpoint),
                "missing_keys": list(missing),
                "unexpected_keys": list(unexpected),
            }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x).reshape(x.shape[0], -1)


__all__ = ["MedicalNet3DClassifier", "MedicalNet3DEncoder"]
