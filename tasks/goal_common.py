from __future__ import annotations

"""Shared preprocessing and result conversion for the Goal Task adapters."""

import importlib
import os
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from core.exceptions import MissingSeriesError
from data.structures import Series
from pipeline.context import PipelineContext
from tasks.results import BinaryResult, CategoricalResult, Goal4Result


LOCATION_LABELS = (
    "Brainstem", "RightParietal", "RightFrontal", "RightBasalGanglia",
    "RightTemporal", "RightCerebellar", "RightOccipital", "LeftParietal",
    "LeftFrontal", "LeftBasalGanglia", "LeftTemporal", "LeftCerebellar",
    "LeftOccipital", "Other", "Unknown",
)
MORPHOLOGY_LABELS = ("Regular", "Irregular")
WHO_LABELS = ("1", "2", "3", "4")
PATTERN_LABELS = (
    "None", "Ring", "RimEnhancing", "Nodular", "GroundGlass", "Gyriform",
    "Multifocal", "Other",
)
SIGNAL_LABELS = ("Low", "Iso", "High")
INPUT_MODALITIES = ("t1", "t1ce", "t2", "flair")


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


def _medicalnet_factory(depth: int, *, in_channels: int, num_classes: int) -> tuple[nn.Module, str]:
    """Build either the newer classifier-style or official MedicalNet model.

    The public MedicalNet repository exposes ``resnet50``/``resnet101`` in
    ``models.resnet``; it does not expose the classifier-style ``generate_model``
    function used by some forks.  Supporting both keeps the training code
    compatible with the mounted public model package.
    """
    _ensure_medicalnet_on_path()
    errors: list[str] = []
    for module_name in ("models.resnet", "medicalnet.models.resnet"):
        try:
            module = importlib.import_module(module_name)
            generate_model = getattr(module, "generate_model", None)
            if generate_model is not None:
                model = generate_model(
                    model_depth=depth,
                    n_classes=num_classes,
                    n_input_channels=in_channels,
                    shortcut_type="B",
                    conv1_t_size=7,
                    conv1_t_stride=1,
                    no_max_pool=False,
                )
                return model, "classifier_api"
            factory = getattr(module, f"resnet{depth}", None)
            if factory is not None:
                model = factory(
                    sample_input_W=32,
                    sample_input_H=32,
                    sample_input_D=16,
                    shortcut_type="B",
                    no_cuda=True,
                    num_seg_classes=num_classes,
                )
                return model, "official_api"
            errors.append(f"{module_name}: no supported MedicalNet factory")
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}: {exc}")
    raise ImportError(
        "MedicalNet is not importable. Add the MedicalNet repository to PYTHONPATH "
        "or install its package before selecting backend=medicalnet. "
        + " | ".join(errors)
    )


def _ensure_medicalnet_on_path() -> None:
    workspace = Path(
        os.environ.get("COMPETITION_WORKSPACE", "/2026aicompetition/workspace")
    )
    candidates = (
        str(workspace / "model-medicalnet/MedicalNet-master"),
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_dir() and candidate not in sys.path:
            sys.path.insert(0, candidate)
            return


def _load_compatible(module: nn.Module, checkpoint: str | Path, *, strict: bool) -> dict[str, Any]:
    raw = _unwrap_state_dict(torch.load(checkpoint, map_location="cpu"))
    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        key = key.removeprefix("backbone.")
        normalized[key] = value
    target = module.state_dict()
    compatible = {
        key: value
        for key, value in normalized.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    mismatched = sorted(
        key for key, value in normalized.items()
        if key in target and tuple(value.shape) != tuple(target[key].shape)
    )
    missing = sorted(set(target) - set(compatible))
    unexpected = sorted(set(normalized) - set(target))
    module.load_state_dict(compatible, strict=False)
    report = {
        "path": str(checkpoint),
        "loaded_keys": len(compatible),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "mismatched_keys": mismatched,
    }
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(f"MedicalNet checkpoint is not compatible: {report}")
    return report


class _OfficialMedicalNetFeatures(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        last_block = backbone.layer4[-1]
        if hasattr(last_block, "conv3"):
            self.feature_dim = int(last_block.conv3.out_channels)
        elif hasattr(last_block, "conv2"):
            self.feature_dim = int(last_block.conv2.out_channels)
        else:
            raise RuntimeError("Cannot determine MedicalNet feature dimension")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.backbone
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return torch.nn.functional.adaptive_avg_pool3d(x, 1).flatten(1)


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
        backbone, self._api = _medicalnet_factory(
            depth, in_channels=in_channels, num_classes=num_classes
        )
        self.checkpoint_report: dict[str, Any] = {}
        if checkpoint:
            self.checkpoint_report = _load_compatible(
                backbone, checkpoint, strict=strict
            )
        if self._api == "official_api":
            self.input_adapter = nn.Identity() if in_channels == 1 else nn.Conv3d(in_channels, 1, 1)
            self.encoder = _OfficialMedicalNetFeatures(backbone)
            self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)
        else:
            self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._api == "official_api":
            return self.classifier(self.encoder(self.input_adapter(x)))
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
        self.backbone, api = _medicalnet_factory(
            depth, in_channels=in_channels, num_classes=1
        )
        self.checkpoint_report: dict[str, Any] = {}
        if checkpoint:
            self.checkpoint_report = _load_compatible(
                self.backbone, checkpoint, strict=strict
            )
        if api == "official_api":
            self.input_adapter = nn.Identity() if in_channels == 1 else nn.Conv3d(in_channels, 1, 1)
            self.encoder = _OfficialMedicalNetFeatures(self.backbone)
            self.feature_dim = self.encoder.feature_dim
        else:
            classifier = getattr(self.backbone, "fc", None)
            if classifier is None or not hasattr(classifier, "in_features"):
                raise RuntimeError("MedicalNet backbone does not expose a final fc layer")
            self.feature_dim = int(classifier.in_features)
            self.backbone.fc = nn.Identity()
            self.input_adapter = nn.Identity()
            self.encoder = self.backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.input_adapter(x)).reshape(x.shape[0], -1)




def resize_volume(
    array: np.ndarray | torch.Tensor,
    target_shape: Sequence[int],
    *,
    is_mask: bool = False,
) -> np.ndarray | torch.Tensor:
    """Resize a 3-D or channel-first 4-D volume without changing its type."""

    shape = tuple(int(x) for x in target_shape)
    if len(shape) != 3 or any(x < 1 for x in shape):
        raise ValueError(f"target_shape must contain three positive integers, got {shape}")
    original_is_numpy = isinstance(array, np.ndarray)
    tensor = torch.as_tensor(array)
    if tensor.ndim == 3:
        tensor = tensor[None, None]
        restore = "3d"
    elif tensor.ndim == 4:
        tensor = tensor[None]
        restore = "4d"
    elif tensor.ndim == 5:
        restore = "5d"
    else:
        raise ValueError(f"volume must be 3-D, channel-first 4-D, or 5-D, got {tensor.shape}")
    tensor = tensor.float()
    mode = "nearest" if is_mask else "trilinear"
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    result = F.interpolate(tensor, size=shape, mode=mode, **kwargs)
    if is_mask:
        result = (result >= 0.5).to(torch.uint8)
    if restore == "3d":
        result = result[0, 0]
    elif restore == "4d":
        result = result[0]
    if original_is_numpy:
        return result.cpu().numpy()
    return result




def canonical_modality(value: object) -> str | None:
    """Use the training dataset's SeriesType mapping."""
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    if "flair" in text:
        return "flair"
    if "t1ce" in text or "t1c" in text or "enh" in text or "t1plusc" in text:
        return "t1ce"
    if text == "t1" or text.startswith("t1_") or text.endswith("t1"):
        return "t1"
    if "t2" in text:
        return "t2"
    return None


def series_modality(series: Series) -> str | None:
    # Series.modality contains the loader's authoritative SeriesType.xlsx
    # value. Do not concatenate it with descriptions or opaque numeric UIDs.
    for value in (series.modality, series.metadata.get("SeriesDescription"), series.series_uid):
        result = canonical_modality(value)
        if result is not None:
            return result
    return None


def load_checkpoint(model: torch.nn.Module, path: str | Path, *, device: torch.device) -> None:
    """Load a training checkpoint and fail loudly on an architecture mismatch.

    Training jobs write either a raw state dict or ``{"model": state_dict}``;
    DataParallel adds a ``module.`` prefix.  Normalize those harmless wrapper
    differences, then require every inference parameter to be present with the
    expected shape so a random or partially loaded model cannot reach scoring.
    """
    checkpoint_path = Path(path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    state = payload
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "net"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint must contain a state dict: {checkpoint_path}")
    raw = {
        str(key).removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    target = model.state_dict()
    candidates = [raw]
    for prefix in ("model.", "goal3.", "goal4."):
        candidates.append({key.removeprefix(prefix): value for key, value in raw.items()})
    candidate = max(
        candidates,
        key=lambda item: sum(
            key in target and tuple(value.shape) == tuple(target[key].shape)
            for key, value in item.items()
        ),
    )
    compatible = {
        key: value
        for key, value in candidate.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    missing = sorted(set(target) - set(compatible))
    unexpected = sorted(set(candidate) - set(target))
    if not compatible or missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch for {checkpoint_path}: "
            f"loaded={len(compatible)}/{len(target)}, "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    model.load_state_dict(compatible, strict=True)


class _TorchTask:
    def __init__(self, weights_path: str | Path | None = None, input_shape: tuple[int, int, int] = (32, 32, 16)) -> None:
        self.weights_path = Path(weights_path).expanduser() if weights_path else None
        self.input_shape = input_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: torch.nn.Module | None = None

    def _finish_load(self, model: torch.nn.Module) -> None:
        if self.weights_path is not None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(f"checkpoint not found: {self.weights_path}")
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def _input(self, context: PipelineContext) -> torch.Tensor:
        arrays = []
        for modality in INPUT_MODALITIES:
            series = _find_series(context.study.series, modality)
            if series is None:
                arrays.append(np.zeros(self.input_shape, dtype=np.float32))
                continue
            values = np.asarray(series.image, dtype=np.float32)
            finite = np.isfinite(values)
            if not finite.any():
                values = np.zeros(values.shape, dtype=np.float32)
            else:
                values = np.nan_to_num(values, copy=True)
                lo, hi = np.percentile(values[finite], (1.0, 99.0))
                values = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
            arrays.append(
                np.asarray(
                    resize_volume(values, self.input_shape, is_mask=False),
                    dtype=np.float32,
                )
            )
        tensor = torch.from_numpy(
            np.stack(arrays, axis=0).astype(np.float32, copy=False)
        ).unsqueeze(0).to(self.device)
        return tensor


def _goal4_result(outputs: dict[str, torch.Tensor]) -> Goal4Result:
    def categorical(name: str, labels: tuple[str, ...]) -> CategoricalResult:
        probabilities = torch.softmax(outputs[name], dim=1)[0].cpu().numpy().astype(float)
        index = int(np.argmax(probabilities))
        return CategoricalResult(labels[index], {label: float(probabilities[i]) for i, label in enumerate(labels)})

    def binary(name: str) -> BinaryResult:
        probability = float(torch.sigmoid(outputs[name])[0, 0].item())
        return BinaryResult(probability >= 0.5, probability)

    pattern = categorical("enhancement_pattern", PATTERN_LABELS)
    who = categorical("who_grade", WHO_LABELS)
    who = CategoricalResult(
        predicted=int(who.predicted) if who.predicted is not None else None,
        probabilities=who.probabilities,
    )
    return Goal4Result(
        location=categorical("location", LOCATION_LABELS).predicted or "Unknown",
        morphology=categorical("morphology", MORPHOLOGY_LABELS),
        who_grade=who,
        enhancement=binary("enhancement"),
        enhancement_pattern=pattern,
        necrosis=binary("necrosis"),
        cystic_change=binary("cystic_change"),
        hemorrhage=binary("hemorrhage"),
        calcification=binary("calcification"),
        margin_clear=binary("margin_clear"),
        lobulation=binary("lobulation"),
        signal_t2wi=categorical("signal_t2wi", SIGNAL_LABELS),
        signal_flair=categorical("signal_flair", SIGNAL_LABELS),
        conclusion="Trainable Goal 4 baseline prediction.",
    )


def _select_series(series: tuple[Series, ...], hints: tuple[str, ...]) -> Series:
    for hint in hints:
        item = _find_series(series, hint)
        if item is not None:
            return item
    if not series:
        raise MissingSeriesError("study has no image series")
    return series[0]


def _find_series(series: tuple[Series, ...], hint: str) -> Series | None:
    canonical_hint = canonical_modality(hint)
    for item in series:
        if canonical_hint is not None and series_modality(item) == canonical_hint:
            return item
        if canonical_hint is None:
            text = " ".join(
                (
                    item.modality or "",
                    str(item.metadata.get("SeriesDescription", "")),
                    item.series_uid,
                )
            ).lower()
            if hint.lower() in text:
                return item
    return None


def _restore_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    restored = F.interpolate(tensor, size=shape, mode="nearest")[0, 0].numpy()
    return (restored >= 0.5).astype(np.uint8)
