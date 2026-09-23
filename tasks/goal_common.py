from __future__ import annotations

"""Shared preprocessing and result conversion for the Goal Task adapters."""

from pathlib import Path
import numpy as np
import torch
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
            series = _select_series(context.study.series, (modality,))
            values = np.asarray(series.image, dtype=np.float32)
            finite = np.isfinite(values)
            if not finite.any():
                values = np.zeros(values.shape, dtype=np.float32)
            else:
                values = np.nan_to_num(values, copy=True)
                lo, hi = np.percentile(values[finite], (1.0, 99.0))
                values = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
            arrays.append(np.asarray(values, dtype=np.float32))
        tensor = torch.from_numpy(np.stack(arrays, axis=0).astype(np.float32, copy=False)).unsqueeze(0).to(self.device)
        return F.interpolate(tensor, size=self.input_shape, mode="trilinear", align_corners=False)


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
        for item in series:
            text = " ".join((item.modality or "", str(item.metadata.get("SeriesDescription", "")), item.series_uid)).lower()
            if hint in text:
                return item
    if not series:
        raise MissingSeriesError("study has no image series")
    return series[0]


def _restore_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    restored = F.interpolate(tensor, size=shape, mode="nearest")[0, 0].numpy()
    return (restored >= 0.5).astype(np.uint8)
