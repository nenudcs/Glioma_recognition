"""Official Goal 3 case-level task."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from core.resize import resize_volume
from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.goal3.model import Goal3Model
from tasks.results import Goal3Result
from tasks.goal_common import _TorchTask


MODALITIES = ("t1", "t1ce", "t2", "flair")


def _shape(name: str) -> tuple[int, int, int]:
    raw = os.environ.get(name, "64,64,32").replace("x", ",")
    values = tuple(int(x.strip()) for x in raw.split(","))
    if len(values) != 3 or any(x < 1 for x in values):
        raise ValueError(f"{name} must be D,H,W, got {raw!r}")
    return values


def _modality(series) -> str | None:
    text = " ".join((series.modality or "", str(series.metadata.get("SeriesDescription", "")), series.series_uid)).lower()
    compact = "".join(ch for ch in text if ch.isalnum())
    if "flair" in compact or "t2flair" in compact: return "flair"
    if "t1ce" in compact or "t1c" in compact or "enhanced" in compact or "enh" in compact: return "t1ce"
    if "t2" in compact: return "t2"
    if "t1" in compact: return "t1"
    return None


def _normalize_resize(series, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(series.image, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any(): return np.zeros(shape, dtype=np.float32)
    values = np.nan_to_num(values, copy=True)
    lo, hi = np.percentile(values[finite], (1.0, 99.0))
    values = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)
    return np.asarray(resize_volume(values, shape, is_mask=False), dtype=np.float32)


class Goal3Task(_TorchTask, StudyTask[Goal3Result]):
    name = "goal3"

    def load_model(self) -> None:
        self.input_shape = _shape("GOAL3_INPUT_SHAPE")
        model = Goal3Model(in_channels=4, backend=os.environ.get("GOAL3_BACKEND", "tiny"))
        checkpoint = os.environ.get("GOAL3_CHECKPOINT")
        if checkpoint:
            payload = torch.load(Path(checkpoint), map_location="cpu")
            state = payload.get("model", payload) if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Goal3 checkpoint mismatch: missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}")
        model.to(self.device); model.eval(); self.model = model

    def _input(self, context: PipelineContext) -> torch.Tensor:
        by_modality = {}
        for series in context.study.series:
            key = _modality(series)
            if key and key not in by_modality:
                by_modality[key] = series
        channels = [_normalize_resize(by_modality[m], self.input_shape) if m in by_modality else np.zeros(self.input_shape, dtype=np.float32) for m in MODALITIES]
        return torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(self.device)

    def predict(self, context: PipelineContext) -> Goal3Result:
        if self.model is None: raise RuntimeError("Goal3 model has not been loaded")
        with torch.inference_mode():
            probability = float(torch.sigmoid(self.model(self._input(context)))[0].item())
        return Goal3Result(tumor_probability=probability)


TorchGoal3Task = Goal3Task
__all__ = ["Goal3Task", "TorchGoal3Task"]
