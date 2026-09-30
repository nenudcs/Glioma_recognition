"""Official Goal 4 case-level task."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from tasks.goal_common import resize_volume
from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.goal4.model import Goal4Model
from tasks.goal_common import _goal4_result, _TorchTask, series_modality, load_checkpoint
from tasks.results import Goal4Result


MODALITIES = ("t1", "t1ce", "t2", "flair")


def _default_checkpoint() -> Path:
    workspace = Path(os.environ.get("COMPETITION_WORKSPACE", "/2026aicompetition/workspace"))
    return workspace / "Tumor-segment-training/checkpoint/goal4_medicalnet_930_fixed/best.pt"


def _shape(name: str) -> tuple[int, int, int]:
    raw = os.environ.get(name, "64,64,32").replace("x", ","); values = tuple(int(x.strip()) for x in raw.split(","))
    if len(values) != 3 or any(x < 1 for x in values): raise ValueError(f"{name} must be D,H,W, got {raw!r}")
    return values


def _modality(series) -> str | None:
    return series_modality(series)


def _normalize_resize(series, shape):
    values = np.asarray(series.image, dtype=np.float32); finite = np.isfinite(values)
    if not finite.any(): return np.zeros(shape, dtype=np.float32)
    values = np.nan_to_num(values, copy=True); lo, hi = np.percentile(values[finite], (1.0, 99.0)); values = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)
    return np.asarray(resize_volume(values, shape, is_mask=False), dtype=np.float32)


class Goal4Task(_TorchTask, StudyTask[Goal4Result]):
    name = "goal4"

    def load_model(self) -> None:
        self.input_shape = _shape("GOAL4_INPUT_SHAPE")
        in_channels = int(os.environ.get("GOAL4_IN_CHANNELS", str(len(MODALITIES))))
        if in_channels != len(MODALITIES):
            raise ValueError("GOAL4_IN_CHANNELS must be 4 for the trained four-channel checkpoint")
        backend = os.environ.get("GOAL4_BACKEND", "medicalnet")
        model = Goal4Model(in_channels=in_channels, backend=backend)
        checkpoint = os.environ.get("GOAL4_CHECKPOINT", str(_default_checkpoint()))
        if checkpoint:
            load_checkpoint(model, checkpoint, device=torch.device("cpu"))
        model.to(self.device); model.eval(); self.model = model

    def _input(self, context: PipelineContext) -> torch.Tensor:
        by_modality = {}
        for series in context.study.series:
            key = _modality(series)
            if key and key not in by_modality: by_modality[key] = series
        channels = [_normalize_resize(by_modality[m], self.input_shape) if m in by_modality else np.zeros(self.input_shape, dtype=np.float32) for m in MODALITIES]
        return torch.from_numpy(np.stack(channels, axis=0)).unsqueeze(0).to(self.device)

    def predict(self, context: PipelineContext) -> Goal4Result:
        if self.model is None: raise RuntimeError("Goal4 model has not been loaded")
        with torch.inference_mode(): outputs = self.model(self._input(context))
        return _goal4_result(outputs)


TorchGoal4Task = Goal4Task
__all__ = ["Goal4Task", "TorchGoal4Task"]
