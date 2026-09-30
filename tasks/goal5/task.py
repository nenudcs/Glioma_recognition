"""Official Goal 5 task with two nnUNet models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to

from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.goal5.model import Goal5Model
from tasks.results import Goal5Result
from tasks.goal_common import _TorchTask, _restore_mask, _select_series


def _find(series, names):
    for name in names:
        key = "".join(ch for ch in name.lower() if ch.isalnum())
        for item in series:
            text = " ".join((item.modality or "", str(item.metadata.get("SeriesDescription", "")), item.series_uid)).lower()
            compact = "".join(ch for ch in text if ch.isalnum())
            if key == "t1" and ("t1ce" in compact or "t1c" in compact or "enh" in compact):
                continue
            if key == "t2" and "flair" in compact:
                continue
            if key in compact:
                return item
    return None


def _as_reference(series):
    return nib.Nifti1Image(np.asarray(series.image, dtype=np.float32), np.asarray(series.affine, dtype=np.float64))


class _NNUNetRunner:
    def __init__(self, model_folder: str, device: str) -> None:
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError as exc:
            raise ImportError("nnUNet v2 is required for GOAL5_BACKEND=nnunet") from exc
        device_obj = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                         perform_everything_on_device=True, device=device_obj,
                                         verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        self.predictor.initialize_from_trained_model_folder(model_folder, use_folds=(0,), checkpoint_name="checkpoint_final.pth")

    def predict(self, images: list[nib.Nifti1Image]) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="goal5_nnunet_") as tmp:
            root = Path(tmp); paths = []
            for index, image in enumerate(images):
                path = root / f"case_{index:04d}.nii.gz"; nib.save(image, str(path)); paths.append(str(path))
            output = root / "prediction"
            self.predictor.predict_from_files([paths], [str(output)], overwrite=True,
                                              num_processes_preprocessing=1,
                                              num_processes_segmentation_export=1)
            result = root / "prediction.nii.gz"
            if not result.is_file(): raise RuntimeError(f"nnUNet did not write {result}")
            return (np.asarray(nib.load(str(result)).dataobj) > 0).astype(np.uint8)


class Goal5Task(_TorchTask, StudyTask[Goal5Result]):
    name = "goal5"

    def load_model(self) -> None:
        backend = os.environ.get("GOAL5_BACKEND", "tiny").lower()
        if backend == "nnunet":
            core = os.environ.get("GOAL5_CORE_NNUNET_MODEL_FOLDER")
            abnormal = os.environ.get("GOAL5_ABNORMAL_NNUNET_MODEL_FOLDER")
            if not core or not abnormal:
                raise RuntimeError("set GOAL5_CORE_NNUNET_MODEL_FOLDER and GOAL5_ABNORMAL_NNUNET_MODEL_FOLDER")
            device = os.environ.get("GOAL5_DEVICE", "auto")
            self.backend = "nnunet"; self.core_runner = _NNUNetRunner(core, device); self.abnormal_runner = _NNUNetRunner(abnormal, device); self.model = None; return
        if backend != "tiny": raise ValueError(f"unsupported Goal5 backend: {backend!r}")
        self.backend = "tiny"; self.core_runner = None; self.abnormal_runner = None; self._finish_load(Goal5Model())

    @staticmethod
    def _images(context: PipelineContext, reference) -> list[nib.Nifti1Image]:
        ref = _as_reference(reference); images = []
        for modality in ("t1", "t1ce", "t2", "flair"):
            source = _find(context.study.series, (modality,))
            if source is None:
                data = np.zeros(ref.shape, dtype=np.float32)
            else:
                src = _as_reference(source)
                if src.shape == ref.shape and np.allclose(src.affine, ref.affine): data = np.asarray(src.dataobj, dtype=np.float32)
                else: data = np.asarray(resample_from_to(src, ref, order=1).dataobj, dtype=np.float32)
                data = np.nan_to_num(data, copy=False)
            images.append(nib.Nifti1Image(data, ref.affine, ref.header))
        return images

    def predict(self, context: PipelineContext) -> Goal5Result:
        if self.backend == "nnunet":
            t1ce = _select_series(context.study.series, ("t1ce", "t1+c", "t1 enhanced", "t1"))
            flair = _select_series(context.study.series, ("flair", "t2flair", "t2"))
            core = self.core_runner.predict(self._images(context, t1ce))
            abnormal = self.abnormal_runner.predict(self._images(context, flair))
            if core.shape != t1ce.image.shape: core = _restore_mask(core, t1ce.image.shape)
            if abnormal.shape != flair.image.shape: abnormal = _restore_mask(abnormal, flair.image.shape)
            return Goal5Result(core_mask=core.astype(np.uint8), core_source_series_uid=t1ce.series_uid,
                               flair_mask=abnormal.astype(np.uint8), flair_source_series_uid=flair.series_uid)
        if self.model is None: raise RuntimeError("Goal5 model has not been loaded")
        t1ce = _select_series(context.study.series, ("t1ce", "t1+c", "t1 enhanced", "t1")); flair = _select_series(context.study.series, ("flair", "t2flair", "t2"))
        with torch.inference_mode(): masks = (torch.sigmoid(self.model(self._input(context))) >= 0.5).to(torch.uint8).cpu().numpy()[0]
        return Goal5Result(core_mask=_restore_mask(masks[0], t1ce.image.shape), core_source_series_uid=t1ce.series_uid, flair_mask=_restore_mask(masks[1], flair.image.shape), flair_source_series_uid=flair.series_uid)


TorchGoal5Task = Goal5Task
__all__ = ["Goal5Task", "TorchGoal5Task"]
