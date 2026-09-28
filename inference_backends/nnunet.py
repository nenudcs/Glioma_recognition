"""Thin nnUNet v2 inference adapter.

Training remains the official nnUNet workflow (``nnUNet_plan_and_preprocess``
and ``nnUNet_train``).  This adapter makes an exported nnUNet model usable by
the repository's StudyTask without coupling the core service to nnUNet at
import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class NNUNetPredictorAdapter:
    """Run a local nnUNet v2 model and return a binary mask in source shape."""

    def __init__(
        self,
        *,
        model_folder: str | Path,
        folds: tuple[int, ...] = (0,),
        device: str = "cuda",
        verbose: bool = False,
    ) -> None:
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        except ImportError as exc:
            raise ImportError(
                "nnUNet v2 is not installed; install it before selecting the nnunet backend"
            ) from exc
        import torch

        device_obj = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=True,
            device=device_obj,
            verbose=verbose,
            verbose_preprocessing=verbose,
            allow_tqdm=verbose,
        )
        self.predictor.initialize_from_trained_model_folder(
            str(model_folder), use_folds=folds, checkpoint_name="checkpoint_final.pth"
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Predict one channel-first volume and return ``uint8`` mask.

        The adapter tries nnUNet's logits API first, then its label API.  The
        source affine/shape conversion remains the caller's responsibility.
        """

        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 3:
            array = array[None]
        if array.ndim != 4:
            raise ValueError(f"nnUNet input must be (C,D,H,W), got {array.shape}")
        if hasattr(self.predictor, "predict_logits_from_preprocessed_data"):  # nnUNet v2
            logits = self.predictor.predict_logits_from_preprocessed_data(array)
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()
            return (np.argmax(np.asarray(logits), axis=0) > 0).astype(np.uint8)
        raise RuntimeError(
            "installed nnUNet predictor does not expose an in-memory logits API; "
            "use the repository's preprocessed-file adapter for that version"
        )


__all__ = ["NNUNetPredictorAdapter"]
