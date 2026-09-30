from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from core.config import Settings
from core.exceptions import OutputValidationError
from core.runner import EvaluationJob, EvaluationRunner
from data.loader import DatasetLoader
from output.validator import OutputValidator


class EndToEndTest(unittest.TestCase):
    def test_dummy_pipeline_writes_valid_competition_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            self._make_dataset(dataset_path)
            settings = self._settings(root)

            output = EvaluationRunner(settings).run(
                EvaluationJob("request-1", "evaluation-1", dataset_path),
                send_callback=False,
            )

            self.assertTrue((output / "duplicate_pairs.jsonl").is_file())
            for accession in ("ACC001", "ACC002"):
                prediction_path = output / accession / "prediction.json"
                payload = json.loads(prediction_path.read_text(encoding="utf-8"))
                self.assertEqual(accession, payload["AccessionNumber"])
                self.assertEqual(0.0, payload["Prediction"]["TumorProbability"])
                self.assertEqual({"core", "flair"}, set(payload["SegmentationMaskURI"]))

            dataset = DatasetLoader().load(dataset_path)
            OutputValidator().validate(output, dataset)

    def test_validator_rejects_non_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            self._make_dataset(dataset_path)
            settings = self._settings(root)
            output = EvaluationRunner(settings).run(
                EvaluationJob("request-2", "evaluation-2", dataset_path),
                send_callback=False,
            )
            prediction_path = output / "ACC001" / "prediction.json"
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
            mask_path = output / "ACC001" / prediction["SegmentationMaskURI"]["core"]
            mask_image = nib.load(str(mask_path))
            bad = np.asanyarray(mask_image.dataobj).copy()
            bad.flat[0] = 2
            nib.save(nib.Nifti1Image(bad, mask_image.affine), str(mask_path))

            with self.assertRaises(OutputValidationError):
                OutputValidator().validate(output, DatasetLoader().load(dataset_path))

    @staticmethod
    def _settings(root: Path) -> Settings:
        return Settings(
            workspace=root / "workspace",
            answer_root=root / "workspace" / "answer",
            log_root=root / "workspace" / "logs",
            callback_url=None,
        )

    @staticmethod
    def _make_dataset(root: Path) -> None:
        affine = np.array(
            [
                [1.2, 0.0, 0.0, -10.0],
                [0.0, 1.2, 0.0, -20.0],
                [0.0, 0.0, 3.0, 5.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        for index, accession in enumerate(("ACC001", "ACC002")):
            for series_uid in ("T1CE", "FLAIR"):
                directory = root / accession / series_uid
                directory.mkdir(parents=True)
                image = np.full((5, 6, 4), index, dtype=np.float32)
                nib.save(
                    nib.Nifti1Image(image, affine),
                    str(directory / f"{series_uid}.nii.gz"),
                )


if __name__ == "__main__":
    unittest.main()

