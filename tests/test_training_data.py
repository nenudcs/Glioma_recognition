from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from training.dataset import BrainNiftiDataset
from training.study_dataset import StudyNiftiDataset, StudyRecord
from training.resize import resize_volume


class TrainingDataTest(unittest.TestCase):
    def test_resize_uses_nearest_for_masks(self) -> None:
        image = np.zeros((2, 2, 2), dtype=np.float32)
        image[0, 0, 0] = 1
        mask = resize_volume(image, (4, 4, 4), is_mask=True)
        self.assertEqual(np.uint8, mask.dtype)
        self.assertEqual({0, 1}, set(np.unique(mask)))

    def test_dataset_loads_one_image_and_selected_mask_lazily(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "ACC" / "SERIES"
            directory.mkdir(parents=True)
            image = np.ones((3, 4, 5), dtype=np.float32)
            nib.save(nib.Nifti1Image(image, np.eye(4)), directory / "SERIES.nii.gz")
            nib.save(nib.Nifti1Image((image > 0).astype(np.float32), np.eye(4)), directory / "水肿_2_mask.nii.gz")
            index = root / "file_index.csv"
            with index.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "AccessionNumber", "SeriesUid", "SeriesType", "series_dir",
                        "series_path", "series_file_exists", "mask_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "AccessionNumber": "ACC", "SeriesUid": "SERIES",
                        "SeriesType": "FLAIR", "series_dir": str(directory),
                        "series_path": str(directory / "SERIES.nii.gz"),
                        "series_file_exists": "true", "mask_count": "1",
                    }
                )
            dataset = BrainNiftiDataset(index, task="segmentation", mask_name="水肿")
            self.assertEqual(1, len(dataset))
            item = dataset[0]
            self.assertEqual((1, 3, 4, 5), tuple(item["image"].shape))
            self.assertEqual((1, 3, 4, 5), tuple(item["mask"].shape))

    def test_study_dataset_zero_fills_missing_modality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "t1.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 4, 5), dtype=np.float32), np.eye(4)), image_path)
            record = StudyRecord(
                accession="ACC",
                modalities={"t1": str(image_path), "t1ce": None, "t2": None, "flair": None},
                label=1.0,
                core_mask=None,
                abnormal_mask=None,
            )
            item = StudyNiftiDataset([record])[0]
            self.assertEqual((4, 32, 32, 16), tuple(item["image"].shape))
            self.assertEqual([1.0, 0.0, 0.0, 0.0], item["modality_present"].tolist())


if __name__ == "__main__":
    unittest.main()
