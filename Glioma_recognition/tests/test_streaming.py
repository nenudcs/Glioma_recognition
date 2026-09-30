from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from openpyxl import Workbook

from core.config import Settings
from core.exceptions import InvalidInputError, OutputValidationError
from core.runner import EvaluationJob, EvaluationRunner
from data.loader import DatasetLoader, tolerant_mode
from output.validator import OutputValidator


class StreamingTest(unittest.TestCase):
    def test_loader_selects_exact_original_in_series_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            directory = root / "ACC001" / "SERIES-A"
            directory.mkdir(parents=True)

            self._save_image(directory / "SERIES-A.nii.gz", 0)
            self._save_image(directory / "SERIES-A(1).nii.gz", 1)
            self._save_image(directory / "SERIES-A(2)(1).nii.gz", 2)
            (directory / "SERIES-A.json").write_text(
                json.dumps({"ProtocolName": "from-original-sidecar"}),
                encoding="utf-8",
            )

            study = next(DatasetLoader().iter_studies(root))

            self.assertEqual(1, len(study.series))
            self.assertEqual("SERIES-A.nii.gz", study.series[0].source_path.name)
            self.assertEqual("SERIES-A", study.series[0].series_uid)
            self.assertEqual("from-original-sidecar", study.series[0].modality)

    def test_loader_rejects_multiple_files_without_exact_original(self) -> None:
        """无「主名 == 目录名」唯一原件时：fail-fast 报错 / 容错模式只跳过该序列。

        上游 README §3 要求「没有唯一匹配时明确报错」；正式评测默认打开
        ``GLIOMA_LOADER_TOLERANT`` 时改为跳过该序列目录 —— 1 个脏目录不该换整批 0 分。
        两种模式都**不得**静默采纳错误的影像：fail-fast 报错，容错模式则必须让同检查
        下**正常的那一路序列照常产出**。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            directory = root / "ACC001" / "SERIES"
            directory.mkdir(parents=True)
            self._save_image(directory / "first.nii", 1)
            self._save_image(directory / "second.nii", 2)
            good = root / "ACC001" / "SERIES-OK"
            good.mkdir(parents=True)
            self._save_image(good / "SERIES-OK.nii.gz", 3)

            if tolerant_mode():
                study = next(DatasetLoader().iter_studies(root))
                self.assertEqual(["SERIES-OK"], [s.series_uid for s in study.series])
                return
            with self.assertRaisesRegex(InvalidInputError, "exactly one original"):
                tuple(DatasetLoader().iter_studies(root))

    def test_loader_uses_xlsx_series_type_without_replacing_uid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            for series_uid in ("SERIES-A", "SERIES-B"):
                directory = root / "ACC001" / series_uid
                directory.mkdir(parents=True)
                self._save_image(directory / f"{series_uid}.nii.gz", 0)

            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["AccessionNumber", "SeriesUid", "SeriesType"])
            sheet.append(["a cc001", "series- a", "T1CE (增强)"])
            workbook.save(root / "SeriesType.xlsx")

            study = next(DatasetLoader().iter_studies(root))
            by_uid = {series.series_uid: series for series in study.series}

            self.assertEqual("T1CE (增强)", by_uid["SERIES-A"].modality)
            self.assertEqual("SERIES-A", by_uid["SERIES-A"].series_uid)
            self.assertEqual("SERIES-B", by_uid["SERIES-B"].modality)

    def test_loader_rejects_conflicting_xlsx_series_types(self) -> None:
        """表里同键冲突：fail-fast 报错 / 容错模式整表降级为「无表继续」。

        容错模式降级后描述退回 ``series_uid``（本例无 sidecar），模态识别随后接手
        兜底 —— 比让整批评测因为一行脏标注作废要合适。
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            directory = root / "ACC001" / "SERIES-A"
            directory.mkdir(parents=True)
            self._save_image(directory / "SERIES-A.nii.gz", 0)

            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["AccessionNumber", "SeriesUid", "SeriesType"])
            sheet.append(["ACC001", "SERIES-A", "T1"])
            sheet.append(["A CC001", "SERIES- A", "T2"])
            workbook.save(root / "SeriesType.xlsx")

            if tolerant_mode():
                study = next(DatasetLoader().iter_studies(root))
                self.assertEqual(["SERIES-A"], [s.series_uid for s in study.series])
                self.assertEqual("SERIES-A", study.series[0].modality)
                return
            with self.assertRaisesRegex(InvalidInputError, "conflicting SeriesType"):
                next(DatasetLoader().iter_studies(root))

    def test_loader_reads_only_the_current_study(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path = Path(temporary) / "dataset"
            self._make_dataset(dataset_path)

            with patch("data.loader.nib.load", wraps=nib.load) as load:
                studies = DatasetLoader().iter_studies(dataset_path)
                self.assertEqual(0, load.call_count)
                self.assertEqual("ACC001", next(studies).accession_number)
                self.assertEqual(2, load.call_count)
                self.assertEqual("ACC002", next(studies).accession_number)
                self.assertEqual(4, load.call_count)

    def test_runner_validates_before_loading_the_next_study(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            self._make_dataset(dataset_path)
            events: list[str] = []

            class RecordingLoader(DatasetLoader):
                def iter_studies(self, path: str | Path):
                    for study in super().iter_studies(path):
                        events.append(f"load {study.accession_number}")
                        yield study

            class RecordingValidator(OutputValidator):
                def validate_study(self, directory: Path, study: object) -> None:
                    super().validate_study(directory, study)
                    events.append(f"validate {study.accession_number}")

            runner = EvaluationRunner(
                self._settings(root),
                loader=RecordingLoader(),
                validator=RecordingValidator(),
            )
            runner.run(
                EvaluationJob("request-order", "evaluation-order", dataset_path),
                send_callback=False,
            )

            self.assertEqual(
                [
                    "load ACC001",
                    "validate ACC001",
                    "load ACC002",
                    "validate ACC002",
                ],
                events,
            )

    def test_failed_study_removes_staging_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            self._make_dataset(dataset_path)

            class FailingValidator(OutputValidator):
                def validate_study(self, directory: Path, study: object) -> None:
                    raise OutputValidationError("expected failure")

            settings = self._settings(root)
            runner = EvaluationRunner(settings, validator=FailingValidator())
            with self.assertRaises(Exception) as raised:           # noqa: B017 - 下面按名字精确断言
                runner.run(
                    EvaluationJob("request-fail", "evaluation-fail", dataset_path),
                    send_callback=False,
                )

            # fail-fast（默认）：单例失败 → 整批作废，抛 OutputValidationError。
            # 容错模式（正式评测）：单例被跳过；本测试两例都失败 → 因「没有任何一例
            # 成功产出」抛 ValueError。**两种模式都不得留下 staging 或已发布目录。**
            self.assertEqual(
                "ValueError" if tolerant_mode() else "OutputValidationError",
                type(raised.exception).__name__,
            )
            self.assertFalse((settings.answer_root / "evaluation-fail").exists())
            self.assertEqual([], list(settings.answer_root.glob(".*.tmp-*")))

    def test_loader_failure_does_not_log_the_previous_accession(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset"
            self._make_dataset(dataset_path)

            class FailingAfterFirstLoader(DatasetLoader):
                def iter_studies(self, path: str | Path):
                    studies = super().iter_studies(path)
                    yield next(studies)
                    raise InvalidInputError("cannot load next study")

            settings = self._settings(root)
            runner = EvaluationRunner(settings, loader=FailingAfterFirstLoader())
            with self.assertRaisesRegex(InvalidInputError, "next study"):
                runner.run(
                    EvaluationJob("request-log", "evaluation-log", dataset_path),
                    send_callback=False,
                )

            records = [
                json.loads(line)
                for line in (settings.log_root / "inference.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertIsNone(records[-1]["accession_number"])

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
        for accession in ("ACC001", "ACC002"):
            for series_uid in ("FLAIR", "T1CE"):
                directory = root / accession / series_uid
                directory.mkdir(parents=True)
                nib.save(
                    nib.Nifti1Image(
                        np.zeros((2, 3, 4), dtype=np.float32),
                        np.eye(4),
                    ),
                    str(directory / f"{series_uid}.nii.gz"),
                )

    @staticmethod
    def _save_image(path: Path, value: float) -> None:
        nib.save(
            nib.Nifti1Image(
                np.full((2, 3, 4), value, dtype=np.float32),
                np.eye(4),
            ),
            str(path),
        )


if __name__ == "__main__":
    unittest.main()
