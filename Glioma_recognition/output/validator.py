from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from core.exceptions import OutputValidationError
from data.structures import CompetitionDataset, Series
from output.schemas import DUPLICATE_KEYS, MASK_KEYS, PREDICTION_REQUIRED_KEYS


class OutputValidator:
    def validate(
        self,
        evaluation_dir: Path,
        dataset: CompetitionDataset,
    ) -> None:
        expected_accessions = {study.accession_number for study in dataset.studies}
        self.validate_final_layout(evaluation_dir, expected_accessions)
        self.validate_duplicates(
            evaluation_dir / "duplicate_pairs.jsonl",
            expected_accessions,
        )
        for study in dataset.studies:
            self.validate_study(evaluation_dir / study.accession_number, study)

    def validate_final_layout(
        self,
        evaluation_dir: Path,
        expected_accessions: set[str],
    ) -> None:
        if not evaluation_dir.is_dir():
            self._fail(f"missing evaluation directory: {evaluation_dir}")

        actual_accessions = {
            path.name for path in evaluation_dir.iterdir() if path.is_dir()
        }
        if actual_accessions != expected_accessions:
            self._fail(
                "study directories do not match dataset: "
                f"expected={sorted(expected_accessions)}, actual={sorted(actual_accessions)}"
            )

    def validate_study(self, directory: Path, study: Any) -> None:
        prediction_path = directory / "prediction.json"
        payload = self._read_json(prediction_path)
        missing = PREDICTION_REQUIRED_KEYS - payload.keys()
        if missing:
            self._fail(f"{prediction_path} missing keys: {sorted(missing)}")
        if payload.get("AccessionNumber") != study.accession_number:
            self._fail(f"wrong AccessionNumber in {prediction_path}")
        self._validate_numbers(payload, str(prediction_path))

        mask_uris = payload.get("SegmentationMaskURI")
        if not isinstance(mask_uris, dict) or set(mask_uris) != MASK_KEYS:
            self._fail(f"invalid SegmentationMaskURI in {prediction_path}")
        for label, uri in mask_uris.items():
            if not isinstance(uri, str):
                self._fail(f"{label} mask URI is not a string")
            relative = Path(uri)
            if relative.is_absolute():
                self._fail(f"mask URI must be relative: {uri}")
            mask_path = (directory / relative).resolve()
            try:
                mask_path.relative_to(directory.resolve())
            except ValueError:
                self._fail(f"mask URI escapes study directory: {uri}")
            series_uid = mask_path.parent.name
            try:
                source = study.series_by_uid(series_uid)
            except KeyError:
                self._fail(f"mask URI references unknown series: {uri}")
            self._validate_mask(mask_path, source)

    def validate_duplicates(self, path: Path, accessions: set[str]) -> None:
        if not path.is_file():
            self._fail(f"missing {path}")
        pairs: set[tuple[str, str]] = set()
        counts: Counter[str] = Counter()
        line_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                line_count += 1
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    self._fail(f"invalid JSONL at {path}:{line_number}: {exc}")
                if not isinstance(item, dict) or set(item) != DUPLICATE_KEYS:
                    self._fail(f"invalid duplicate fields at {path}:{line_number}")
                left = item["StudyUID"]
                right = item["StudyUID_dup"]
                if left not in accessions or right not in accessions:
                    self._fail(f"unknown study in duplicate pair at line {line_number}")
                if left == right:
                    self._fail(f"self-pair at {path}:{line_number}")
                key = tuple(sorted((left, right)))
                if key in pairs:
                    self._fail(f"duplicate pair at {path}:{line_number}")
                pairs.add(key)
                counts[left] += 1
                counts[right] += 1
                self._probability(item["PairProb"], f"{path}:{line_number}.PairProb")

        if line_count == 0:
            self._fail(f"{path} must contain at least one valid record")
        over_limit = [accession for accession, count in counts.items() if count > 200]
        if over_limit:
            self._fail(f"more than 200 duplicate candidates for {over_limit}")

    def _validate_mask(self, path: Path, source: Series) -> None:
        if not path.is_file():
            self._fail(f"missing mask: {path}")
        try:
            image = nib.load(str(path))
            array = np.asanyarray(image.dataobj)
        except Exception as exc:
            self._fail(f"cannot reload mask {path}: {exc}")
        if array.shape != source.image.shape:
            self._fail(
                f"mask shape mismatch for {path}: {array.shape} != {source.image.shape}"
            )
        # affine 容差取 1e-4（而非 1e-5）：
        #  · NIfTI 的 sform 以 **float32** 存储，坐标量级 ~200mm 时 float32 的
        #    ulp 约 1.5e-5，已大于 1e-5 —— 那是实现精度问题，不是数据不一致；
        #  · 用 1e-5 会把几何完全正确的正常病例判为不合法，进而让**整个
        #    evaluation 失败**（代价远大于容差放宽带来的风险）。
        # shape 仍然要求**精确相等**（不受浮点影响），足以拦住真正错位的掩码。
        if not np.allclose(image.affine, source.affine, rtol=0.0, atol=1e-4):
            delta = float(np.abs(np.asarray(image.affine) - np.asarray(source.affine)).max())
            self._fail(f"mask affine mismatch for {path}: max|delta|={delta:.3g} (>1e-4)")
        if not np.isfinite(array).all():
            self._fail(f"mask contains non-finite values: {path}")
        values = set(np.unique(array).tolist())
        if not values <= {0, 1}:
            self._fail(f"mask is not binary: {path}, values={sorted(values)}")

    def _validate_numbers(self, item: Any, location: str, parent_key: str = "") -> None:
        if isinstance(item, dict):
            if parent_key == "probabilities":
                values = []
                for key, value in item.items():
                    self._probability(value, f"{location}.{key}")
                    values.append(float(value))
                total = sum(values)
                if values and not (
                    math.isclose(total, 0.0, abs_tol=1e-6)
                    or math.isclose(total, 1.0, abs_tol=1e-4)
                ):
                    self._fail(f"probabilities do not sum to 0 or 1 at {location}")
            else:
                for key, value in item.items():
                    child = f"{location}.{key}"
                    if "prob" in key.lower() and key != "probabilities":
                        self._probability(value, child)
                    else:
                        self._validate_numbers(value, child, key)
        elif isinstance(item, list):
            for index, value in enumerate(item):
                self._validate_numbers(value, f"{location}[{index}]", parent_key)
        elif isinstance(item, float) and not math.isfinite(item):
            self._fail(f"non-finite number at {location}")

    def _probability(self, value: Any, location: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._fail(f"probability is not numeric at {location}")
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            self._fail(f"probability outside [0,1] at {location}: {value}")

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            self._fail(f"cannot read JSON {path}: {exc}")
        if not isinstance(payload, dict):
            self._fail(f"JSON root must be an object: {path}")
        return payload

    @staticmethod
    def _fail(message: str) -> None:
        raise OutputValidationError(message)
