from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np

from data.structures import Series
from pipeline.aggregator import PredictionAggregator
from pipeline.context import PipelineContext
from tasks.results import DuplicatePair, DuplicateResult


class OutputWriter:
    def __init__(
        self,
        answer_root: Path,
        aggregator: PredictionAggregator | None = None,
    ) -> None:
        self.answer_root = answer_root
        self.aggregator = aggregator or PredictionAggregator()

    def write_staging(
        self,
        evaluation_id: str,
        contexts: dict[str, PipelineContext],
        duplicates: DuplicateResult,
    ) -> Path:
        """Compatibility wrapper for callers that already hold every context."""
        staging = self.begin(evaluation_id)
        try:
            for _, context in sorted(contexts.items()):
                self.write_study(staging, context)
            self.write_duplicates(staging, set(contexts), duplicates)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return staging

    def begin(self, evaluation_id: str) -> Path:
        safe_evaluation_id = _safe_component(evaluation_id, "evaluation_id")
        self.answer_root.mkdir(parents=True, exist_ok=True)
        final_dir = self.answer_root / safe_evaluation_id
        if final_dir.exists():
            raise FileExistsError(f"evaluation output already exists: {final_dir}")

        staging = self.answer_root / f".{safe_evaluation_id}.tmp-{uuid.uuid4().hex}"
        staging.mkdir()
        return staging

    def write_study(self, staging: Path, context: PipelineContext) -> Path:
        accession_dir = staging / _safe_component(
            context.study.accession_number,
            "accession_number",
        )
        accession_dir.mkdir()
        self._write_masks(accession_dir, context)
        self._write_json(
            accession_dir / "prediction.json",
            self.aggregator.build(context),
        )
        return accession_dir

    def write_duplicates(
        self,
        staging: Path,
        accessions: set[str],
        duplicates: DuplicateResult,
    ) -> Path:
        pairs = self._normalize_pairs(duplicates.pairs)
        if not pairs:
            ordered_accessions = sorted(accessions)
            if len(ordered_accessions) < 2:
                raise ValueError(
                    "duplicate_pairs.jsonl requires at least two studies for one valid pair"
                )
            pairs = [
                DuplicatePair(
                    ordered_accessions[0],
                    ordered_accessions[1],
                    0.0,
                )
            ]

        path = staging / "duplicate_pairs.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for pair in pairs:
                line = {
                    "StudyUID": pair.left_accession,
                    "StudyUID_dup": pair.right_accession,
                    "PairProb": pair.probability,
                }
                handle.write(
                    json.dumps(
                        line,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def publish(self, staging: Path, evaluation_id: str) -> Path:
        final_dir = self.answer_root / _safe_component(evaluation_id, "evaluation_id")
        if final_dir.exists():
            raise FileExistsError(f"evaluation output already exists: {final_dir}")
        staging.rename(final_dir)
        return final_dir

    def _write_masks(self, accession_dir: Path, context: PipelineContext) -> None:
        result = context.goal5
        if result is None:
            raise ValueError("Goal5 result is missing")

        planned: dict[Path, tuple[np.ndarray, Series]] = {}
        for mask, series_uid in (
            (result.core_mask, result.core_source_series_uid),
            (result.flair_mask, result.flair_source_series_uid),
        ):
            series = context.study.series_by_uid(series_uid)
            safe_uid = _safe_component(series_uid, "series_uid")
            path = accession_dir / safe_uid / f"{safe_uid}.nii.gz"
            if path in planned and not np.array_equal(planned[path][0], mask):
                raise ValueError(
                    f"core and flair masks target the same series but differ: {series_uid}"
                )
            planned[path] = (mask, series)

        for path, (mask, series) in planned.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            affine = np.asarray(series.affine, dtype=np.float64)
            image = nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), affine)
            image.set_data_dtype(np.uint8)
            # 显式固定几何，避免读回时的浮点差异导致校验失败：
            #  · sform 与 qform 写成同一 affine 且 code=1，保证 Validator 重读时
            #    走 sform（不再由 qform 四元数重建，那会引入 ~1e-5 量级误差）；
            #  · 清掉缩放字段，防止 0/1 掩码被 scl_slope/scl_inter 变换后
            #    读出非 {0,1}（规范规定掩码体素必须严格 0/1，否则该例分割记 0 分）。
            image.header.set_sform(affine, code=1)
            image.header.set_qform(affine, code=1)
            image.header["scl_slope"] = 1.0
            image.header["scl_inter"] = 0.0
            nib.save(image, str(path))

    @staticmethod
    def _normalize_pairs(pairs: tuple[DuplicatePair, ...]) -> list[DuplicatePair]:
        best: dict[tuple[str, str], DuplicatePair] = {}
        for pair in pairs:
            key = tuple(sorted((pair.left_accession, pair.right_accession)))
            if key[0] == key[1]:
                continue
            current = best.get(key)
            if current is None or pair.probability > current.probability:
                best[key] = DuplicatePair(key[0], key[1], pair.probability)

        counts: dict[str, int] = {}
        selected: list[DuplicatePair] = []
        for pair in sorted(
            best.values(),
            key=lambda item: (-item.probability, item.left_accession, item.right_accession),
        ):
            if counts.get(pair.left_accession, 0) >= 200:
                continue
            if counts.get(pair.right_accession, 0) >= 200:
                continue
            selected.append(pair)
            counts[pair.left_accession] = counts.get(pair.left_accession, 0) + 1
            counts[pair.right_accession] = counts.get(pair.right_accession, 0) + 1
        return selected

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def _safe_component(value: str, label: str) -> str:
    text = str(value).strip()
    if (
        not text
        or text in {".", ".."}
        or Path(text).is_absolute()
        or re.search(r"[\\/\x00-\x1f]", text)
    ):
        raise ValueError(f"unsafe {label}: {value!r}")
    return text
