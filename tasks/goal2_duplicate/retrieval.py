"""[运行] 全库近似重复索引：粗检索（余弦）+ 精排（平移互相关）+ Top-K。

「先扫全库」的流程（赛方要求「识别重复影像」必须能看到整个测试集）：

1. 处理第一个检查时，从 ``Series.source_path`` 反推数据集根目录，遍历全库；
2. 每条序列只抽**中间 K 层**算两级描述子（``preprocess.py``），
   粗描述子堆成一个 ``(N, D)`` 矩阵；
3. 当前检查的每条序列与全库做一次矩阵乘法取 Top 候选（亚二次，避免逐个 pair 比较），
   过滤掉粗检索余弦低于 ``coarse_floor`` 的候选；
4. 候选对在精描述子上做带平移搜索的 NCC（``model.slab_similarity``），
   相似度 ≥ ``min_sim`` 的按概率降序保留 ``top_k`` 条。

内存与时间：粗描述子约 0.5 KB/卷、精描述子约 12 KB/卷（float16），
单对精排约 2 ms；全库扫描是主要成本，可用 ``scan_workers`` 并行读盘。
"""
from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .model import study_similarity
from .postprocess import DuplicateMatch, probability_from_similarity
from .preprocess import (
    DescriptorParams,
    VolumeDescriptor,
    descriptor_from_path,
    group_dataset,
)

logger = logging.getLogger(__name__)

DEFAULT_COARSE_CANDIDATES = 64


@dataclass
class ScanReport:
    """全库扫描统计（写入诊断信息，便于排障与限时）。"""

    root: str | None = None
    studies: int = 0
    volumes: int = 0
    skipped: int = 0
    truncated: bool = False
    duration_ms: int | None = None
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "studies": self.studies,
            "volumes": self.volumes,
            "skipped": self.skipped,
            "truncated": self.truncated,
            "duration_ms": self.duration_ms,
            "errors": self.errors[:5],
        }


class NearDuplicateIndex:
    """全库近似重复索引（只保存轻量特征，不长期持有影像）。"""

    def __init__(
        self,
        params: DescriptorParams | None = None,
        *,
        mode: str = "near",
        top_k: int = 20,
        min_sim: float = 0.50,
        coarse_floor: float = 0.80,
        coarse_candidates: int = DEFAULT_COARSE_CANDIDATES,
        center: float = 0.90,
        scale: float = 0.02,
    ) -> None:
        self.params = params or DescriptorParams()
        if mode not in {"near", "exact"}:
            raise ValueError("mode 必须是 'near' 或 'exact'")
        self.mode = mode
        self.top_k = max(1, int(top_k))
        self.min_sim = float(min_sim)
        self.coarse_floor = float(coarse_floor)
        self.coarse_candidates = max(self.top_k, int(coarse_candidates))
        self.center = float(center)
        self.scale = float(scale)

        self._entries: list[tuple[str, str, VolumeDescriptor]] = []
        self._accession_rows: dict[str, list[int]] = defaultdict(list)
        self._by_accession: dict[str, list[tuple[str, VolumeDescriptor]]] = defaultdict(list)
        self._exact: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._matrix: np.ndarray | None = None

    # -- 写入 ---------------------------------------------------------------
    def add(self, accession: str, descriptors: Sequence[tuple[str, VolumeDescriptor]]) -> None:
        """把一个检查的全部序列特征加入索引（``reset()`` 后应重新加入）。"""
        for series_uid, descriptor in descriptors:
            row = len(self._entries)
            self._entries.append((str(accession), str(series_uid), descriptor))
            self._accession_rows[str(accession)].append(row)
            self._by_accession[str(accession)].append((str(series_uid), descriptor))
            self._exact[descriptor.exact].append((str(accession), str(series_uid)))
        self._matrix = None

    def __len__(self) -> int:
        return len(self._accession_rows)

    @property
    def volumes(self) -> int:
        return len(self._entries)

    def accessions(self) -> list[str]:
        return sorted(self._accession_rows)

    def descriptors_of(self, accession: str) -> list[tuple[str, VolumeDescriptor]]:
        return list(self._by_accession.get(str(accession), ()))

    def _ensure_matrix(self) -> tuple[np.ndarray, list[tuple[str, str, VolumeDescriptor]]]:
        if self._matrix is None:
            if not self._entries:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
            else:
                self._matrix = np.stack(
                    [entry[2].coarse for entry in self._entries]
                ).astype(np.float32, copy=False)
        return self._matrix, self._entries

    # -- 检索 ---------------------------------------------------------------
    def match(
        self,
        accession: str,
        descriptors: Sequence[tuple[str, VolumeDescriptor]],
        *,
        top_k: int | None = None,
    ) -> list[DuplicateMatch]:
        """返回与 ``accession`` 重复的其它检查（概率降序，最多 ``top_k`` 条）。"""
        if not descriptors:
            return []
        limit = self.top_k if top_k is None else max(1, int(top_k))
        matrix, entries = self._ensure_matrix()
        best: dict[str, DuplicateMatch] = {}

        # 1) 完全一致的拷贝：中间层像素相同 → 概率 1.0
        for series_uid, descriptor in descriptors:
            for other, other_series in self._exact.get(descriptor.exact, ()):
                if other == accession:
                    continue
                previous = best.get(other)
                if previous is None or previous.similarity < 1.0:
                    best[other] = DuplicateMatch(
                        accession=other,
                        probability=1.0,
                        similarity=1.0,
                        series_uid=str(series_uid),
                        other_series_uid=str(other_series),
                        exact=True,
                    )

        # 2) 粗检索 → 精排（``mode='exact'`` 时只保留指纹命中）
        if self.mode == "near" and matrix.size and matrix.shape[1] == descriptors[0][1].coarse.shape[0]:
            own_rows = self._accession_rows.get(str(accession), [])
            for series_uid, descriptor in descriptors:
                scores = matrix @ descriptor.coarse
                if own_rows:
                    scores[np.asarray(own_rows, dtype=np.int64)] = -1.0
                count = min(self.coarse_candidates, scores.shape[0])
                if count < 1:
                    continue
                if count < scores.shape[0]:
                    candidate_rows = np.argpartition(-scores, count - 1)[:count]
                else:
                    candidate_rows = np.arange(scores.shape[0])
                for row in candidate_rows:
                    if float(scores[row]) < self.coarse_floor:
                        continue
                    other_accession, other_series, other_descriptor = entries[int(row)]
                    if other_accession == accession:
                        continue
                    similarity = study_similarity(
                        [(series_uid, descriptor)],
                        [(other_series, other_descriptor)],
                        self.params,
                    )[0]
                    if similarity < self.min_sim:
                        continue
                    previous = best.get(other_accession)
                    if previous is not None and previous.similarity >= similarity:
                        continue
                    best[other_accession] = DuplicateMatch(
                        accession=other_accession,
                        probability=probability_from_similarity(
                            similarity, self.center, self.scale
                        ),
                        similarity=float(similarity),
                        series_uid=str(series_uid),
                        other_series_uid=str(other_series),
                    )

        ordered = sorted(best.values(), key=lambda item: (-item.probability, item.accession))
        return ordered[:limit]

    def all_pairs(self, max_pairs_per_study: int = 200) -> list[tuple[str, str, float, float]]:
        """全库自比对：``(left, right, prob, sim)``，去重、按概率降序。"""
        best: dict[tuple[str, str], tuple[float, float]] = {}
        for accession in self.accessions():
            for match in self.match(
                accession,
                self.descriptors_of(accession),
                top_k=max_pairs_per_study,
            ):
                key = tuple(sorted((accession, match.accession)))
                if key[0] == key[1]:
                    continue
                current = best.get(key)
                if current is None or match.probability > current[0]:
                    best[key] = (match.probability, match.similarity)
        return [
            (left, right, probability, similarity)
            for (left, right), (probability, similarity) in sorted(
                best.items(), key=lambda item: (-item[1][0], item[0])
            )
        ]

    # -- 全库扫描 -----------------------------------------------------------
    @classmethod
    def from_dataset(
        cls,
        root: str | Path,
        *,
        params: DescriptorParams | None = None,
        max_volumes: int = 0,
        accession_prefixes: Sequence[str] = (),
        skip_dirs: Sequence[Path] = (),
        workers: int = 1,
        **kwargs: object,
    ) -> tuple["NearDuplicateIndex", ScanReport]:
        """扫描整个数据集建立索引；``max_volumes > 0`` 时限制扫描条数。"""
        import time

        index = cls(params, **kwargs)          # type: ignore[arg-type]
        report = ScanReport(root=str(Path(root).expanduser()))
        grouped = group_dataset(root, skip_dirs=skip_dirs)
        if not grouped:
            return index, report

        budget = max_volumes if max_volumes > 0 else None
        tasks: list[tuple[str, str, Path]] = []
        for accession, series in grouped.items():
            if accession_prefixes and not accession.startswith(tuple(accession_prefixes)):
                continue
            for series_uid, path in series:
                if budget is not None and len(tasks) >= budget:
                    report.truncated = True
                    break
                tasks.append((accession, series_uid, path))
            if report.truncated:
                break

        def one(task: tuple[str, str, Path]):
            accession, series_uid, path = task
            try:
                return accession, series_uid, descriptor_from_path(path, index.params), None
            except Exception as exc:  # noqa: BLE001 - 单个文件读不了就跳过
                return accession, series_uid, None, f"{path}: {type(exc).__name__}: {exc}"

        started = time.perf_counter()
        if workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(one, tasks))
        else:
            results = [one(task) for task in tasks]

        collected: dict[str, list[tuple[str, VolumeDescriptor]]] = defaultdict(list)
        for accession, series_uid, descriptor, error in results:
            if descriptor is None:
                report.skipped += 1
                if len(report.errors) < 20 and error is not None:
                    report.errors.append(error)
                continue
            collected[accession].append((series_uid, descriptor))
            report.volumes += 1
        for accession in sorted(collected):
            index.add(accession, collected[accession])
            report.studies += 1
        report.duration_ms = round((time.perf_counter() - started) * 1000)
        return index, report
