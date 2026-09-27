"""[运行] 重复影像的纯推理入口：一个 Study → 命中列表（含成对概率）。

``DuplicateProbe`` 由 ``tasks/goal2_stitched/task.py`` 在**检查级**调用（与拼接检测同一个
任务链位置），这样做有两个原因：

1. 闸门必须在写 ``prediction.json`` 之前决定，而 ``DatasetTask.update()`` 在检查级任务
   之后才执行；
2. 重复检测需要「当前检查的序列已经在内存里」（``Series.image``），
   探针直接复用，不再二次读盘。

状态与生命周期：

* ``load()``：服务启动时调用一次（只打印配置）；
* ``reset()``：每次 evaluation 开始时清空索引与命中记录（由 ``Goal2DuplicateRecorder.reset()``
  调用），保证不泄漏上一次评测的状态；
* ``match()``：第一次调用时按需建索引（``scan=True`` 时先扫全库；``False`` 时增量比对）。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from .config import Goal2DuplicateConfig
from .postprocess import DuplicateMatch
from .preprocess import dataset_root_of, descriptors_from_series
from .retrieval import NearDuplicateIndex, ScanReport

logger = logging.getLogger(__name__)


class DuplicateProbe:
    """逐例重复检测探针（检查级，由 Goal2 特殊影像任务持有）。"""

    name = "goal2_duplicate"
    model_version = "near-duplicate-v1"

    def __init__(self, config: Goal2DuplicateConfig | None = None) -> None:
        self.config = config or Goal2DuplicateConfig.from_env()
        self.index: NearDuplicateIndex | None = None
        self.scan_report = ScanReport()
        self._scanned_root: Path | None = None

    # -- 生命周期 -----------------------------------------------------------
    def load(self) -> None:
        logger.info("goal2_duplicate: %s", self.config.describe())

    def reset(self) -> None:
        """清空一次 evaluation 的全部状态（规范 §18.2：不得泄漏上一轮结果）。"""
        self.index = None
        self.scan_report = ScanReport()
        self._scanned_root = None

    @property
    def volumes(self) -> int:
        return 0 if self.index is None else self.index.volumes

    @property
    def studies(self) -> int:
        return 0 if self.index is None else len(self.index)

    # -- 索引 ---------------------------------------------------------------
    def _new_index(self) -> NearDuplicateIndex:
        return NearDuplicateIndex(
            self.config.descriptor_params(),
            mode=self.config.mode,
            **self.config.index_kwargs(),
        )

    def _ensure_index(self, study, warnings: list[str] | None = None) -> None:
        if self.index is not None:
            return
        self.index = self._new_index()
        first = study.series[0]
        root = dataset_root_of(
            first.source_path,
            study.accession_number,
            first.series_uid,
        )
        self._scanned_root = root
        if not self.config.scan:
            self.scan_report = ScanReport(root=str(root))
            return
        try:
            self.index, self.scan_report = NearDuplicateIndex.from_dataset(
                root,
                params=self.config.descriptor_params(),
                max_volumes=self.config.scan_max_volumes,
                workers=self.config.scan_workers,
                mode=self.config.mode,
                **self.config.index_kwargs(),
            )
            logger.info(
                "goal2_duplicate: 全库索引完成 root=%s %s",
                root,
                self.scan_report.as_dict(),
            )
            if self.scan_report.skipped and warnings is not None:
                warnings.append(
                    f"goal2_duplicate: 全库扫描跳过 {self.scan_report.skipped} 条序列"
                    f"（示例：{self.scan_report.errors[:1]}）"
                )
        except Exception as exc:  # noqa: BLE001 - 扫描失败退回增量模式
            logger.error("goal2_duplicate: 全库扫描失败 root=%s %s", root, exc)
            self.index = self._new_index()
            self.scan_report = ScanReport(root=str(root))
            self.scan_report.errors.append(f"{type(exc).__name__}: {exc}")
            if warnings is not None:
                warnings.append(f"goal2_duplicate: 全库扫描失败，退回增量模式（{exc}）")

    # -- 逐例比对 -----------------------------------------------------------
    def descriptors(self, study):
        """从当前检查已加载的影像抽描述子（不额外读盘）。"""
        return descriptors_from_series(
            ((series.series_uid, series.image) for series in study.series),
            self.config.descriptor_params(),
        )

    def matches(self, study) -> list[DuplicateMatch]:
        """当前检查的命中列表（概率降序）。失败时返回空列表，不抛异常。"""
        if not self.config.enabled:
            return []
        self._ensure_index(study)
        assert self.index is not None
        descriptors = self.descriptors(study)
        matched = self.index.match(study.accession_number, descriptors)
        if not self.config.scan:
            # 增量模式：本检查特征入库，供后续检查比对
            self.index.add(study.accession_number, descriptors)
        floor = self.config.report_min_prob
        return [item for item in matched if item.probability >= floor]

    def match(self, study, warnings: list[str] | None = None) -> dict[str, object]:
        """检查级入口：返回写入 ``context.diagnostics["goal2"]["duplicate"]`` 的字典。"""
        started = time.perf_counter()
        detail: dict[str, object] = {
            "enabled": bool(self.config.enabled),
            "mode": self.config.mode,
            "gate": bool(self.config.gate),
            "gate_probability": self.config.gate_probability,
            "flagged": False,
            "best_probability": 0.0,
            "matches": [],
            "gate_matches": [],
            "match_count": 0,
            "index_studies": self.studies,
            "index_volumes": self.volumes,
            "scanned_root": None if self._scanned_root is None else str(self._scanned_root),
            "scan": self.scan_report.as_dict(),
        }
        if not self.config.enabled:
            detail["duration_ms"] = 0
            return detail
        try:
            matched = self.matches(study)
        except Exception as exc:  # noqa: BLE001 - 重复检测失败不应终止 evaluation
            logger.error("goal2_duplicate: 比对失败 accession=%s %s", study.accession_number, exc)
            if warnings is not None:
                warnings.append(f"goal2_duplicate: 比对失败（{type(exc).__name__}: {exc}）")
            detail["error"] = f"{type(exc).__name__}: {exc}"
            detail["duration_ms"] = round((time.perf_counter() - started) * 1000)
            return detail

        best = max((item.probability for item in matched), default=0.0)
        strong = [
            item.accession
            for item in matched
            if item.probability >= self.config.gate_probability
        ]
        detail.update(
            {
                "flagged": bool(strong),
                "best_probability": round(float(best), 6),
                "matches": [item.as_dict() for item in matched],
                "gate_matches": strong,
                "match_count": len(matched),
                "index_studies": self.studies,
                "index_volumes": self.volumes,
                "scanned_root": None if self._scanned_root is None else str(self._scanned_root),
                "scan": self.scan_report.as_dict(),
                "duration_ms": round((time.perf_counter() - started) * 1000),
            }
        )
        return detail
