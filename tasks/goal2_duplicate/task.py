"""[运行] 重复影像的数据集级入口（规范 §7：``DatasetTask`` 的 reset/update/finalize）。

流程：

* ``Goal2StitchedTask``（检查级）已经算出每个检查的命中列表并写进
  ``context.diagnostics["goal2"]["duplicate"]["matches"]``；
* 本任务在 ``update()`` 里把这些命中整理成 ``DuplicatePair``（去重、丢弃自配对、
  每例最多 ``GOAL2_DUPLICATE_MAX_PAIRS``，默认 200）；
* ``finalize()`` 输出 ``DuplicateResult`` → 管线 Writer 按概率降序再次去重/截断后写
  ``duplicate_pairs.jsonl``；
* ``reset()`` 同时清空重复探针的索引，保证一次 evaluation 一份干净状态。

本文件只 import ``tasks.base`` / ``tasks.results`` 与同包的 ``config`` / ``inference`` /
``postprocess``；不 import 训练/标定模块。
"""
from __future__ import annotations

import logging
from collections import defaultdict

from tasks.base import DatasetTask
from tasks.results import DuplicatePair, DuplicateResult

from .config import Goal2DuplicateConfig
from .inference import DuplicateProbe
from .postprocess import normalize_pairs, pairs_from_matches

# 诊断键由 goal2_stitched 统一维护（两条检测写同一个 "goal2" 键）。
from tasks.goal2_stitched.gating import DIAGNOSTIC_KEY

logger = logging.getLogger(__name__)


class Goal2DuplicateRecorder(DatasetTask[DuplicateResult]):
    """数据集级重复病例汇总（``DuplicateResult`` → ``duplicate_pairs.jsonl``）。"""

    name = "goal2_duplicate"

    def __init__(
        self,
        config: Goal2DuplicateConfig | None = None,
        probe: DuplicateProbe | None = None,
        max_pairs_per_study: int | None = None,
    ) -> None:
        self.config = config or Goal2DuplicateConfig.from_env()
        self.probe = probe
        self.max_pairs_per_study = int(max_pairs_per_study or self.config.max_pairs_per_study)
        self._pairs: dict[tuple[str, str], float] = {}
        self._counts: dict[str, int] = defaultdict(int)

    # -- 生命周期 -----------------------------------------------------------
    def load_model(self) -> None:
        logger.info("goal2_duplicate: %s", self.config.describe())

    def reset(self) -> None:
        """新一轮 evaluation：清空配对记录，并重置重复探针的索引。"""
        self._pairs = {}
        self._counts = defaultdict(int)
        if self.probe is not None:
            self.probe.reset()

    def update(self, study, context) -> None:
        detail = context.diagnostics.get(DIAGNOSTIC_KEY, {})
        duplicate = detail.get("duplicate", {}) if detail else {}
        matches = duplicate.get("matches", []) if isinstance(duplicate, dict) else []
        for pair in pairs_from_matches(study.accession_number, matches):
            key = tuple(sorted((pair.left_accession, pair.right_accession)))
            if key[0] == key[1] or key in self._pairs:
                continue
            if self._counts[key[0]] >= self.max_pairs_per_study:
                continue
            if self._counts[key[1]] >= self.max_pairs_per_study:
                continue
            self._pairs[key] = float(pair.probability)
            self._counts[key[0]] += 1
            self._counts[key[1]] += 1

    def finalize(self) -> DuplicateResult:
        pairs = tuple(
            DuplicatePair(left, right, probability)
            for (left, right), probability in sorted(self._pairs.items())
        )
        selected = normalize_pairs(pairs, self.max_pairs_per_study)
        logger.info(
            "goal2_duplicate: %d 个候选对（截断后 %d 个）",
            len(pairs),
            len(selected),
        )
        return DuplicateResult(pairs=tuple(selected))
