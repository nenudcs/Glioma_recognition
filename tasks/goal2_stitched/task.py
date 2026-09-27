"""[运行] 目标二插件入口（规范 §5.1 / §7；比赛 Pipeline 只能通过本文件进入 Goal2）。

一次检查里同时完成两件事（都是「特殊影像」识别，且都要在写 ``prediction.json`` 之前
决定是否对下游任务上闸）：

1. **拼接**：本包的规则模型（``inference.py`` → ``model.py``）；
2. **重复**：可选的 ``duplicate_probe``（``tasks/goal2_duplicate`` 的
   ``DuplicateProbe``），命中结果同样写进 ``context.diagnostics["goal2"]``。

本文件只 import 本包的 ``config``/``inference``/``postprocess``/``gating``、
``tasks.base``/``tasks.results``；**不 import** 任何训练/标定模块
（``dataset`` / ``evaluate``），也不读比赛目录、不写结果文件。
"""
from __future__ import annotations

import logging
import time

from tasks.base import StudyTask
from tasks.results import StitchedResult

from .config import Goal2StitchedConfig
from .gating import DIAGNOSTIC_KEY, gate_reason, mark_gate
from .inference import StitchedInference
from .postprocess import StudyScore, diagnostics, series_types_of

logger = logging.getLogger(__name__)


class Goal2StitchedTask(StudyTask[StitchedResult]):
    """目标二：拼接影像检查级概率（``StitchedResult.stitched_probability``）。

    ``duplicate_probe`` 为 ``None`` 时只做拼接检测；``tasks/real_pipeline.py``
    默认注入重复检测探针，这样两条检测都发生在**同一个任务链位置**，
    下游闸门只需要看一个诊断键。
    """

    name = "goal2_stitched"

    def __init__(
        self,
        config: Goal2StitchedConfig | None = None,
        duplicate_probe: object | None = None,
    ) -> None:
        self.config = config or Goal2StitchedConfig.from_env()
        self.inference = StitchedInference(self.config)
        self.duplicate = duplicate_probe

    # -- 生命周期 -----------------------------------------------------------
    def load_model(self) -> None:
        self.inference.load()
        loader = getattr(self.duplicate, "load", None)
        if callable(loader):
            loader()

    # -- 推理 ---------------------------------------------------------------
    def predict(self, context) -> StitchedResult:
        study = context.study
        started = time.perf_counter()
        series_types = series_types_of(study.series)

        if self.config.enabled:
            study_score = self.inference.score_study(study, context.warnings)
            probability = self.inference.probability(study_score)
        else:
            study_score = StudyScore(0.0, ())
            probability = 0.0

        duplicate_detail: dict[str, object] = {"enabled": False}
        if self.duplicate is not None:
            duplicate_detail = self.duplicate.match(study, warnings=context.warnings)

        duration_ms = round((time.perf_counter() - started) * 1000)
        detail = {
            "task_name": self.name,
            "model_version": self.inference.model_version,
            **diagnostics(
                study_score,
                probability,
                self.config,
                series_types,
                duration_ms,
            ),
        }
        detail["duplicate"] = duplicate_detail
        context.diagnostics[DIAGNOSTIC_KEY] = detail

        reason = self._gate_reason(study_score, duplicate_detail)
        if reason is not None:
            mark_gate(context, reason[0], reason[1])

        logger.info(
            "goal2_stitched: accession=%s prob=%.6f score=%.6f flagged=%s duplicate=%s "
            "duration_ms=%d model_version=%s",
            study.accession_number,
            probability,
            study_score.score,
            self.config.enabled and study_score.score >= self.config.threshold,
            duplicate_detail.get("flagged", False),
            duration_ms,
            self.inference.model_version,
        )
        return StitchedResult(stitched_probability=float(probability))

    # -- 闸门 ---------------------------------------------------------------
    def _gate_reason(
        self,
        study_score: StudyScore,
        duplicate_detail: dict[str, object],
    ) -> tuple[str, dict[str, object]] | None:
        """返回 ``(原因, 细节)``；``None`` 表示不上闸。"""
        if duplicate_detail.get("flagged") and duplicate_detail.get("gate"):
            return (
                "duplicate",
                {
                    "matches": duplicate_detail.get("gate_matches", []),
                    "probability": duplicate_detail.get("best_probability", 0.0),
                },
            )
        if (
            self.config.enabled
            and self.config.gate
            and study_score.score >= self.config.threshold
        ):
            return (
                "stitched",
                {
                    "score": round(float(study_score.score), 6),
                    "threshold": self.config.threshold,
                },
            )
        return None
