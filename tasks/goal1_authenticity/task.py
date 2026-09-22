"""[运行] 目标一插件入口（规范 §5.1 / §7 / §9.1）。

**比赛 Pipeline 只能通过本文件进入 Goal1**；本文件只 import：

* 本包的 ``config`` / ``inference`` / ``postprocess`` / ``checkpoint``
* ``tasks.base`` / ``tasks.results``
* ``core.exceptions``

禁止 import ``dataset`` / ``augmentations`` / ``losses`` / ``train`` / ``evaluate``
（规范 §17.1「比赛运行只能从 task.py 进入 Goal，禁止传递导入训练专用模块」）。

失败语义（规范 §8.2 / §9.1）：

* 单条序列读不出/打不了分 → 记 ``context.warnings``，继续其它序列（可降级）；
* **整个 Study 没有任何可打分序列 → 抛 ``MissingSeriesError``（不可降级，终止该 evaluation）**；
* 权重没加载成功 → 抛 ``ModelInferenceError``（禁止输出看似成功的半成品）。
"""
from __future__ import annotations

import logging
import time

from core.exceptions import MissingSeriesError, ModelInferenceError
from tasks.base import StudyTask
from tasks.results import Goal1Result

from .config import Goal1Config
from .inference import AuthenticityInference
from .postprocess import SeriesScore, aggregate_scores, diagnostics

logger = logging.getLogger(__name__)


class Goal1AuthenticityTask(StudyTask[Goal1Result]):
    """目标一：非人体/伪造影像检查级概率（``Goal1Result.not_human_probability``）。"""

    name = "goal1_authenticity"

    def __init__(self, config: Goal1Config | None = None) -> None:
        self.config = config or Goal1Config.from_env()
        self.inference = AuthenticityInference(self.config)
        self.load_error: str | None = None

    # -- 生命周期 -----------------------------------------------------------
    def load_model(self) -> None:
        """服务初始化阶段调用一次（规范 §5.2：禁止逐 Study 读权重）。"""
        try:
            self.inference.load()
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"{type(exc).__name__}: {exc}"
            if self.config.strict:
                raise
            logger.error(
                "goal1_authenticity: 权重加载失败，预测时将对每例报错（%s）；"
                "确认 checkpoint 是否已迁移到规范路径",
                self.load_error,
            )
            return
        self.load_error = None

    def predict(self, context) -> Goal1Result:
        if not self.inference.ready:
            raise ModelInferenceError(
                f"goal1_authenticity 权重不可用：{self.load_error or 'model not loaded'}"
            )

        study = context.study
        started = time.perf_counter()
        scores: list[SeriesScore] = []
        for series in study.series:
            series_type = str(getattr(series, "modality", "") or "")
            try:
                probability = self.inference.score_volume(series.image)
                error = None if probability is not None else "empty volume"
            except Exception as exc:  # noqa: BLE001 - 单序列可降级
                probability = None
                error = f"{type(exc).__name__}: {exc}"
                context.warnings.append(
                    f"goal1_authenticity: 序列 {series.series_uid} 打分失败：{error}"
                )
            scores.append(SeriesScore(series.series_uid, probability, series_type, error))

        probability = aggregate_scores(scores, self.config.series_aggregation)
        if probability is None:
            # §9.1：整个 Study 没有有效影像 = 不可降级输入错误
            raise MissingSeriesError(
                f"study {study.accession_number!r} 没有任何可打分的序列"
            )

        duration_ms = round((time.perf_counter() - started) * 1000)
        context.diagnostics["goal1"] = {
            "task_name": self.name,
            "model_version": self.inference.model_version,
            "duration_ms": duration_ms,
            **diagnostics(scores, probability, self.config.series_aggregation),
        }
        logger.info(
            "goal1_authenticity: accession=%s probability=%.6f series=%d/%d duration_ms=%d "
            "model_version=%s",
            study.accession_number,
            probability,
            sum(1 for item in scores if item.probability is not None),
            len(scores),
            duration_ms,
            self.inference.model_version,
        )
        return Goal1Result(not_human_probability=float(probability))
