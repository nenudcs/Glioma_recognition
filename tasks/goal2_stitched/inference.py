"""[运行] 拼接检测的纯推理入口：一个 Study → 拼接分数/概率。

只接收已加载的 ``Series.image``（不读比赛目录、不写结果），因此：

* 比赛链路零额外 I/O（Loader 已经把当前检查的序列读进内存）；
* 单条序列打分失败可降级（记 ``context.warnings``，继续其它序列）；
* 全部序列都失败时给出分数 0.0（= 无证据），不抛异常——与 Goal1 不同，
  拼接检测的「无证据」不会破坏输出格式，报错反而会让整个 evaluation 失败。
"""
from __future__ import annotations

import logging

from .config import Goal2StitchedConfig
from .model import probability_from_score, volume_score, worst_slice, slice_residual_scores, reduce_scores
from .postprocess import SeriesScore, StudyScore, aggregate_series

logger = logging.getLogger(__name__)


class StitchedInference:
    """规则式拼接检测器（无权重，``load_model()`` 只做配置校验）。"""

    model_version = "rule-stitched-v1"

    def __init__(self, config: Goal2StitchedConfig | None = None) -> None:
        self.config = config or Goal2StitchedConfig.from_env()

    @property
    def ready(self) -> bool:
        return True

    def load(self) -> None:
        logger.info("goal2_stitched: %s", self.config.describe())

    # -- 单条序列 -----------------------------------------------------------
    def score_series(self, image, series_uid: str = "", series_type: str = "") -> SeriesScore:
        try:
            scores = slice_residual_scores(image, self.config.metric, self.config.band)
            value = reduce_scores(scores, self.config.statistic)
            return SeriesScore(
                series_uid=str(series_uid),
                score=float(value),
                worst_slice=worst_slice(scores),
                slices=int(scores.size),
                series_type=str(series_type),
            )
        except Exception as exc:  # noqa: BLE001 - 单序列可降级
            return SeriesScore(
                series_uid=str(series_uid),
                score=None,
                worst_slice=None,
                slices=0,
                series_type=str(series_type),
                error=f"{type(exc).__name__}: {exc}",
            )

    # -- 检查级 -------------------------------------------------------------
    def score_study(self, study, warnings: list[str] | None = None) -> StudyScore:
        scores: list[SeriesScore] = []
        for series in study.series:
            item = self.score_series(
                series.image,
                getattr(series, "series_uid", ""),
                str(getattr(series, "modality", "") or ""),
            )
            if item.error is not None and warnings is not None:
                warnings.append(
                    f"goal2_stitched: 序列 {item.series_uid} 打分失败：{item.error}"
                )
            scores.append(item)
        return aggregate_series(scores)

    def volume_score(self, image) -> float:
        """兼容入口：单条体数据的拼接分数（标定/自检脚本使用）。"""
        return float(
            volume_score(image, self.config.statistic, self.config.metric, self.config.band)
        )

    def probability(self, study_score: StudyScore) -> float:
        return probability_from_score(
            study_score.score,
            self.config.threshold,
            self.config.scale,
        )
