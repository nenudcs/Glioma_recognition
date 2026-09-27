"""[运行] 拼接检测的后处理：序列级 → 检查级聚合、分数 → 概率、诊断信息。

检查级分数 = 该检查所有序列分数的**最大值**（一层拼接就足以判定该检查）；
概率 = ``sigmoid(scale * (score / threshold - 1))``，分数等于阈值时 0.5，
单调、有界，可直接作为 ``IsStitchedProb`` 上报。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import Goal2StitchedConfig
from .model import probability_from_score


@dataclass(frozen=True)
class SeriesScore:
    """单条序列的拼接分数（``score=None`` 表示该序列打分失败，可降级）。"""

    series_uid: str
    score: float | None
    worst_slice: int | None
    slices: int
    series_type: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        detail: dict[str, object] = {
            "series_uid": self.series_uid,
            "series_type": self.series_type,
            "score": self.score,
            "worst_slice": self.worst_slice,
            "slices": self.slices,
        }
        if self.error is not None:
            detail["error"] = self.error
        return detail


@dataclass(frozen=True)
class StudyScore:
    """检查级拼接分数（= 各序列最大值）。"""

    score: float
    series: tuple[SeriesScore, ...]

    @property
    def scored_series(self) -> int:
        return sum(1 for item in self.series if item.score is not None)

    def as_dict(self) -> dict[str, object]:
        return {
            "score": round(float(self.score), 6),
            "series": [item.as_dict() for item in self.series],
            "scored_series": self.scored_series,
        }


def aggregate_series(scores: Sequence[SeriesScore]) -> StudyScore:
    """检查级分数 = 各序列分数的最大值；全部失败时为 0.0。"""
    values = [float(item.score) for item in scores if item.score is not None]
    return StudyScore(max(values) if values else 0.0, tuple(scores))


def probability(study_score: StudyScore, config: Goal2StitchedConfig) -> float:
    """检查级分数 → ``IsStitchedProb``。"""
    return probability_from_score(study_score.score, config.threshold, config.scale)


def flagged(study_score: StudyScore, config: Goal2StitchedConfig) -> bool:
    """是否判定为拼接影像（分数达到标定阈值）。"""
    return bool(study_score.score >= config.threshold)


def diagnostics(
    study_score: StudyScore,
    probability_value: float,
    config: Goal2StitchedConfig,
    series_types: dict[str, str] | None = None,
    duration_ms: int | None = None,
) -> dict[str, object]:
    """写入 ``context.diagnostics["goal2"]`` 的拼接部分（规范 §20.2）。"""
    detail: dict[str, object] = {
        **study_score.as_dict(),
        "probability": round(float(probability_value), 6),
        "threshold": config.threshold,
        "metric": config.metric,
        "statistic": config.statistic,
        "band": config.band,
        "flagged": flagged(study_score, config),
    }
    if series_types:
        detail["series_types"] = dict(series_types)
    if duration_ms is not None:
        detail["duration_ms"] = int(duration_ms)
    return detail


def series_types_of(series: Iterable[object]) -> dict[str, str]:
    """``{series_uid: 序列类型}``：类型来自 Loader 解析的 ``Series.modality``。"""
    return {
        str(getattr(item, "series_uid", "")): str(getattr(item, "modality", "") or "")
        for item in series
    }
