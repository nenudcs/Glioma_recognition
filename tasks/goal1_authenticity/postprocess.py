"""[运行] 序列级 → 检查级聚合（规范 §9.1）。

训练与推理必须用同一套聚合口径：一条序列取 ``K // 2`` 个最高 **logit** 求均值后
sigmoid（在 ``inference.py`` 内完成），检查级再对多条序列做聚合（本文件）：

* ``max``：只要一条序列可疑就标记该检查（默认）；
* ``mean``：所有可打分序列的平均。

聚合结果为空表示「整个 Study 没有可打分的影像」——按 §9.1 属于不可降级错误，
由 ``task.py`` 抛出公共异常。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

SERIES_AGGREGATIONS = ("max", "mean")


@dataclass(frozen=True)
class SeriesScore:
    series_uid: str
    probability: float | None
    series_type: str = ""
    error: str | None = None


def aggregate(probabilities: Sequence[float], mode: str = "max") -> float | None:
    """序列级概率 → 检查级概率；没有可用概率时返回 ``None``。"""
    values = [float(value) for value in probabilities if value is not None]
    if not values:
        return None
    if mode == "mean":
        return float(np.mean(values))
    return float(max(values))


def aggregate_scores(scores: Iterable[SeriesScore], mode: str = "max") -> float | None:
    return aggregate([item.probability for item in scores if item.probability is not None], mode)


def diagnostics(scores: Sequence[SeriesScore], probability: float | None, mode: str) -> dict[str, object]:
    """写入 ``context.diagnostics`` 的结构化信息（日志与排障用，不进 prediction.json）。"""
    return {
        "probability": probability,
        "aggregation": mode,
        "series_count": len(scores),
        "scored_series": sum(1 for item in scores if item.probability is not None),
        "series": [
            {
                "series_uid": item.series_uid,
                "series_type": item.series_type,
                "probability": item.probability,
                "error": item.error,
            }
            for item in scores
        ],
    }
