"""[运行] 拼接影像规则模型（无神经网络参数，纯函数：体数据 → 分数）。

规则（与赛方对「拼接影像」的定义一致：跨患者/跨部位拼接会在某一层留下突变）：

```text
对每一层 i：d_i = mean(|I_i - (I_{i-1} + I_{i+1}) / 2|) / mean|I|
```

一层拼接就足以判定，所以检查级分数默认取所有层的**最大值**；层方向两端默认丢弃
``band`` 比例（颈部、空气进出视野在 FOV 边缘本来就有大的层间变化）。

三种口径（``GOAL2_STITCHED_METRIC``）：

* ``curvature``（默认）：当前层 vs 上下层均值 —— 直接对应上面的公式；
* ``adjacent``：相邻层差 ``mean|I_i - I_{i-1}|``；
* ``local``：层间差除以**该卷自身**的局部层间差中位数（相对自己是否异常）。

本文件只做数学，不读文件、不写结果、不抛业务异常。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DEFAULT_BAND, METRICS, STATISTICS
from .preprocess import brightness_scale, move_slices_first, to_float_volume

LOCAL_WINDOW = 4


def slice_residual_scores(
    volume: np.ndarray | str | Path,
    metric: str = "curvature",
    band: float = DEFAULT_BAND,
) -> np.ndarray:
    """逐层差异分数（无量纲，长度 = 层数 - 2，已按 ``band`` 裁掉两端）。"""
    if metric not in METRICS:
        raise ValueError(f"metric 必须是 {METRICS} 之一")
    array = to_float_volume(volume)
    if array.ndim < 3 or min(array.shape) < 1:
        return np.zeros(0, dtype=np.float32)
    moved = np.asarray(move_slices_first(array), dtype=np.float32)
    count = moved.shape[0]
    if count < 3:
        return np.zeros(0, dtype=np.float32)

    axes = tuple(range(1, moved.ndim))
    previous, current, following = moved[:-2], moved[1:-1], moved[2:]
    curvature = np.abs(current - 0.5 * (previous + following)).mean(axis=axes)
    adjacent = np.abs(np.diff(moved, axis=0)).mean(axis=axes)      # 长度 count-1
    scale = brightness_scale(moved)

    if metric == "adjacent":
        raw = adjacent
    elif metric == "local":
        floor = 0.02 * scale
        raw = np.zeros_like(curvature)
        for index in range(curvature.size):
            low = max(0, index - LOCAL_WINDOW)
            high = min(curvature.size, index + LOCAL_WINDOW + 1)
            baseline = max(float(np.median(curvature[low:high])), floor)
            raw[index] = curvature[index] / baseline
        return np.asarray(raw, dtype=np.float32)
    else:
        raw = curvature

    raw = np.asarray(raw / scale, dtype=np.float32)
    if 0.0 < band < 0.5 and raw.size > 2:
        margin = int(round(raw.size * band))
        if margin > 0 and raw.size - 2 * margin >= 1:
            raw = raw[margin:raw.size - margin]
    return raw


def reduce_scores(scores: np.ndarray, statistic: str = "max") -> float:
    """层分数 → 单序列分数（``statistic`` = ``max`` / ``p99`` / ``mean``）。"""
    if statistic not in STATISTICS:
        raise ValueError(f"statistic 必须是 {STATISTICS} 之一")
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return 0.0
    if statistic == "p99":
        return float(np.percentile(values, 99))
    if statistic == "mean":
        return float(values.mean())
    return float(values.max())


def volume_score(
    volume: np.ndarray | str | Path,
    statistic: str = "max",
    metric: str = "curvature",
    band: float = DEFAULT_BAND,
) -> float:
    """单条序列（一个体数据）的拼接分数。"""
    scores = slice_residual_scores(volume, metric, band)
    return reduce_scores(scores, statistic)


def worst_slice(scores: np.ndarray) -> int | None:
    """分数最高的层号（1 基，标注 ``band`` 裁剪后的相对层号）。"""
    values = np.asarray(scores)
    if values.size == 0:
        return None
    return int(np.argmax(values)) + 1


def probability_from_score(score: float, threshold: float, scale: float) -> float:
    """分数 → ``[0,1]`` 概率：分数 = 阈值时 0.5，越大越接近 1。"""
    threshold = max(float(threshold), 1e-6)
    ratio = float(score) / threshold - 1.0
    ratio = max(-60.0, min(60.0, float(scale) * ratio))
    return float(1.0 / (1.0 + math.exp(-ratio)))


def score_series_list(
    series: Sequence[tuple[str, np.ndarray | str | Path]],
    statistic: str = "max",
    metric: str = "curvature",
    band: float = DEFAULT_BAND,
) -> list[tuple[str, float | None, int | None, int, str | None]]:
    """批量打分：``[(series_uid, score|None, worst_slice, slices, error|None), ...]``。"""
    results: list[tuple[str, float | None, int | None, int, str | None]] = []
    for series_uid, source in series:
        try:
            scores = slice_residual_scores(source, metric, band)
        except Exception as exc:  # noqa: BLE001 - 单条序列可降级
            results.append((str(series_uid), None, None, 0, f"{type(exc).__name__}: {exc}"))
            continue
        results.append(
            (
                str(series_uid),
                reduce_scores(scores, statistic),
                worst_slice(scores),
                int(np.asarray(scores).size),
                None,
            )
        )
    return results
