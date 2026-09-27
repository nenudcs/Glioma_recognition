"""[运行] 重复影像的后处理：相似度 → 概率、成对结果规整（去重 / 自配对 / Top-200）。

概率映射 ``PairProb = sigmoid((similarity - center) / scale)``：

* 单调、有界，直接作为提交字段 ``PairProb``；
* ``similarity = center`` 时概率 0.5，与闸门阈值（``GOAL2_DUPLICATE_GATE_PROB``）同口径；
* 排序指标（AUC-PR）只依赖单调性，标定只影响「判定为重复」的绝对阈值。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from tasks.results import DuplicatePair


def probability_from_similarity(
    similarity: float,
    center: float = 0.90,
    scale: float = 0.02,
) -> float:
    """相似度 → ``[0,1]`` 概率（相似度 = ``center`` 时 0.5）。"""
    scale = max(float(scale), 1e-6)
    ratio = (float(similarity) - float(center)) / scale
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, ratio)))))


def similarity_from_probability(
    probability: float,
    center: float = 0.90,
    scale: float = 0.02,
) -> float:
    """概率阈值 → 对应相似度阈值（用于配置校验与文档）。"""
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return float(center) + float(scale) * math.log(probability / (1.0 - probability))


@dataclass(frozen=True)
class DuplicateMatch:
    """一条命中记录：与当前检查重复的**其它**检查及其概率。"""

    accession: str
    probability: float
    similarity: float
    series_uid: str = ""
    other_series_uid: str = ""
    exact: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "accession": self.accession,
            "probability": round(float(self.probability), 6),
            "similarity": round(float(self.similarity), 6),
            "series_uid": self.series_uid,
            "other_series_uid": self.other_series_uid,
            "exact": bool(self.exact),
        }


def normalize_pairs(
    pairs: Iterable[DuplicatePair],
    max_pairs_per_study: int = 200,
) -> list[DuplicatePair]:
    """去重（同一无序对保留最高概率）、丢弃自配对、按概率降序截断到每例 Top-N。

    与赛方稀疏性约束一致：每例检查最多提交与其相关的 200 个候选对，
    按模型内部置信度降序截取。管线 Writer 也会做同一件事，这里是插件侧的兜底与自检。
    """
    best: dict[tuple[str, str], DuplicatePair] = {}
    for pair in pairs:
        key = tuple(sorted((str(pair.left_accession), str(pair.right_accession))))
        if key[0] == key[1]:
            continue
        current = best.get(key)
        if current is None or pair.probability > current.probability:
            best[key] = DuplicatePair(key[0], key[1], float(pair.probability))

    counts: dict[str, int] = {}
    selected: list[DuplicatePair] = []
    for pair in sorted(best.values(), key=lambda item: (-item.probability, item.left_accession)):
        if counts.get(pair.left_accession, 0) >= max_pairs_per_study:
            continue
        if counts.get(pair.right_accession, 0) >= max_pairs_per_study:
            continue
        selected.append(pair)
        counts[pair.left_accession] = counts.get(pair.left_accession, 0) + 1
        counts[pair.right_accession] = counts.get(pair.right_accession, 0) + 1
    return selected


def pairs_from_matches(
    accession: str,
    matches: Sequence[Mapping[str, object] | DuplicateMatch],
) -> list[DuplicatePair]:
    """把某个检查的命中列表整理成 ``DuplicatePair``（兼容 dict 与 dataclass 两种形态）。"""
    pairs: list[DuplicatePair] = []
    for item in matches:
        if isinstance(item, DuplicateMatch):
            other, probability = item.accession, item.probability
        else:
            other = str(item.get("accession", ""))
            probability = float(item.get("probability", 0.0))
        if not other or other == accession:
            continue
        pairs.append(DuplicatePair(str(accession), other, float(probability)))
    return pairs


def similarity_distribution(values: Sequence[float]) -> dict[str, object]:
    """相似的分布摘要（自检/标定输出用）。"""
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 6),
        "p50": round(float(np.median(array)), 6),
        "p90": round(float(np.quantile(array, 0.90)), 6),
        "max": round(float(array.max()), 6),
    }
