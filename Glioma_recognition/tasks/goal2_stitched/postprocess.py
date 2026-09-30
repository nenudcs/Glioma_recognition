"""Goal2-A 后处理：拼接概率的合法化与可信度判定（规范 §9.2）。

与 Goal1 同构：裁剪到 ``[0,1]`` 并对非有限值兜底，保证写出的 JSON 中
不出现 ``NaN``/``Infinity``（规范 §12.2 硬性要求）。
"""
from __future__ import annotations

import math

#: 无法打分时的安全默认值：不判定为"拼接"（假阳性会直接损失该例得分）
SAFE_DEFAULT = 0.0


def to_check_level(probability: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """把任意数值收敛为检查级合法概率。"""
    if probability is None or not math.isfinite(float(probability)):
        return SAFE_DEFAULT
    return float(min(hi, max(lo, float(probability))))


def is_reliable(probability: float, margin: float = 0.05) -> bool:
    """判断拼接判断是否足够可信。"""
    p = to_check_level(probability)
    return abs(p - 0.5) >= margin
