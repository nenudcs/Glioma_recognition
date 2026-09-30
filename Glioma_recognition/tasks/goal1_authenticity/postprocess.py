"""Goal1 后处理：序列级 → 检查级概率聚合（规范 §9.1）。

规范明确要求在 ``postprocess.py`` 完成"序列级到检查级"的聚合。
当前骨干在**整脑全局视图**上一次性给出检查级 logits，因此这里的聚合是
"裁剪到合法区间 + 非有限值兜底"，而不是跨序列平均。

**为什么必须做非有限值兜底**：规范要求所有概率是 ``[0,1]`` 内的有限浮点数，
JSON 中不得出现 ``NaN``/``Infinity``。一旦上游给出 NaN 而这里直接透传，
Writer 写出的 JSON 就会非法，Validator 会判整例失败——代价远大于一个默认概率。
"""
from __future__ import annotations

import math

#: 无法得到有效打分时使用的安全默认值：不判定为"非人体"（假阳性代价更高）
SAFE_DEFAULT = 0.0


def to_check_level(logit_probability: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """把任意数值收敛为检查级合法概率。"""
    if logit_probability is None or not math.isfinite(float(logit_probability)):
        return SAFE_DEFAULT
    return float(min(hi, max(lo, float(logit_probability))))


def is_reliable(probability: float, margin: float = 0.05) -> bool:
    """判断该检查级的判断是否足够可信（留给聚合层决定是否采用启发式）。"""
    p = to_check_level(probability)
    return abs(p - 0.5) >= margin
