"""Goal3 后处理：肿瘤概率的合法化（规范 §9.4）。

Goal3 用 ROC-AUC 评估，输出的是**排序分数**，因此这里**不做阈值化**——
任何改变单调性的后处理都会直接降低 AUC。只做非有限值兜底与区间裁剪。
"""
from __future__ import annotations

import math

#: 无法打分时的默认值。取 0.5 而非 0/1：AUC 只看排序，
#: 中间值对并列样本的惩罚最小，不会人为制造极端误判。
SAFE_DEFAULT = 0.5


def to_probability(logit_probability: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """收敛为 ``[0,1]`` 内的有限概率（保持单调，不改变排序）。"""
    if logit_probability is None or not math.isfinite(float(logit_probability)):
        return SAFE_DEFAULT
    return float(min(hi, max(lo, float(logit_probability))))
