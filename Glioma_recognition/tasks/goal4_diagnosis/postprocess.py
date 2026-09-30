"""Goal4 后处理：把分类头输出组装成 ``Goal4Result``（规范 §9.5）。

三个必须注意的点：

1. **内部枚举 → 比赛字符串**走 ``output.competition_mapping``，不在这里写字符串；
2. **概率必须归一**：规范要求 ``sum(probabilities) == 1``（容差 1e-4），
   逐项 round 后的累积舍入误差会超标，因此统一用残差补偿把和补回 1.0；
3. **``predicted`` 必须与 ``probabilities`` 自洽**：取最大概率项，
   并在极少数并列情况下按枚举顺序确定，保证结果可复现。

另：阴性病例（``TumorProbability < 0.5``）时 ``WHO_Grade.predicted`` 输出 null、
概率全 0——这是规范示例中的写法，也是语义正确的（不存在胶质瘤分级）。
"""
from __future__ import annotations

import math
from typing import Any


def normalize_probs(probs: list[float]) -> list[float]:
    """归一化到 sum==1（含残差补偿，满足 sum 容差 1e-4 的硬约束）。"""
    finite = [float(p) if math.isfinite(float(p)) else 0.0 for p in probs]
    s = sum(finite)
    if s <= 0:
        n = max(1, len(finite))
        vals = [round(1.0 / n, 4)] * n
    else:
        vals = [round(max(0.0, p) / s, 4) for p in finite]
    if vals:
        residual = round(1.0 - sum(vals), 4)
        if residual:
            j = max(range(len(vals)), key=lambda i: vals[i])
            vals[j] = round(max(0.0, min(1.0, vals[j] + residual)), 4)
    return vals


def categorical(names: list[str], probs: Any, output_names: list[str] | None = None
                ) -> dict[str, Any]:
    """类别字段 ``{"predicted":..., "probabilities":{...}}``。

    ``output_names``：对外暴露的枚举子集。用于"模型训练时多一个兜底类、
    但规范示例只列了部分取值"的场景（如 Morphology 的内部含 NA，而示例仅两类）——
    **模型头不变，只在输出层裁剪并重新归一**。
    """
    keep = [n for n in (output_names or names) if n in names] or list(names)
    if probs is not None and len(probs) == len(names):
        vals = normalize_probs([float(x) for x in probs])
        pick = names[max(range(len(vals)), key=lambda i: vals[i])]
        if keep != names:
            idx = [names.index(n) for n in keep]
            sub = normalize_probs([vals[i] for i in idx])
            j = keep.index(pick) if pick in keep else max(range(len(sub)), key=lambda i: sub[i])
            top = max(sub)
            if sub[j] < top - 1e-9:
                sub = [min(v, top) for v in sub]
                sub[j] = top
                sub = normalize_probs(sub)
            return {"predicted": keep[j], "probabilities": dict(zip(keep, sub))}
        j = names.index(pick)
        top = max(vals)
        if vals[j] < top - 1e-9:                                  # 自洽：predicted 取到最大值
            vals = [min(v, top) for v in vals]
            vals[j] = top
            vals = normalize_probs(vals)
        return {"predicted": pick, "probabilities": dict(zip(names, vals))}
    return {"predicted": keep[0], "probabilities": {n: 0.0 for n in keep}}


def binary(present: bool, probability: float) -> dict[str, Any]:
    """二分类字段：``{"present": bool, "probability": float}``。"""
    p = float(probability)
    p = p if math.isfinite(p) else 0.5
    return {"present": bool(present), "probability": round(min(1.0, max(0.0, p)), 4)}
