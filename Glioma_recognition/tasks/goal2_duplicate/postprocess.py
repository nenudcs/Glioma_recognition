"""Goal2-B 后处理：pair 规范化（规范 §9.3 与 §18.2 的硬性要求）。

规范要求的五条规则，缺一条都会让 ``duplicate_pairs.jsonl`` 被判非法：

1. **无向**：``(A,B)`` 与 ``(B,A)`` 是同一对 → 按字典序归一化为 ``(min, max)``；
2. **去重取最大**：同一对多次出现只保留**最高**概率；
3. **禁 self-pair**：除非数据集中只有一个检查（此时规范要求至少一行，
   只能写自比对占位行，由 Writer 处理）；
4. **确定性排序**：按概率降序、accession 升序——保证同一份输入两次运行结果完全一致；
5. **每例 ≤ 200 对**：按排序结果贪心截断。
"""
from __future__ import annotations

from typing import Any


def normalize_pairs(pairs: list[dict], max_per_study: int = 200) -> list[dict]:
    """把原始配对结果规范化为可直接写 JSONL 的形式。"""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    n_acc = len({p["a"] for p in pairs} | {p["b"] for p in pairs})

    for p in pairs:
        a, b = str(p["a"]), str(p["b"])
        if a == b and n_acc > 1:
            continue                                              # 禁 self-pair
        key = (a, b) if a <= b else (b, a)                        # 无向归一化
        prob = float(p.get("prob", 0.0))
        prob = 0.0 if prob != prob else min(1.0, max(0.0, prob))  # NaN → 0，并裁剪
        cur = best.get(key)
        if cur is None or prob > cur["prob"]:
            best[key] = {"a": key[0], "b": key[1], "prob": prob,
                         "sim": float(p.get("sim", 0.0))}

    ordered = sorted(best.values(), key=lambda r: (-r["prob"], r["a"], r["b"]))

    if max_per_study <= 0:
        return ordered
    used: dict[str, int] = {}
    out: list[dict] = []
    for r in ordered:
        if used.get(r["a"], 0) >= max_per_study or used.get(r["b"], 0) >= max_per_study:
            continue
        out.append(r)
        used[r["a"]] = used.get(r["a"], 0) + 1
        used[r["b"]] = used.get(r["b"], 0) + 1
    return out
