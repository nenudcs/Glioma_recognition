"""重复影像匹配（目标二中的"重复影像"，指标 Recall@10%FPR / Precision@15%Recall / AUC-PR）。

三项指标都是**排序敏感**的（阈值无关或弱相关），因此核心是把真重复对排到前面：

1. **学习式嵌入**：多任务网络的全局头给出 L2 归一化嵌入，余弦相似度；
   由 ``DuplicatePairDataset`` 用金标准正对 + 随机负对训练（早期版本该损失恒被跳过）；
2. **手工指纹**：模态集合、各序列 shape/zoom、仿射矩阵、强度直方图；
   对"同一检查被重复上传/同一患者前后复查"这类几何与强度几乎一致的强信号非常有效；
3. 两者按权重融合 → 用验证集金标准**标定**为概率 → 每例 Top-K（规范 ≤200）。

规范要点：``(A,B)`` 与 ``(B,A)`` 视为同一对；重复出现取 ``PairProb`` 最大；
未提交的对默认 0；文件至少一行有效记录。
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 手工指纹相似度
# --------------------------------------------------------------------------- #
def fingerprint_similarity(fa: dict | None, fb: dict | None) -> float | None:
    """返回 [0,1] 的指纹相似度；信息不足时返回 None（不参与融合）。

    **权重设计要点**：同一中心/同协议扫描的检查，其 shape/zoom 必然相同，
    因此这两者只能作为**弱**证据；真正的强证据是"仿射矩阵（含原点）逐位一致"
    与"强度直方图高度吻合"——这正是"同一检查被重复上传/前后复查"的特征。
    早期版本给了 shape/zoom 0.7 的权重，导致同协议的不同病例被误判为重复。
    """
    if not fa or not fb:
        return None
    parts, weights = [], []

    # 1) 几何：仿射（强）+ shape/zoom（弱）
    mods = [k[6:] for k in fa if k.startswith("shape_")]
    if mods:
        aff_same, geom_same, tot = 0.0, 0.0, 0
        for m in mods:
            if f"shape_{m}" not in fb:
                continue
            tot += 1
            a, b = fa.get(f"aff_{m}"), fb.get(f"aff_{m}")
            if a and b and a == b:
                aff_same += 1.0                                     # 仿射逐位一致 → 强证据
            if fa[f"shape_{m}"] == fb[f"shape_{m}"] and \
                    fa.get(f"zoom_{m}") == fb.get(f"zoom_{m}"):
                geom_same += 0.25                                   # 同协议 → 弱证据
        if tot:
            parts.append(float(np.clip((0.75 * aff_same + geom_same) / tot, 0.0, 1.0)))
            weights.append(0.45)

    # 2) 强度直方图相似度（Bhattacharyya 系数）—— **按模态名对齐**
    ha, hb = fa.get("hist") or {}, fb.get("hist") or {}
    if isinstance(ha, list):                                      # 兼容旧格式
        ha = {str(i): v for i, v in enumerate(ha)}
    if isinstance(hb, list):
        hb = {str(i): v for i, v in enumerate(hb)}
    sims = []
    for m, x in ha.items():
        y = hb.get(m)
        if y is None:
            continue
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = min(len(x), len(y))
        if n:
            sims.append(float(np.sqrt(np.clip(x[:n], 0, None) * np.clip(y[:n], 0, None)).sum()))
    if sims:
        parts.append(float(np.clip(np.mean(sims), 0, 1)))
        weights.append(0.20)

    # 3) 空间缩略图的**相对差异**（核心证据）：重复检查的体素几乎逐点一致
    #    （NCC 对轻微噪声不敏感，实测正/负样本间隔仅 0.01；相对差异则间隔显著）
    ta, tb = fa.get("thumb") or {}, fb.get("thumb") or {}
    if isinstance(ta, list):
        ta = {}
    if isinstance(tb, list):
        tb = {}
    near = []
    for m, x in ta.items():
        y = tb.get(m)
        if y is None:
            continue
        x, y = np.asarray(x, np.float32).ravel(), np.asarray(y, np.float32).ravel()
        x, y = np.nan_to_num(x), np.nan_to_num(y)
        if x.size != y.size or x.size == 0:
            continue
        scale = float(np.abs(x).mean()) + 1e-6
        rel = float(np.abs(x - y).mean() / scale)                 # 0 = 完全一致
        if not np.isfinite(rel):
            continue
        # rel=0 → 1.0；rel=0.03 → 0.37；rel≥0.1 → ≈0
        near.append(float(np.exp(-(rel / 0.03) ** 2)))
    if near:
        parts.append(float(np.clip(np.mean(near), 0, 1)))
        weights.append(0.35)

    # 4) 模态集合 Jaccard（弱）
    ma, mb = set(fa.get("mods") or []), set(fb.get("mods") or [])
    if ma and mb:
        parts.append(len(ma & mb) / max(1, len(ma | mb)))
        weights.append(0.10)

    if not parts:
        return None
    return float(np.clip(np.average(parts, weights=weights), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# 标定
# --------------------------------------------------------------------------- #
def calibrate(sim_pos: np.ndarray, sim_neg: np.ndarray, target_fpr: float = 0.10) -> dict:
    """用验证折的金标准对拟合 logistic 标定参数 ``(a, b)``：``p = sigmoid(a*(sim-b))``。"""
    a, b = 12.0, 0.85
    if len(sim_neg) >= 20:
        b = float(np.quantile(sim_neg, 1.0 - target_fpr))
    if len(sim_pos) >= 10 and len(sim_neg) >= 20:
        mid_pos = float(np.median(sim_pos))
        denom = max(1e-3, mid_pos - b)
        a = float(np.clip(6.0 / denom, 4.0, 60.0))
    return {"a": a, "b": b, "target_fpr": target_fpr,
            "n_pos": int(len(sim_pos)), "n_neg": int(len(sim_neg))}


def pair_prob(sim: float, calib: dict | None) -> float:
    calib = calib or {"a": 12.0, "b": 0.85}
    a, b = calib.get("a", 12.0), calib.get("b", 0.85)
    return float(1.0 / (1.0 + np.exp(-a * (sim - b))))


# --------------------------------------------------------------------------- #
# 配对
# --------------------------------------------------------------------------- #
def match_pairs(embeds: dict[str, np.ndarray], calib: dict | None = None, topk: int = 200,
                min_prob: float = 0.005, feats: dict | None = None,
                w_fp: float = 0.5, per_study_k: int | None = None) -> list[dict]:
    """``embeds: {accession: 嵌入}`` → ``[{"a","b","prob","sim"}]``（每例 ≤ topk）。

    ``per_study_k``：每例**实际提交**的候选数（≤ ``topk``）。三项指标中 AUC-PR 对
    低分长尾敏感（未提交对按 0 计），因此提交"少而准"通常优于"多而杂"；
    ``topk`` 仍按规范上限 200。
    """
    accs = sorted(embeds)
    if not accs:
        return []
    if len(accs) == 1:                                            # 规范：至少一行有效记录
        a = accs[0]
        return [{"a": a, "b": a, "prob": 0.5, "sim": 1.0}]

    M = np.stack([np.asarray(embeds[a], np.float32).ravel() for a in accs])
    nrm = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / np.maximum(nrm, 1e-8)
    S = M @ M.T                                                   # 余弦相似度
    S01 = (S + 1.0) / 2.0                                         # 归一到 [0,1] 便于与指纹融合

    if feats and w_fp > 0:
        # 只在**嵌入相似度的 Top-K 候选**上计算指纹（复杂度 O(N·K) 而非 O(N²)），
        # 既保证不漏掉真正的重复对（它们必然排在嵌入相似度前列），
        # 又避免大测试集上指纹计算成为瓶颈。
        cand_k = min(len(accs) - 1, max(int(topk), 100))
        for i, a in enumerate(accs):
            order_i = np.argsort(-S01[i])[: cand_k + 1]
            fa = feats.get(a)
            for j in order_i:
                if j <= i:
                    continue
                fs = fingerprint_similarity(fa, feats.get(accs[j]))
                if fs is not None:
                    v = (1.0 - w_fp) * S01[i, j] + w_fp * fs
                    S01[i, j] = S01[j, i] = v

    out: list[dict] = []
    K = int(per_study_k) if per_study_k else int(topk)
    K = max(1, min(K, int(topk)))                                 # 规范上限 200
    for i, a in enumerate(accs):
        sims = S01[i].copy()
        sims[i] = -1.0
        order = np.argsort(-sims)[:K]
        for j in order:
            if sims[j] <= 0:
                continue
            p = pair_prob(float(sims[j]), calib)
            if p >= min_prob:
                out.append({"a": a, "b": accs[j], "prob": p, "sim": float(sims[j])})
    best: dict[tuple[str, str], dict] = {}
    for r in out:                                                 # 同一对去重取最大
        k = tuple(sorted((r["a"], r["b"])))
        if k not in best or r["prob"] > best[k]["prob"]:
            best[k] = r
    res = sorted(best.values(), key=lambda r: -r["prob"])
    if not res:                                                   # 全低于阈值 → 保底一行
        i, j = np.unravel_index(np.argmax(S01 - np.eye(len(accs)) * 9), S01.shape)
        res = [{"a": accs[i], "b": accs[j], "prob": max(0.5, pair_prob(float(S01[i, j]), calib)),
                "sim": float(S01[i, j])}]
    return res


def gold_similarities(embeds: dict[str, np.ndarray], gold_pairs: list[list[str]],
                      n_neg: int = 20000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """用金标准对与随机负对构造相似度分布（用于标定）。"""
    accs = sorted(embeds)
    pos, neg = [], []
    for a, b in gold_pairs:
        if a in embeds and b in embeds:
            pos.append(float(np.dot(embeds[a], embeds[b])))
    rng = np.random.default_rng(seed)
    for _ in range(min(n_neg, max(1, len(accs)) * 4)):
        i, j = rng.integers(0, len(accs), 2)
        if i == j:
            continue
        neg.append(float(np.dot(embeds[accs[i]], embeds[accs[j]])))
    return np.asarray(pos), np.asarray(neg)
