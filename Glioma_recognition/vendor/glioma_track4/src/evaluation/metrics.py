"""赛道4 指标：分割（Dice / NSD / HD95）与重复影像（AUC-PR / Recall@10%FPR / Precision@15%Recall）。"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# 分割
# --------------------------------------------------------------------------- #
def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 0
    g = gt > 0
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0
    return float(2.0 * (p & g).sum() / s)


def _surface(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage as ndi
    m = mask > 0
    if not m.any():
        return np.zeros(0, dtype=bool)
    er = ndi.binary_erosion(m, iterations=1, border_value=0)
    return m & ~er


def nsd(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0), tol_mm: float = 1.0) -> float:
    """Normalized Surface Dice（容差 tol_mm）。"""
    from scipy import ndimage as ndi
    ps, gs = _surface(pred), _surface(gt)
    if ps.sum() == 0 and gs.sum() == 0:
        return 1.0
    if ps.sum() == 0 or gs.sum() == 0:
        return 0.0
    dt_g = ndi.distance_transform_edt(~gs, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~ps, sampling=spacing)
    good_p = (dt_g[ps] <= tol_mm).sum()
    good_g = (dt_p[gs] <= tol_mm).sum()
    return float((good_p + good_g) / (ps.sum() + gs.sum()))


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    from scipy import ndimage as ndi
    ps, gs = _surface(pred), _surface(gt)
    if ps.sum() == 0 or gs.sum() == 0:
        return float("nan")
    dt_g = ndi.distance_transform_edt(~gs, sampling=spacing)
    dt_p = ndi.distance_transform_edt(~ps, sampling=spacing)
    d = np.concatenate([dt_g[ps], dt_p[gs]])
    return float(np.percentile(d, 95))


# --------------------------------------------------------------------------- #
# 重复影像
# --------------------------------------------------------------------------- #
def auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average Precision（等价于 PR 曲线下面积，用阶梯插值）。"""
    order = np.argsort(-scores)
    s, y = scores[order], labels[order].astype(float)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = tp / max(1.0, y.sum())
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float = 0.10) -> float:
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    best = 0.0
    tp = fp = 0
    for yi in y:
        tp += int(yi == 1)
        fp += int(yi == 0)
        if fp / n_neg <= fpr:
            best = max(best, tp / n_pos)
    return float(best)


def precision_at_recall(scores: np.ndarray, labels: np.ndarray, target_recall: float = 0.15) -> float:
    order = np.argsort(-scores)
    s, y = scores[order], labels[order]
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    tp = fp = 0
    for yi in y:
        tp += int(yi == 1)
        fp += int(yi == 0)
        if tp / n_pos >= target_recall:
            return float(tp / max(1, tp + fp))
    return float(tp / max(1, tp + fp))


def duplicate_report(pairs: list[dict], gold: set[tuple[str, str]], all_studies: list[str]) -> dict:
    """由预测对（含 prob）与金标准对计算三项指标。"""
    scores, labels = [], []
    seen: set[tuple[str, str]] = set()
    for r in pairs:
        k = tuple(sorted((str(r["a"]), str(r["b"]))))
        if k in seen:
            continue
        seen.add(k)
        scores.append(float(r["prob"]))
        labels.append(1.0 if k in gold else 0.0)
    for g in gold:                                        # 未出现的对按规范视为 0 概率
        k = tuple(sorted(g))
        if k not in seen:
            scores.append(0.0)
            labels.append(1.0)
    if not scores:
        return {"auc_pr": float("nan"), "recall@10fpr": float("nan"), "precision@15recall": float("nan"),
                "n_pred": 0, "n_gold": len(gold)}
    s, y = np.asarray(scores), np.asarray(labels)
    return {"auc_pr": auc_pr(s, y), "recall@10fpr": recall_at_fpr(s, y, 0.10),
            "precision@15recall": precision_at_recall(s, y, 0.15),
            "n_pred": len(pairs), "n_gold": len(gold)}
