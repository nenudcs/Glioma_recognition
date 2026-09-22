"""[研发] 损失函数（规范 §5.1；比赛运行入口不得导入本文件）。

正类（伪造/非人体）在赛方训练集中极度不平衡，默认按 ``neg / pos`` 自动设置
``pos_weight``；需要时用 ``--pos-weight`` 显式覆盖。
"""
from __future__ import annotations

import numpy as np


def positive_weight(labels: np.ndarray, explicit: float = -1.0) -> float:
    """``explicit > 0`` 时直接用；否则取 ``max(neg / pos, 1.0)``。"""
    if explicit > 0:
        return float(explicit)
    labels = np.asarray(labels, dtype=np.float64)
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    return float(max(negatives / max(positives, 1.0), 1.0))


def build_criterion(pos_weight: float, device):
    """BCEWithLogitsLoss + ``pos_weight``（返回 ``torch.nn.Module``）。"""
    import torch
    import torch.nn as nn

    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
