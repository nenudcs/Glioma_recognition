"""Goal3 病灶识别 的数据集入口（[研发] 文件）。

数据集与强增强的实现在算法工程侧（``glioma_track4``）**单点维护**，中央
仓库不复制一份——规范 §17.1 要求"公共结构只允许单点维护"，而数据增强
恰是最容易被各 Goal 各自改动、进而导致实验不可比的部分。

本文件导出该 Goal 的**稳定数据入口**；实现委托给
``tasks._common.training.helpers.build_datasets``（统一训练走同一条路径，
因此不会出现"Goal 侧与统一训练两套增强"的隐性分叉）。
"""
from __future__ import annotations

from typing import Any

GOAL = "goal3_tumor"


def build_datasets(raw: dict[str, Any], fold: int = 0, limit: int = 0,
                   seed: int = 42, patch: tuple = (96, 96, 96)):
    """返回 ``(train_ds, val_ds)``。

    Args:
        raw: 本 Goal 的 ``config.yaml`` 内容。
        fold: 折号（对应 ``folds.json`` 的 key）。
        limit: 只用前 N 例（冒烟用；0 = 全量）。
        seed: 增强与配对采样的随机种子。
        patch: 训练 patch 尺寸。

    Returns:
        ``(train_ds, val_ds)``，均带 ``special``（fake/stitched）监督；
        训练集另带配对样本（embed 头的监督，验证集刻意不产，保证指标可复现）。

    Raises:
        FileNotFoundError: 缺折划分文件（需先跑探针与折划分脚本）。
    """
    from tasks._common.training.helpers import build_datasets as _impl

    return _impl(raw, fold=fold, limit=limit, seed=seed, patch=patch)
