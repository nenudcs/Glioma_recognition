"""Goal5 分割与可解释性 的数据增强入口（[研发] 文件）。

增强策略（翻转 / 弹性形变 / 偏置场 / 噪声 / 强度扰动 / 随机裁剪）与数据集
同处算法工程侧单点维护，理由见 ``dataset.py`` 的说明。

**推理侧禁止使用随机增强**：``preprocess.py`` 必须保持确定性，否则同一份
输入两次推理会得到不同掩膜，既无法复现，也无法通过"掩码与源序列几何一致"
的校验。
"""
from __future__ import annotations

from typing import Any

GOAL = "goal5_segmentation"

#: 关闭增强的配置（键名保持完整，便于下游统一读取）
_DISABLED: dict[str, Any] = {"enabled": False}


def build_augment(raw: dict[str, Any]) -> dict[str, Any]:
    """返回本 Goal 的数据增强配置。

    ``config.yaml`` 的 ``augment`` 段原样透出；未配置时按"启用基础增强"
    处理。返回**副本**，调用方修改不会波及其他 Goal。
    """
    cfg = dict(raw.get("augment") or {})
    cfg.setdefault("enabled", True)
    return cfg


def disabled() -> dict[str, Any]:
    """返回"关闭增强"的配置（用于确定性评估与结果复现）。"""
    return dict(_DISABLED)
