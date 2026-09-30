"""Goal5 模型工厂：统一 core / flair 的构建入口。

规范 §5.1：Goal5 对外只有一个 StudyTask，内部默认分别构建 core 与 flair/T2 模型，
并**允许通过 factory.py 切换为单个多头模型**。本工程默认采用后者（共享骨干），
并由本工厂集中决策，以便将来不改上层代码即可切换形态。
"""
from __future__ import annotations

from dataclasses import dataclass

from tasks.goal5_segmentation.config import Goal5Config
from tasks.goal5_segmentation.models.core import build_core_model, core_channel
from tasks.goal5_segmentation.models.flair import build_flair_model, flair_channel


@dataclass(frozen=True)
class Goal5Models:
    """Goal5 的两路分割模型及其通道索引。"""

    core_model: object
    flair_model: object
    core_channel: int
    flair_channel: int
    shared: bool               # True = 两路共享同一骨干（同一份权重、一次前向）


def build_goal5_models(cfg: Goal5Config, cls_spec: list[tuple[str, int]] | None = None,
                       shared: bool = True) -> Goal5Models:
    """构建 Goal5 的两路模型。

    Args:
        shared: True  → 两路共用同一骨干（省显存、只跑一次前向）；
                False → 分别构建（各自权重，便于独立演进）。
    """
    core = build_core_model(cfg, cls_spec)
    flair = core if shared else build_flair_model(cfg, cls_spec)
    return Goal5Models(
        core_model=core,
        flair_model=flair,
        core_channel=core_channel(cfg),
        flair_channel=flair_channel(cfg),
        shared=shared,
    )
