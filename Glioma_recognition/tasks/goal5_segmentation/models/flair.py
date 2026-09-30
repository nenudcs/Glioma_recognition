"""Goal5 的 FLAIR/T2 周围总异常区（flair）模型。

与 ``core.py`` 共享同一个多任务骨干，只声明 flair 对应的输出通道。
独立训练的替换方式与 ``core.py`` 的说明一致。
"""
from __future__ import annotations

from tasks.goal5_segmentation.config import Goal5Config
from tasks.goal5_segmentation.models.core import build_core_model


def build_flair_model(cfg: Goal5Config, cls_spec: list[tuple[str, int]] | None = None):
    """默认与 core 共用骨干（同一份权重、不同输出通道）。

    若将来把 flair 训练成独立模型，只需改为返回另一个网络，并把
    ``Goal5Config.flair_ckpt_rel`` 指向其权重。
    """
    return build_core_model(cfg, cls_spec)


def flair_channel(cfg: Goal5Config) -> int:
    """flair 掩码在 ``seg`` 输出中的通道索引。"""
    return int(cfg.flair_channel)
