"""Goal5 的 T1 增强核心区（core）模型。

本工程采用**共享多任务骨干 + 多通道输出**（规范 §5.1 明确允许"通过 factory.py
切换为单个多头模型"）：一次前向同时产出 core / peri(flair) 两个通道，
比训练两个独立网络更省显存、也更快。

因此本文件不重复定义骨干，而是复用 ``tasks._common`` 中的**单点实现**，
只声明"哪一个输出通道属于 core"。

若要换成**独立模型**（例如 core 单独训练），只需让 ``build_core_model()``
返回另一个网络、并把 ``Goal5Config.core_ckpt_rel`` 指向它的权重，
上层 ``inference.py`` / ``task.py`` 无需任何改动。
"""
from __future__ import annotations

from tasks._common.mednext import MedNeXtNet
from tasks._common.unet3d import GliomaNet
from tasks.goal5_segmentation.config import Goal5Config


def build_core_model(cfg: Goal5Config, cls_spec: list[tuple[str, int]] | None = None):
    """构建产出 core 通道的骨干（共享权重的多任务网络）。

    Returns:
        具备 ``forward -> {"seg": [B,2,D,H,W], "cls": [...], "special": ..., "embed": ...}``
        契约的网络；core 对应 ``seg[:, cfg.core_channel]``。
    """
    spec = cls_spec or []
    if cfg.arch == "mednext":
        return MedNeXtNet(
            in_ch=cfg.in_channels, base=cfg.base, depth=cfg.depth, cls_spec=spec,
            blocks_per_stage=cfg.blocks_per_stage, k=cfg.k, expand=cfg.expand,
            aniso_z=cfg.aniso_z, max_ch=cfg.max_ch,
        )
    return GliomaNet(in_ch=cfg.in_channels, base=cfg.base, cls_spec=spec)


def core_channel(cfg: Goal5Config) -> int:
    """core 掩码在 ``seg`` 输出中的通道索引。"""
    return int(cfg.core_channel)
