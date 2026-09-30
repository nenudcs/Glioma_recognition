"""Goal2-A 影像规范性审核 · 拼接影像检测 的插件配置（规范 §5.1）。

规范 §5.2：本文件只声明**相对路径**，绝对根目录由 ``core.config.Settings``
的 ``ckpt_root`` 提供。

注意本 Goal 与 Goal2-B（``goal2_duplicate``）是**同一目标下的两个子任务**，
但权重文件不同（``model.pt`` vs ``encoder.pt``）：前者只需 ``special`` 头的
通道 1，后者只需 ``embed`` 头。共享骨干的前向缓存按 ``ckpt_rel`` 隔离，
因此两者在同一 Study 上各取所需、互不串味。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal2StitchedConfig:
    """Goal2-A 推理期参数（默认值与 ``SingleHeadStudyTask`` 的基线一致）。"""

    #: 权重相对路径（规范 §5.2）
    ckpt_rel: str = "goal2_stitched/model.pt"
    #: 共享骨干输入通道数（必须与训练一致）
    in_channels: int = 4
    #: 骨干结构：mednext / resunet
    arch: str = "mednext"
    #: 整脑视图：物理边长（mm）与输出尺寸——**训练与推理必须一致**
    global_size: int = 96
    global_size_mm: float = 192.0
    #: 推理期 TTA 翻折轴
    tta_flips: tuple[str, ...] = ("x", "y")
    tta_batch: int = 2
    #: 公共网格 spacing（1mm）
    common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
