"""Goal3 病灶识别 的插件配置（规范 §5.1 把"配置"列为独立交付项）。

规范 §5.2：本文件只声明**相对路径**，绝对根目录由 ``core.config.Settings``
的 ``ckpt_root`` 提供。

本 Goal 读共享骨干的 ``cls`` 头中名为 ``TumorProbability`` 的一路。
**已知限制**：本地与公开训练数据里没有非肿瘤性病变（脑梗死/脑脓肿等）
负样本，该头在当前数据上无法有效训练、AUC 也无法在本地评估；
链路口径已就绪，待补充负样本后即可启用（详见本 Goal 的 README）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal3Config:
    """Goal3 推理期参数（默认值与 ``SingleHeadStudyTask`` 的基线一致）。"""

    #: 权重相对路径（规范 §5.2）
    ckpt_rel: str = "goal3_tumor/model.pt"
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
