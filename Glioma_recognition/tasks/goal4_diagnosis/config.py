"""Goal4 辅助诊断配置（规范 §5.2：只声明相对路径）。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal4Config:
    ckpt_rel: str = "goal4_diagnosis/model.pt"
    arch: str = "mednext"
    in_channels: int = 4
    global_size: int = 96
    global_size_mm: float = 192.0
    common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tta_flips: tuple[str, ...] = ("x", "y")
    tta_batch: int = 2
    #: 阳性判定阈值（TumorProbability 低于此值时输出"非肿瘤"口径）
    tumor_threshold: float = 0.5
