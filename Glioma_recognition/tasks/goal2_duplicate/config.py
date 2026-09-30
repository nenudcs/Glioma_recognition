"""Goal2-B 重复影像检测配置。

规范 §5.2：只声明**相对路径**，绝对根目录由 ``Settings.ckpt_root`` 提供。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DuplicateConfig:
    #: 重复影像用的嵌入权重（规范 §5.2 约定 encoder.pt）
    ckpt_rel: str = "goal2_duplicate/encoder.pt"

    arch: str = "mednext"
    in_channels: int = 4
    global_size: int = 96
    global_size_mm: float = 192.0
    common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    #: 规范上限：每个 accession 最多参与 200 对
    topk: int = 200
    #: 实际提交的候选数（≤ topk）。三项指标对低分长尾敏感，
    #: 未提交的对按 0 计，因此"少而准"通常优于"多而杂"。
    per_study_k: int = 50
    #: 低于此概率的对不进入候选（避免长尾淹没）
    min_prob: float = 0.005
    #: 手工指纹（几何+强度）与学习式嵌入的融合权重。
    #: 实测这一项是决定性的：纯嵌入 AUC-PR 0.120 → 融合后 0.600。
    fp_weight: float = 0.5
