"""Goal5 的分割配置。

规范 §5.2：本文件只声明**相对路径**，绝对根目录由 ``core.config.Settings.ckpt_root``
提供，模型代码中不得出现任何硬编码的比赛路径。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal5Config:
    #: 相对 checkpoint 根目录的权重路径（规范 §5.2 约定）
    core_ckpt_rel: str = "goal5_segmentation/core.pt"
    flair_ckpt_rel: str = "goal5_segmentation/flair.pt"

    #: 网络结构（必须与训练时一致，否则权重加载失败）
    arch: str = "mednext"
    in_channels: int = 4
    base: int = 32
    depth: int = 4
    blocks_per_stage: int = 2
    k: int = 3
    expand: int = 2
    aniso_z: bool = False
    max_ch: int = 320

    #: 形态学后处理
    min_tumor_voxels: int = 30
    keep_components: int = 3
    bridge_mm: float = 10.0

    #: 公共网格（1mm）与滑窗
    common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    #: 层厚超过 ``max_spacing_factor × common_spacing`` 的轴**保持原始 spacing**
    #: （把 3~5mm 层厚强行插值到 1mm 只会产生虚假细节）。
    #:
    #: ⚠️ **必须与训练侧一致**：训练侧 ``configs/preprocess.yaml`` 的
    #: ``geometry.max_spacing_factor`` 默认 **1.5**。此前本值缺失、由
    #: ``tasks/_common/spatial.target_grid`` 的默认参数 **4.0** 兜底 ——
    #: 于是同一条 3mm 层厚的 FLAIR：训练时保持 3mm，推理时被插值到 1mm，
    #: **公共网格形状与训练分布不一致** → 输入 OOD → 分割输出塌陷（空掩膜）。
    max_spacing_factor: float = 1.5
    patch: tuple[int, int, int] = (96, 96, 96)
    #: 滑窗重叠：训练侧 ``preprocess.yaml`` 的 ``inference.overlap`` 为 **0.4**。
    overlap: float = 0.4
    tta_flips: tuple[str, ...] = ("x", "y")
    tta_batch: int = 2
    global_size: int = 96
    global_size_mm: float = 192.0

    #: 二值化阈值；训练完成后由 14_calibrate_thresholds.py 写回权重，
    #: 加载时以权重里记录的阈值为准，这里的默认值只作兜底。
    default_thresholds: tuple[float, float] = (0.5, 0.5)

    #: 通道语义（与训练时的 seg 通道顺序一致，改动即破坏权重兼容）
    core_channel: int = 0
    flair_channel: int = 1
