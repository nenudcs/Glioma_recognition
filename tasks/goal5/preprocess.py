"""Goal5 的确定性预处理：把 ``Study`` 变成 1mm 公共网格上的多通道体积。

规范 §5.1：``preprocess.py`` 属于 **[运行]** 交付文件，必须是**确定性**的
（不含随机增强），且不得扫描比赛目录——它只接收已加载的领域对象 ``Study``。

设计要点：
1. **固定 1mm 公共网格**：不同序列的层厚常不一致（如 T1C 1mm、FLAIR 3mm），
   必须统一到同一网格，否则多通道无法对齐、掩膜也无法写回；
2. **逐通道独立 z-score**：MRI 强度无绝对物理意义，跨序列做全局归一化会破坏对比；
3. **缺模态用零通道占位**并以同序返回，让下游保持通道语义稳定。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from data.series_selector import select as select_series
from data.structures import Study
from tasks.goal5.config import Goal5Config
from tasks.goal5.spatial import resample_to, spacing_of, target_grid

#: 通道顺序固定，与训练时的 in_channels 一一对应（改动即破坏权重兼容）
CHANNEL_ORDER: tuple[str, ...] = ("t1c", "flair", "t2", "t1")

#: 每个输入通道的**取用链**：优先自己，缺了按训练时的规则用替代模态顶替。
#:
#: ⚠️ **必须与训练侧** ``glioma_track4/configs/preprocess.yaml`` **的** ``channels`` **逐字一致**：
#:
#: .. code-block:: yaml
#:
#:     channels:
#:       - {name: t1c,   fallback: [t1, t2]}
#:       - {name: flair, fallback: [t2]}
#:       - {name: t2,    fallback: []}
#:       - {name: t1,    fallback: []}
#:
#: 此前本文件的通道填充**没有 fallback**（缺就直接填零通道），而训练时是**用替代模态顶替**的。
#: 后果：官方数据里约 47% 的检查号没有 T1 增强 ——
#: 训练时模型见到的是「``t1c`` 通道里装着 T1 影像」，推理时却收到「``t1c`` 通道全零」，
#: **输入分布完全不同** → 分割输出塌陷成空掩膜（实测空掩膜率约 70%）。
#:
#: 注意 ``t1c`` 的链尾是 ``t2``、``flair`` 的链尾也是 ``t2``，所以**同一条 t2 序列可能
#: 同时顶替两个通道** —— 这与训练侧 ``pick_series`` 的行为一致，不要"顺手去重"。
CHANNEL_FALLBACK: dict[str, tuple[str, ...]] = {
    "t1c": ("t1c", "t1", "t2"),
    "flair": ("flair", "t2"),
    "t2": ("t2",),
    "t1": ("t1",),
}

#: 参考网格的模态优先级（选层厚最接近 1mm 的那个作为基准）。
#:
#: ⚠️ **必须与训练侧一致**：训练侧 ``dataset.py`` 用的是 ``("t1c", "flair", "t2", "t1")``。
#: 此前本文件写的是 ``("t1c", "t1", "flair", "t2")`` —— 缺 T1CE 时训练用 **FLAIR** 建网格、
#: 推理用 **T1** 建网格，**两边建出形状与 affine 都不同的公共网格**（约 47% 的检查号受影响）。
_REF_PRIORITY: tuple[str, ...] = ("t1c", "flair", "t2", "t1")

#: **公共网格体素数上限**（防爆护栏，详见 ``tasks/_common/volume.MAX_GRID_VOXELS``）。
#: 一例峰值内存 ≈ ``(C+13)×V×4B``；真实 1mm 脑 V≈8.9M 无压力，触发护栏的几乎只有
#: "参考序列 spacing/affine 元数据异常 → 病态巨大网格"。明确报错 + 容错跳过，
#: 远好于"被平台 OOM-kill → 整批评测作废"。
MAX_GRID_VOXELS: int = int(os.environ.get("GLIOMA_MAX_GRID_VOXELS", "200_000_000")
                           .replace("_", "") or 200_000_000)


def _check_grid_size(grid_shape: tuple, study) -> None:
    """公共网格防爆护栏：与 ``tasks/_common/volume._check_grid_size`` 同口径。"""
    n_vox = int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])
    if n_vox <= MAX_GRID_VOXELS:
        return
    est = (4 + 13) * n_vox * 4 / 1e9
    raise ValueError(
        f"study {getattr(study, 'accession_number', '?')!r}: 公共网格 "
        f"{tuple(int(x) for x in grid_shape)} = {n_vox / 1e6:.0f} M 体素，"
        f"超过上限 {MAX_GRID_VOXELS / 1e6:.0f} M（GLIOMA_MAX_GRID_VOXELS），"
        f"预计单例峰值约 {est:.0f} GB —— 容器会被 OOM-kill。\n"
        f"  最常见成因：参考序列的 spacing/affine 元数据异常（target_grid 原样采用了病态网格）。\n"
        f"  容错模式（GLIOMA_LOADER_TOLERANT=1）下本例被跳过、其余照常产出；"
        f"或确认数据确实超大后 export GLIOMA_MAX_GRID_VOXELS={n_vox} 临时放行。"
    )


@dataclass(frozen=True)
class PreparedVolume:
    """预处理结果：公共网格体积 + 几何 + 各通道来源信息。"""

    volume: np.ndarray                     # [C, D, H, W] float32
    affine: np.ndarray                     # 公共网格 voxel→RAS
    shape: tuple[int, int, int]
    channel_sources: dict[str, dict]       # {模态: {"series_uid":..., "spacing":...}}
    missing: tuple[str, ...]               # 缺失的通道（已用零占位）


def _zscore(vol: np.ndarray, clip: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """按前景体素做 z-score（只统计非零区域，避免空气主导统计量）。"""
    fg = vol[vol > 0]
    if fg.size < 16:
        return vol.astype(np.float32)
    lo, hi = np.percentile(fg, list(clip))
    x = np.clip(vol, lo, hi)
    m, s = float(x[vol > 0].mean()), float(x[vol > 0].std())
    return ((x - m) / (s if s > 1e-6 else 1.0)).astype(np.float32)


def _any_series_ref(study: Study):
    """该 Study 里**任意一路有影像**的序列（几何参考用）；没有则 ``None``。

    用途：4 个通道一路都填不上时（整例序列被数据信息表标成 `其他`，或只有 DWI/ADC/SWI），
    公共网格仍需要一个参考几何 —— **几何与模态无关**，任意一路序列的 affine/shape
    都能把网格建出来，掩膜也才有地方重采样。
    """
    for s in (getattr(study, "series", None) or ()):
        if getattr(s, "image", None) is not None and getattr(s, "affine", None) is not None:
            return s
    return None


def build_volume(study: Study, cfg: Goal5Config) -> PreparedVolume:
    """把 ``Study`` 归一化到 1mm 公共网格，返回多通道体积。

    4 个通道**一个都填不上**时不再抛错（**全放开口径**）：改为"全零通道 + 借任意一路
    序列的几何"，``missing`` 会把"四个通道全缺"暴露出来（`task.py` 会转成
    ``context.warnings``），不静默。

    ⚠️ 为什么必须这样（这是一次真实"整批评测作废"的根因）：
    本文件此前是 `tasks/_common/volume.py` 的**早期副本**，那份已经改成全放开口径、
    这份还保留着硬失败 —— 只要**一例**检查的序列全被表标成 `其他`（或没有目标模态），
    这里就抛 `ValueError`；而 `core/runner.py::_run_streaming` 对整批只包了一层 try
    （**没有 per-case 容错**）→ staging 被 rmtree、**777 例答案全部作废**。
    审计要求两份实现**同源**，这里与 `tasks/_common/volume.py` 对齐。

    Raises:
        ValueError: 该 Study **连一路影像都没有**（规范 §9.1：不可降级输入错误）。
            注意这与"没有目标模态"是两回事：后者现在会全零通道照走。
    """
    # 先取**全部候选模态**，再按 CHANNEL_FALLBACK 逐通道顶替 ——
    # 与训练侧 ``dataset.pick_series``（``[ch["name"]] + ch["fallback"]``）同口径。
    available = select_series(study, ("t1c", "flair", "t2", "t1"))
    picked: dict[str, object] = {}
    picked_mod: dict[str, str] = {}
    for name in CHANNEL_ORDER:
        for cand in CHANNEL_FALLBACK[name]:
            if cand in available:
                picked[name] = available[cand]
                picked_mod[name] = cand
                break
    if not picked:
        ref = _any_series_ref(study)
        if ref is None:
            seen = [(s.series_uid, s.modality) for s in list(study.series)[:6]]
            from data.modality_fallback import describe_sources

            raise ValueError(
                f"study {study.accession_number!r} 没有任何可用影像"
                f"（共 {len(study.series)} 条；uid/描述前几条={seen}）。"
                f"注意：**只有 `其他` 序列 / 只有 DWI 的检查不算这一类**，"
                f"那种情况会全零通道照走。"
                f"（已自动尝试转模态识别："
                f"{describe_sources([s.source_path for s in study.series])}）"
            )
    else:
        ref_key = next((k for k in _REF_PRIORITY if k in picked), next(iter(picked)))
        ref = picked[ref_key]
    # ``max_factor`` 必须走配置（训练侧 = 1.5），不能用 target_grid 的默认值
    grid_shape, grid_affine = target_grid(ref.image.shape, ref.affine,
                                          tuple(cfg.common_spacing),
                                          max_factor=cfg.max_spacing_factor)
    _check_grid_size(grid_shape, study)

    # 2) 逐通道重采样 + 归一化
    chans: list[np.ndarray] = []
    sources: dict[str, dict] = {}
    missing: list[str] = []
    for name in CHANNEL_ORDER:
        s = picked.get(name)
        if s is None:
            chans.append(np.zeros(grid_shape, dtype=np.float32))
            missing.append(name)
            continue
        arr = np.asarray(s.image, dtype=np.float32)
        if arr.shape != tuple(grid_shape) or not np.allclose(s.affine, grid_affine):
            arr = resample_to(arr, s.affine, grid_shape, grid_affine, order=1)
        chans.append(_zscore(arr))
        sources[name] = {
            "series_uid": s.series_uid,
            # 实际顶替用的模态：``t1c`` 通道可能装着 ``t1``/``t2`` 的影像（与训练侧同规则）。
            # 下游 ``task._restore`` 只看 ``series_uid`` 决定掩膜写回哪条序列，加这个键不影响。
            "modality": picked_mod.get(name, name),
            "spacing": spacing_of(s.affine),
            "shape": tuple(int(x) for x in s.image.shape),
        }

    vol = np.stack(chans, axis=0).astype(np.float32)
    return PreparedVolume(volume=vol, affine=np.asarray(grid_affine, dtype=np.float64),
                          shape=tuple(int(x) for x in grid_shape),
                          channel_sources=sources, missing=tuple(missing))
