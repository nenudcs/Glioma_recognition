"""共享的多通道体积预处理：``Study`` → 1mm 公共网格。

Goal1/2/3/4/5 都需要把检查的多个序列对齐到同一网格后再喂给共享骨干，
因此该逻辑属于**公共模块**（规范 §5.1 禁止每个 Goal 各写一套 NIfTI 读取）。

设计要点：
1. **固定 1mm 公共网格**：不同序列层厚常不一致（T1C 1mm / FLAIR 3mm），
   不统一网格则多通道无法对齐、掩膜也无法写回；
2. **逐通道独立 z-score**：MRI 强度无绝对物理意义；
3. **缺模态零占位**并保持通道顺序固定，让下游语义稳定。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from data.series_selector import guess_modality as _guess_modality
from data.series_selector import select as select_series
from data.structures import Series, Study
from tasks._common.spatial import resample_to, spacing_of, target_grid

#: 通道顺序固定，与训练时 in_channels 一一对应（改动即破坏权重兼容）
CHANNEL_ORDER: tuple[str, ...] = ("t1c", "flair", "t2", "t1")

#: 每个输入通道的**取用链**：优先自己，缺了按训练时的规则用替代模态顶替。
#:
#: ⚠️ **必须与训练侧** ``glioma_track4/configs/preprocess.yaml`` **的** ``channels``
#: **逐字一致**（仓内训练入口 ``tasks/_common/training/helpers.build_datasets``
#: 直接 ``load_config("preprocess.yaml")`` 并把 ``pre_cfg`` 交给
#: ``src.data.dataset.GliomaDataset``，所以那份 yaml 就是训练事实来源）：
#:
#: .. code-block:: yaml
#:
#:     channels:
#:       - {name: t1c,   fallback: [t1, t2]}
#:       - {name: flair, fallback: [t2]}
#:       - {name: t2,    fallback: []}
#:       - {name: t1,    fallback: []}
#:
#: 此前本文件**没有 fallback**（缺就填零通道），而训练时是**用替代模态顶替**的。
#: 后果：官方数据里约 47% 的检查号没有 T1 增强 ——
#: 训练时共享骨干见到的是「``t1c`` 通道里装着 T1/T2 影像」，推理时却收到「``t1c`` 全零」，
#: **输入分布完全不同** → Goal1/2/3/4 四个头（真实性 / 拼接 / 肿瘤概率 / 结构化字段+嵌入）
#: 全部拿到 OOD 输入。这与 Goal5 空掩膜是**同一个成因**，只是 Goal5 还额外把结果落成了掩膜，
#: 所以先被看见。
#:
#: 注意 ``t1c`` 的链尾是 ``t2``、``flair`` 的链尾也是 ``t2``，所以**同一条 t2 序列可能
#: 同时顶替两个通道** —— 这与训练侧 ``pick_series`` 的行为一致，不要"顺手去重"。
CHANNEL_FALLBACK: dict[str, tuple[str, ...]] = {
    "t1c": ("t1c", "t1", "t2"),
    "flair": ("flair", "t2"),
    "t2": ("t2",),
    "t1": ("t1",),
}

#: 参考网格的模态优先级。
#:
#: ⚠️ **必须与训练侧一致**：``glioma_track4/src/data/dataset.py::build_case_volume``
#: 用的是 ``("t1c", "flair", "t2", "t1")``。此前本文件写的是
#: ``("t1c", "t1", "flair", "t2")`` —— 缺 T1CE 时训练用 **FLAIR** 建网格、
#: 推理用 **T1** 建网格，**两边建出形状与 affine 都不同的公共网格**
#: （约 47% 的检查号受影响），而公共网格决定掩膜写回的空间。
_REF_PRIORITY: tuple[str, ...] = ("t1c", "flair", "t2", "t1")

#: 层厚超过 ``max_factor × common_spacing`` 的轴**保持原始 spacing**。
#:
#: ⚠️ **必须与训练侧一致**：``preprocess.yaml`` 的 ``geometry.max_spacing_factor`` 是
#: **1.5**。此前本文件没传这个参数，于是由 ``tasks._common.spatial.target_grid`` 的
#: 默认值 **4.0** 兜底 —— 同一条 3mm 层厚的 FLAIR：训练保持 3mm、推理被插值到 1mm，
#: 网格形状与训练分布不一致。
DEFAULT_MAX_SPACING_FACTOR: float = 1.5

#: **公共网格体素数上限**（防爆护栏）。超过即抛错 —— 明确失败，而不是让容器被
#: 平台 OOM-kill（被平台杀 = 整批评测作废；明确报错 + 容错模式 = 只丢这一例，
#: 日志里还有完整原因）。
#:
#: 阈值怎么来的：一例的峰值内存 ≈ ``(C+13) × V × 4B``（RAM 里 volume/z-score 拷贝/
#: EDT 后处理 + GPU 或 RAM 上的 x0/acc/wacc/seg_prob）。真实 1mm 脑 V≈8.9M，
#: 峰值不到 1 GB；**要打爆 32GB 容器，V 得在 4~6 亿** —— 触发本护栏的几乎只有一种
#: 情况：**某条序列的 spacing/affine 元数据异常**，``target_grid`` 原样采用了一个
#: 病态巨大的网格。默认 2 亿（预计单例峰值 ~14 GB，留 2 倍安全余量）。
MAX_GRID_VOXELS: int = int(os.environ.get("GLIOMA_MAX_GRID_VOXELS", "200_000_000")
                           .replace("_", "") or 200_000_000)


def _check_grid_size(grid_shape: tuple, study) -> None:
    """公共网格防爆护栏：体素数超限 → **带完整成因分析的明确报错**。

    与"被平台 OOM-kill"的区别：这里是 Python 异常，``GLIOMA_LOADER_TOLERANT=1``
    的逐例容错会**跳过这一例、继续跑其余 700+ 例**，staging 也不会被 rmtree；
    而容器被杀 = 整批作废且日志里没有任何原因。
    """
    n_vox = int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])
    if n_vox <= MAX_GRID_VOXELS:
        return
    est = (4 + 13) * n_vox * 4 / 1e9                     # (C+13)×V×4B，GB
    raise ValueError(
        f"study {getattr(study, 'accession_number', '?')!r}: 公共网格 "
        f"{tuple(int(x) for x in grid_shape)} = {n_vox / 1e6:.0f} M 体素，"
        f"超过上限 {MAX_GRID_VOXELS / 1e6:.0f} M（GLIOMA_MAX_GRID_VOXELS）。"
        f"预计这一例的峰值内存约 {est:.0f} GB —— 32GB 容器会被直接 OOM-kill。\n"
        f"  最常见成因：**参考序列的 spacing/affine 元数据异常**（如 affine 声称接近 "
        f"1mm 但 shape 是 1024³ 级），``target_grid`` 于是原样采用病态网格。\n"
        f"  排查：用 nib 查该例各序列的 shape 与 spacing：\n"
        f"    python3 - <<'PY'\n"
        f"    import nibabel as nib, glob\n"
        f"    for p in glob.glob('<该例目录>/**/*.nii*'):\n"
        f"        img = nib.load(p)\n"
        f"        print(p, img.shape, [round(float(v), 3) for v in "
        f"nib.affines.voxel_sizes(img.affine)])\n"
        f"    PY\n"
        f"  处置：修好该例的 NIfTI 元数据后重跑；或（确认数据真的就是超大体积时）"
        f"export GLIOMA_MAX_GRID_VOXELS={n_vox} 临时放行。\n"
        f"  容错模式（GLIOMA_LOADER_TOLERANT=1）下本例会被跳过、其余照常产出。"
    )


@dataclass(frozen=True)
class PreparedVolume:
    """预处理结果：公共网格体积 + 几何 + 各通道来源信息。"""

    volume: np.ndarray                     # [C, D, H, W] float32
    affine: np.ndarray                     # 公共网格 voxel→RAS
    shape: tuple[int, int, int]
    channel_sources: dict[str, dict]       # {模态: {"series_uid":..., "spacing":...}}
    missing: tuple[str, ...]               # 缺失的通道（已用零占位）


def zscore(vol: np.ndarray, clip: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """按前景体素做 z-score（只统计非零区域，避免空气主导统计量）。"""
    fg = vol[vol > 0]
    if fg.size < 16:
        return vol.astype(np.float32)
    lo, hi = np.percentile(fg, list(clip))
    x = np.clip(vol, lo, hi)
    m, s = float(x[vol > 0].mean()), float(x[vol > 0].std())
    return ((x - m) / (s if s > 1e-6 else 1.0)).astype(np.float32)


def _no_usable_series_message(study) -> str:
    """「没有任何可用影像」的报错原文：带上"看到了什么" + 两类成因的区分。

    只报一句"没有任何可用影像"时，既不知道序列叫什么、也无从判断是**命名/路径问题**
    还是**该检查本来就没有目标模态**，而这段堆栈还常常埋在 DataLoader worker 里
    （看不出是路径问题）。因此把两类信息都拼进报错：

    - 每条序列的 ``uid`` / ``描述``，并标注描述的性质：
      ``未解析（描述退化成目录名/UID）`` = 数据信息表与 sidecar 都没给值；
      ``描述里没有模态关键词`` = 拿到了值（表/sidecar）但不是目标模态（如 ``其他``）；
    - 提醒另一半口径：**只有 `其他` 序列 / 只有 DWI 的检查不再走这条路**
      （全放开口径下会"全零通道 + 借几何"照走，见 glioma_track4 README §7.2.1），
      所以这里失败基本等于"这个检查连一路影像都没有"。

    ⚠️ 两个工程各有一份 ``volume.py``（不合并），审计要求**实现同源**，
    因此本函数体内**不能**出现各自工程特有的模块路径 —— 项目相关的辅助函数
    一律走模块级别的别名 import（见 :data:`_guess_modality`）。
    """
    seen = []
    for s in list(study.series)[:6]:
        uid = str(getattr(s, "series_uid", "?") or "?")
        desc = str(getattr(s, "modality", "") or "")
        if not desc or desc == uid:
            note = "未解析（描述退化成目录名/UID）"
        elif _guess_modality(desc) is None:
            note = "描述里没有模态关键词"
        else:
            note = ""
        seen.append((uid, desc, note) if note else (uid, desc))
    return (
        f"study {study.accession_number!r} 没有任何可用影像"
        f"（共 {len(study.series)} 条序列；uid/描述前几条={seen}）。"
        f"注意：**只有 `其他` 序列 / 只有 DWI 的检查不算这一类**，"
        f"那种情况会全零通道照走。"
        f"若 uid 是哈希或 DICOM UID，说明序列类型没读到：确认数据根下有 "
        f"SeriesType.xlsx 或同名 .json sidecar，"
        f"或 export GLIOMA_LABELS_DIR=<它所在目录>；"
        f"详见 glioma_track4/docs/DATASET_ROOT_TROUBLESHOOT.md"
        f"「病例数正常、却报无任何可用序列」"
    )


def _any_series_ref(study: Study) -> "Series | None":
    """该 Study 里**任意一路有影像**的序列（几何参考用），没有则 ``None``。

    用途：4 个通道一路都填不上时（整例序列被数据信息表标成 `其他`，或只有
    DWI/ADC/SWI），公共网格仍需要一个参考几何 —— 而**几何与模态无关**，
    任意一路序列的 affine/shape 都能把网格建出来，掩膜也才有地方重采样。
    """
    for s in getattr(study, "series", None) or ():
        if getattr(s, "image", None) is not None and getattr(s, "affine", None) is not None:
            return s
    return None


def build_volume(study: Study, common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 channels: tuple[str, ...] = CHANNEL_ORDER,
                 max_factor: float = DEFAULT_MAX_SPACING_FACTOR) -> PreparedVolume:
    """把 ``Study`` 归一化到公共网格，返回多通道体积。

    4 个通道**一个都填不上**时不再抛错（**全放开口径**，与算法工程
    ``glioma_track4`` 一致，见那里 README §7.2.1）：改为"全零通道 + 借任意一路序列
    的几何"。这类检查的成因是序列被数据信息表标成 `其他`（或只有 DWI/ADC/SWI）——
    它们不属于这 4 个通道，但**影像与掩膜都在**、掩膜也仍在正确的任务空间里
    （掩膜角色只依赖模态），丢整例就是白丢数据。代价是这一例回传近噪声梯度，
    所以 ``missing`` 会把"四个通道全缺"暴露出来，不静默。

    Raises:
        ValueError: 该 Study **连一路影像都没有**（规范 §9.1：不可降级输入错误）。
            注意这与"没有目标模态"是两回事：后者现在会全零通道照走。
    """
    # 先取**全部候选模态**（各通道取用链的并集），再逐通道按链顶替 ——
    # 与训练侧 ``dataset.pick_series``（``[ch["name"]] + ch["fallback"]``）同口径。
    wanted: list[str] = []
    for name in channels:
        for cand in CHANNEL_FALLBACK.get(name, (name,)):
            if cand not in wanted:
                wanted.append(cand)
    available = select_series(study, tuple(wanted))
    picked: dict[str, Series] = {}
    picked_mod: dict[str, str] = {}
    for name in channels:
        for cand in CHANNEL_FALLBACK.get(name, (name,)):
            if cand in available:
                picked[name] = available[cand]
                picked_mod[name] = cand
                break
    if not picked:
        ref = _any_series_ref(study)
        if ref is None:
            raise ValueError(_no_usable_series_message(study))
    else:
        ref_key = next((k for k in _REF_PRIORITY if k in picked), next(iter(picked)))
        ref = picked[ref_key]
    # ``max_factor`` 必须走常量（训练侧 = 1.5），不能用 target_grid 的默认值 4.0
    grid_shape, grid_affine = target_grid(ref.image.shape, ref.affine,
                                         tuple(common_spacing), max_factor=max_factor)
    _check_grid_size(grid_shape, study)

    chans: list[np.ndarray] = []
    sources: dict[str, dict] = {}
    missing: list[str] = []
    for name in channels:
        s = picked.get(name)
        if s is None:
            chans.append(np.zeros(grid_shape, dtype=np.float32))
            missing.append(name)
            continue
        arr = np.asarray(s.image, dtype=np.float32)
        if arr.shape != tuple(grid_shape) or not np.allclose(s.affine, grid_affine):
            arr = resample_to(arr, s.affine, grid_shape, grid_affine, order=1)
        chans.append(zscore(arr))
        sources[name] = {
            "series_uid": s.series_uid,
            # 实际顶替用的模态：``t1c`` 通道可能装着 ``t1``/``t2`` 的影像（与训练侧同规则）
            "modality": picked_mod.get(name, name),
            "spacing": spacing_of(s.affine),
            "shape": tuple(int(x) for x in s.image.shape),
        }

    vol = np.stack(chans, axis=0).astype(np.float32)
    return PreparedVolume(volume=vol, affine=np.asarray(grid_affine, dtype=np.float64),
                          shape=tuple(int(x) for x in grid_shape),
                          channel_sources=sources, missing=tuple(missing))


def global_view(vol: np.ndarray, size_mm: float = 192.0, out: int = 96,
                spacing: float = 1.0, center: np.ndarray | None = None) -> np.ndarray:
    """取固定**物理尺寸**立方体并缩放到 ``out³``。

    分类/特殊影像/嵌入这三个"全局头"在训练时看到的是固定物理尺度的整脑视图，
    推理侧必须用同样尺度，否则训练与推理的输入分布不一致、全局头性能退化。
    """
    from scipy.ndimage import zoom

    c, d, h, w = vol.shape
    half = float(size_mm) / max(1e-6, float(spacing)) / 2.0
    if center is None:
        nz = np.argwhere(np.abs(vol).sum(0) > 1e-3)
        ctr = ((nz.min(0) + nz.max(0)) / 2.0) if len(nz) else np.array([d / 2, h / 2, w / 2])
    else:
        ctr = np.asarray(center, float)

    sl, pad_lo = [], []
    for i, sz in enumerate((d, h, w)):
        s0 = int(round(ctr[i] - half))
        lo = max(0, -s0)
        s0 = max(0, s0)
        s1 = min(sz, int(round(ctr[i] + half)))
        need = int(round(2 * half)) - (s1 - s0)
        hi = max(0, need - lo)
        sl.append((s0, s1))
        pad_lo.append((lo, hi))

    sub = vol[(slice(None),) + tuple(slice(a, b) for a, b in sl)]
    pad = ((0, 0),) + tuple(pad_lo)
    if any(p != (0, 0) for p in pad[1:]):
        sub = np.pad(sub, pad)
    f = [1.0] + [out / max(1, s) for s in sub.shape[1:]]
    return zoom(sub, f, order=1).astype(np.float32)
