"""[运行] 拼接检测的确定性预处理（规范 §5.1：确定性、不扫描目录、不写结果文件）。

约定（与数据 Loader 的 3-D 结构一致，任何来源体数据都适用）：

* **层方向** = 最短轴（``np.argmin(shape)``）：脑 MRI 常见几何下最短轴就是层方向；
* **亮度尺度** = 全卷 ``mean|I|``：把层间差异无量纲化，使不同机器/序列/对比度可比；
* 只在体素网格上计算层间差异，**不产出掩膜**，因此不涉及 affine 变换链。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

NIFTI_SUFFIXES = (".nii", ".nii.gz")


def to_float_volume(source: np.ndarray | str | Path) -> np.ndarray:
    """``Series.image``（内存数组）或 NIfTI 路径 → 3-D float32 体数据。"""
    if isinstance(source, (str, Path)):
        array = _load_nifti(Path(source))
    else:
        array = np.asarray(source, dtype=np.float32)
    while array.ndim > 3:
        array = array[..., 0]                      # 4-D 及以上：与 Loader 一致取第 0 个体积
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"体数据必须是 3-D（折叠多余轴后），实际 {array.shape}")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(array, dtype=np.float32)


def _load_nifti(path: Path) -> np.ndarray:
    """研发/标定侧读盘（比赛运行链路只用 ``Series.image``，不读文件）。"""
    import nibabel as nib

    try:
        image = nib.load(str(path))
    except Exception:  # noqa: BLE001 - 赛方数据偶有后缀与实际格式不符
        name = path.name
        if name.lower().endswith(".nii.gz"):
            alternative = path.with_name(name[: -len(".gz")])
        elif name.lower().endswith(".nii"):
            alternative = path.with_name(name + ".gz")
        else:
            raise
        image = nib.load(str(alternative))
    try:
        return np.asarray(image.dataobj, dtype=np.float32)
    finally:
        uncache = getattr(image, "uncache", None)
        if callable(uncache):
            try:
                uncache()
            except Exception:  # noqa: BLE001
                pass


def slice_axis(volume: np.ndarray) -> int:
    """层方向 = 最短轴。"""
    return int(np.argmin(volume.shape[:3]))


def move_slices_first(volume: np.ndarray) -> np.ndarray:
    """把层方向搬到第 0 轴，返回 ``(layers, ...)`` 视图。"""
    return np.moveaxis(volume, slice_axis(volume), 0)


def brightness_scale(volume: np.ndarray) -> float:
    """无量纲化用的亮度尺度：``mean|I|``，全 0/异常时退回 ``std`` 或 1。"""
    scale = float(np.mean(np.abs(volume)))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = float(np.std(volume))
    if not np.isfinite(scale) or scale <= 1e-6:
        scale = 1.0
    return scale
