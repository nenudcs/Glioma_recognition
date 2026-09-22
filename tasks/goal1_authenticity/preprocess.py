"""[运行] 确定性预处理（规范 §5.1：preprocess.py 只做确定性预处理）。

体数据 → 检查级输入张量：

1. 0.5/99.5 百分位截断归一化；
2. 最短轴作为层方向，丢弃标准差过低的空层，uniform 取 K 层；
3. 每层双线性缩放到固定尺寸，复制为 3 通道。

本文件不依赖任何训练专用模块（规范 §17.1），训练侧通过
``dataset.py`` / ``augmentations.py`` 复用这里的确定性部分。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

NIFTI_SUFFIXES = (".nii", ".nii.gz")
DEFAULT_SLICES_PER_CASE = 16
DEFAULT_IMAGE_SIZE = 224
DEFAULT_MIN_STD = 0.05


def _alternate_suffix(path: Path) -> Path | None:
    """``x.nii.gz`` <-> ``x.nii``：比赛数据里偶有后缀与实际压缩格式不符的文件。"""
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return path.with_name(name[: -len(".gz")])
    if name.lower().endswith(".nii"):
        return path.with_name(name + ".gz")
    return None


def _sanitize(data: np.ndarray) -> np.ndarray:
    """统一成 3-D float32，并把 NaN/Inf 置零。"""
    array = np.asarray(data, dtype=np.float32)
    while array.ndim > 3:
        array = array[..., 0]
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"体数据在折叠后必须是 3-D，实际为 {array.shape}")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return array


def load_volume(path: str | Path) -> np.ndarray:
    """从磁盘读入一个 3-D 体数据（只读一层时请用别的方式，避免整卷解码）。"""
    import nibabel as nib

    path = Path(path)
    try:
        image = nib.load(str(path))
    except Exception as first_error:  # noqa: BLE001
        alternative = _alternate_suffix(path)
        if alternative is None or not alternative.is_file():
            raise
        logger.warning(
            "goal1: %s 读取失败（%s: %s），改用 %s",
            path,
            type(first_error).__name__,
            first_error,
            alternative,
        )
        image = nib.load(str(alternative))
    try:
        data = np.asarray(image.dataobj, dtype=np.float32)
    finally:
        uncache = getattr(image, "uncache", None)
        if callable(uncache):
            try:
                uncache()
            except Exception:  # noqa: BLE001
                pass
    volume = _sanitize(data)
    if isinstance(volume, np.memmap) or not volume.flags.owndata:
        volume = np.array(volume, dtype=np.float32, copy=True)
    return volume


def to_float_volume(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, (str, Path)):
        return load_volume(source)
    return _sanitize(source)


def normalize_volume(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(volume, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(volume)) if np.isfinite(volume).any() else 0.0
        hi = lo + 1.0
    volume = np.clip(volume, lo, hi)
    std = float(volume.std())
    return (volume - float(volume.mean())) / (std if std > 1e-6 else 1.0)


def slice_axis(volume: np.ndarray) -> int:
    """最短轴作为层方向（对任何来源都适用）。"""
    return int(np.argmin(volume.shape))


def usable_indices(volume: np.ndarray, axis: int, min_std: float = DEFAULT_MIN_STD) -> np.ndarray:
    others = tuple(i for i in range(volume.ndim) if i != axis)
    with np.errstate(invalid="ignore"):
        stds = np.nan_to_num(volume.std(axis=others), nan=0.0)
    keep = np.flatnonzero(stds > min_std)
    if keep.size == 0:
        keep = np.arange(volume.shape[axis])
    return keep


def pick_uniform_positions(count: int, k: int) -> np.ndarray:
    """确定性 uniform 取样——推理与验证统一用这个。"""
    if count <= 0:
        return np.zeros(k, dtype=int)
    pos = np.round(np.linspace(0, count - 1, k)).astype(int)
    return np.clip(pos, 0, count - 1)


def sample_positions(count: int, k: int, mode: str = "uniform", rng=None) -> np.ndarray:
    """训练用层采样（``random`` / ``uniform`` / ``center``）；推理固定走 uniform。"""
    if count <= 0:
        return np.zeros(k, dtype=int)
    if mode == "random":
        if rng is None:
            raise ValueError("random 采样需要 rng")
        pos = np.array([rng.randrange(count) for _ in range(k)])
        pos.sort()
    elif mode == "center":
        pos = np.round(np.linspace(0.3, 0.7, k) * (count - 1)).astype(int)
    else:
        pos = pick_uniform_positions(count, k)
    return np.clip(pos, 0, count - 1)


def resize2d(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape
    if (height, width) == (size, size):
        return image.astype(np.float32, copy=False)
    yi = np.linspace(0, height - 1, size)
    xi = np.linspace(0, width - 1, size)
    y0 = np.floor(yi).astype(int)
    x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = (yi - y0)[:, None].astype(np.float32)
    wx = (xi - x0)[None, :].astype(np.float32)
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def volume_to_slices(
    source: str | Path | np.ndarray,
    k: int = DEFAULT_SLICES_PER_CASE,
    size: int = DEFAULT_IMAGE_SIZE,
    min_std: float = DEFAULT_MIN_STD,
    mode: str = "uniform",
    rng=None,
    augment=None,
) -> np.ndarray:
    """体数据 -> ``(K, 3, size, size)`` float32；``augment`` 仅供训练侧注入。"""
    volume = normalize_volume(to_float_volume(source))
    axis = slice_axis(volume)
    usable = usable_indices(volume, axis, min_std)
    if mode == "uniform" and rng is None:
        positions = usable[pick_uniform_positions(usable.size, k)]
    else:
        positions = usable[sample_positions(usable.size, k, mode, rng)]
    stack = np.moveaxis(np.take(volume, positions, axis=axis), axis, 0)
    images = np.stack([resize2d(stack[i], size) for i in range(stack.shape[0])])
    if augment is not None:
        images = augment(images, rng)
    return np.ascontiguousarray(
        np.repeat(images[:, None, :, :], 3, axis=1),
        dtype=np.float32,
    )
