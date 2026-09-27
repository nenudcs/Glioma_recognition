"""[运行] 重复影像的确定性预处理：体数据 → 描述子，以及「原文件」发现规则。

两级描述子（都只用**中间 K 层**，与 ``middle_slice`` 的层方向口径一致）：

* **粗描述子**：鲁棒裁剪（0.5/99.5 百分位）→ 盒式池化到 ``coarse_grid`` → 零均值单位范数，
  用于全库余弦检索（亚二次候选生成）；
* **精描述子**：同样的裁剪与池化，但保留 ``fine_grid`` 分辨率并逐层零均值单位方差，
  用于带平移搜索的归一化互相关精排。

另有一个**精确指纹**：中间层的像素 SHA1，命中即判定为「同一份像素」，
概率直接取 1.0（完全一致的拷贝不会漏）。

文件发现（``group_dataset``）只在**全库扫描**时使用，规则与管线 ``data/loader.py``
的新版「原文件」规则一致：序列目录里优先取主名与目录名完全一致的 ``<目录名>.nii(.gz)``，
避免把派生副本当成另一个检查而误报重复。
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

NIFTI_SUFFIXES = (".nii", ".nii.gz")
MASK_HINTS = ("mask", "seg", "label", "roi")

MIN_OVERLAP = 8
EPS = 1e-9


# --------------------------------------------------------------------------
# 参数与特征容器
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DescriptorParams:
    """描述子几何参数。"""

    slices: int = 3
    coarse_grid: int = 12
    fine_grid: int = 64
    shift: int = 1

    def __post_init__(self) -> None:
        if self.slices < 1:
            raise ValueError("slices 必须 >= 1")
        if self.coarse_grid < 2:
            raise ValueError("coarse_grid 必须 >= 2")
        if self.fine_grid < 4:
            raise ValueError("fine_grid 必须 >= 4")
        if self.shift < 0:
            raise ValueError("shift 必须 >= 0")
        if self.fine_grid < self.coarse_grid:
            raise ValueError("fine_grid 必须 >= coarse_grid")


@dataclass(frozen=True)
class VolumeDescriptor:
    """一条序列（一个体数据）的重复检测特征。"""

    exact: str                    # 中间层像素 SHA1
    coarse: np.ndarray            # (slices * coarse_grid²,)，L2 归一化
    fine: np.ndarray              # (slices, fine_grid, fine_grid)，逐层标准化，float16

    def as_dict(self) -> dict[str, object]:
        return {
            "exact": self.exact,
            "coarse_shape": list(self.coarse.shape),
            "fine_shape": list(self.fine.shape),
        }


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def is_image(path: Path) -> bool:
    """输入影像：``.nii`` / ``.nii.gz``，且文件名不含 mask/seg/label/roi。"""
    name = path.name.lower()
    return name.endswith(NIFTI_SUFFIXES) and not any(hint in name for hint in MASK_HINTS)


def stem_of(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def slice_axis(shape: Sequence[int]) -> int:
    """层方向 = 最短轴（与 ``Goal2StitchedTask`` 的推理口径一致）。"""
    return int(np.argmin(shape[:3]))


def middle_index(length: int) -> int:
    return max(0, length // 2)


def to_float_volume(source: np.ndarray | str | Path) -> np.ndarray:
    """内存数组或 NIfTI 路径 → 3-D float32（4-D 折叠取第 0 个体积）。"""
    if isinstance(source, (str, Path)):
        import nibabel as nib

        image = nib.load(str(source))
        try:
            array = np.asarray(image.dataobj, dtype=np.float32)
        finally:
            uncache = getattr(image, "uncache", None)
            if callable(uncache):
                try:
                    uncache()
                except Exception:  # noqa: BLE001
                    pass
    else:
        array = np.asarray(source, dtype=np.float32)
    while array.ndim > 3:
        array = array[..., 0]
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"体数据必须是 3-D（折叠多余轴后），实际 {array.shape}")
    if not np.isfinite(array).all():
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(array, dtype=np.float32)


def middle_slab(volume: np.ndarray | str | Path, slices: int = 3) -> np.ndarray:
    """中间 ``slices`` 层，形状 ``(slices, H, W)``。"""
    array = to_float_volume(volume)
    axis = slice_axis(array.shape)
    length = int(array.shape[axis])
    if length < 1:
        raise ValueError(f"体数据的层方向为空：{array.shape}")
    center = length // 2
    half = slices // 2
    indices = np.clip(np.arange(center - half, center - half + slices), 0, length - 1)
    moved = np.moveaxis(array, axis, 0)
    return np.ascontiguousarray(moved[indices], dtype=np.float32)


def middle_slice_from_array(volume: np.ndarray) -> np.ndarray:
    """内存体数据 → 中间一层（用于精确指纹）。"""
    array = to_float_volume(volume)
    axis = slice_axis(array.shape)
    index = middle_index(int(array.shape[axis]))
    slicer = tuple(index if dim == axis else slice(None) for dim in range(array.ndim))
    return np.asarray(array[slicer])


# --------------------------------------------------------------------------
# 池化与标准化
# --------------------------------------------------------------------------
def _box_edges(length: int, grid: int) -> np.ndarray:
    if grid >= length:
        raise ValueError("grid >= length 时应改用最近邻采样")
    edges = np.round(np.linspace(0.0, float(length), grid + 1)).astype(np.int64)
    edges[0] = 0
    edges[-1] = length
    for index in range(1, edges.size):
        if edges[index] <= edges[index - 1]:
            edges[index] = edges[index - 1] + 1
    return edges


def pool2d(plane: np.ndarray, grid: int) -> np.ndarray:
    """盒式池化到 ``grid × grid``（下采样取均值，上采样取最近邻）。"""
    array = np.asarray(plane, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"pool2d 需要 2-D 平面，实际 {array.shape}")
    height, width = array.shape
    if height < 1 or width < 1:
        raise ValueError(f"空平面：{array.shape}")
    if grid >= height or grid >= width:
        rows = np.round(np.linspace(0, height - 1, grid)).astype(np.int64)
        columns = np.round(np.linspace(0, width - 1, grid)).astype(np.int64)
        return np.ascontiguousarray(array[rows][:, columns], dtype=np.float32)

    rows = _box_edges(height, grid)
    columns = _box_edges(width, grid)
    integral = np.zeros((height + 1, width + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(array, axis=0, dtype=np.float64), axis=1)

    r0, r1 = rows[:-1][:, None], rows[1:][:, None]
    c0, c1 = columns[:-1][None, :], columns[1:][None, :]
    sums = integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]
    counts = (r1 - r0) * (c1 - c0)
    return np.ascontiguousarray(sums / np.maximum(counts, 1), dtype=np.float32)


def zscore(plane: np.ndarray) -> np.ndarray:
    array = np.asarray(plane, dtype=np.float32)
    mean = float(array.mean())
    std = float(array.std())
    if not math.isfinite(std) or std <= EPS:
        return np.zeros_like(array)
    return np.ascontiguousarray((array - mean) / std, dtype=np.float32)


def unit_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= EPS:
        return np.zeros_like(array)
    return np.ascontiguousarray(array / norm, dtype=np.float32)


def fingerprint(array: np.ndarray) -> str:
    """解码后像素的 SHA1（同一份像素 → 同一指纹）。"""
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha1(contiguous.tobytes()).hexdigest()


# --------------------------------------------------------------------------
# 描述子
# --------------------------------------------------------------------------
def descriptor_from_volume(
    volume: np.ndarray | str | Path,
    params: DescriptorParams | None = None,
) -> VolumeDescriptor:
    """体数据（内存数组或 NIfTI 路径）→ :class:`VolumeDescriptor`。"""
    params = params or DescriptorParams()
    array = to_float_volume(volume)
    slab = middle_slab(array, params.slices)

    low, high = (float(value) for value in np.percentile(array, [0.5, 99.5]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low, high = float(np.min(array)), float(np.max(array))
    if high <= low:
        high = low + 1.0
    clipped = np.clip(slab, low, high)

    coarse = unit_vector(
        np.concatenate([pool2d(zscore(plane), params.coarse_grid).ravel() for plane in clipped])
    )
    fine = np.stack(
        [zscore(pool2d(plane, params.fine_grid)) for plane in clipped],
        axis=0,
    ).astype(np.float16)

    return VolumeDescriptor(
        exact=fingerprint(middle_slice_from_array(array)),
        coarse=coarse,
        fine=np.ascontiguousarray(fine),
    )


def descriptor_from_path(
    path: str | Path,
    params: DescriptorParams | None = None,
) -> VolumeDescriptor:
    return descriptor_from_volume(path, params)


def descriptors_from_series(
    series: Iterable[tuple[str, np.ndarray | str | Path]],
    params: DescriptorParams | None = None,
) -> tuple[tuple[str, VolumeDescriptor], ...]:
    """``[(series_uid, image), ...]`` → ``[(series_uid, descriptor), ...]``。"""
    return tuple((str(uid), descriptor_from_volume(image, params)) for uid, image in series)


# --------------------------------------------------------------------------
# 文件发现（全库扫描用；与管线 Loader 的「原文件」规则一致）
# --------------------------------------------------------------------------
def select_original_files(root: Path, files: Sequence[Path]) -> list[Path]:
    """按新版 Loader 规则筛选「原文件」（软版本，找不到唯一同名原文件时保留全部）。"""
    selected: list[Path] = []
    grouped: dict[Path, list[Path]] = defaultdict(list)
    for path in files:
        if len(path.relative_to(root).parts) <= 2:
            selected.append(path)
        else:
            grouped[path.parent].append(path)

    for directory, paths in grouped.items():
        if len(paths) == 1:
            selected.extend(paths)
            continue
        originals = [path for path in paths if stem_of(path) == directory.name]
        selected.extend(originals if len(originals) == 1 else paths)
    return sorted(selected)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def group_dataset(
    root: str | Path,
    skip_dirs: Sequence[Path] = (),
) -> dict[str, list[tuple[str, Path]]]:
    """``{accession: [(series_uid, path), ...]}``：只保留原文件。"""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        return {}
    resolved_skips = [Path(path).resolve() for path in skip_dirs if path is not None]
    candidates: list[Path] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file() or not is_image(path):
            continue
        if any(_is_within(path.resolve(), skip) for skip in resolved_skips):
            continue
        candidates.append(path)

    grouped: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for path in select_original_files(base, candidates):
        relative = path.relative_to(base)
        accession = relative.parts[0] if len(relative.parts) > 1 else stem_of(path)
        if len(relative.parts) > 1 and path.parent != base / relative.parts[0]:
            series_uid = path.parent.name
        else:
            series_uid = stem_of(path)
        grouped[accession].append((series_uid, path))
    return {key: sorted(value) for key, value in sorted(grouped.items())}


def dataset_root_of(source_path: str | Path, accession: str, series_uid: str) -> Path:
    """从 ``Series.source_path`` 反推数据集根目录（全库扫描的起点）。"""
    path = Path(source_path).expanduser().resolve()
    if path.parent.name == series_uid and path.parent.parent.name == accession:
        return path.parent.parent.parent
    if path.parent.name == accession:
        return path.parent.parent
    return path.parent
