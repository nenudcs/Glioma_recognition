"""[运行] 重复影像的相似度模型（无神经网络参数，纯函数：两个描述子 → 相似度）。

精排相似度 = **带小范围平移搜索的归一化互相关（NCC）**，对「轻微调整」不敏感：

* 亮度/对比度线性变化 → NCC 天然不变；
* 加噪、偏置场、轻微重采样 → 中间层结构仍高度相关；
* 单体素平移 / 层序反转 → 平移搜索与正反次序都试一遍；
* 强度图之外还比较**梯度幅值图**（边缘结构），对非线性灰度调整更稳。

做法：固定 ``a`` 的中心窗口，与 ``b`` 的全部平移窗口一次性做相关（向量化，单对约 2 ms）。
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .preprocess import EPS, MIN_OVERLAP, VolumeDescriptor, DescriptorParams

EXACT_SHORTCUT = 0.995
_SHIFT_CACHE: dict[int, tuple[tuple[int, int], ...]] = {}


def shift_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """平移搜索窗口（行优先、确定性顺序）。"""
    cached = _SHIFT_CACHE.get(radius)
    if cached is not None:
        return cached
    offsets = tuple(
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    )
    _SHIFT_CACHE[radius] = offsets
    return offsets


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float32).ravel()
    right = np.asarray(b, dtype=np.float32).ravel()
    if left.size < MIN_OVERLAP or left.size != right.size:
        return -1.0
    left = left - float(left.mean())
    right = right - float(right.mean())
    denominator = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator <= EPS:
        return -1.0
    return float(np.dot(left, right) / denominator)


def overlap_ncc(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> float:
    """把 ``b`` 平移 ``(dy, dx)`` 后，在重叠区域上与 ``a`` 的归一化互相关。"""
    if dy == 0 and dx == 0:
        return pearson(a, b)
    height, width = a.shape
    top = max(0, dy)
    bottom = min(height, height + dy)
    left = max(0, dx)
    right = min(width, width + dx)
    if bottom - top < MIN_OVERLAP or right - left < MIN_OVERLAP:
        return -1.0
    return pearson(
        a[top:bottom, left:right],
        b[top - dy:bottom - dy, left - dx:right - dx],
    )


def gradient_magnitude(plane: np.ndarray) -> np.ndarray:
    """中心差分梯度幅值（边缘结构，对亮度/对比度调整不敏感）。"""
    array = np.asarray(plane, dtype=np.float32)
    gradient = np.zeros_like(array)
    if array.shape[0] > 2 and array.shape[1] > 2:
        gy = 0.5 * (array[2:, 1:-1] - array[:-2, 1:-1])
        gx = 0.5 * (array[1:-1, 2:] - array[1:-1, :-2])
        gradient[1:-1, 1:-1] = np.hypot(gx, gy)
    return gradient


def window_ncc(a: np.ndarray, b: np.ndarray, radius: int) -> float:
    """``a`` 的中心窗口 vs ``b`` 的全部平移窗口：一次性向量化求最大 NCC。"""
    height, width = a.shape
    margin = radius
    if height - 2 * margin < MIN_OVERLAP or width - 2 * margin < MIN_OVERLAP:
        margin = 0
    window = a[margin:height - margin, margin:width - margin]
    if margin == 0:
        windows = b[None, :, :]
    else:
        windows = np.stack(
            [
                b[margin + dy:height - margin + dy, margin + dx:width - margin + dx]
                for dy in range(-margin, margin + 1)
                for dx in range(-margin, margin + 1)
            ]
        ).astype(np.float32, copy=False)

    centered = window - float(window.mean())
    window_norm = float(np.linalg.norm(centered))
    flat = windows.reshape(windows.shape[0], -1).astype(np.float32, copy=False)
    flat = flat - flat.mean(axis=1, keepdims=True)
    denominator = np.maximum(window_norm * np.linalg.norm(flat, axis=1), EPS)
    correlations = (flat @ centered.ravel()) / denominator
    return float(correlations.max()) if correlations.size else -1.0


def slice_similarity(a: np.ndarray, b: np.ndarray, radius: int = 1) -> float:
    """单层相似度：强度图与梯度图在平移窗口内的最大相关。"""
    left = np.asarray(a, dtype=np.float32)
    right = np.asarray(b, dtype=np.float32)
    if left.shape != right.shape:
        return -1.0
    best = window_ncc(left, right, radius)
    if best < EXACT_SHORTCUT:            # 已几乎完全一致时无需再看梯度图
        best = max(
            best,
            window_ncc(gradient_magnitude(left), gradient_magnitude(right), radius),
        )
    return best


def slab_similarity(
    left: VolumeDescriptor,
    right: VolumeDescriptor,
    params: DescriptorParams | None = None,
) -> float:
    """两条序列的相似度：各层取最大，层序正/反都比一次，结果裁剪到 ``[0, 1]``。"""
    params = params or DescriptorParams()
    left_planes = np.asarray(left.fine, dtype=np.float32)
    right_planes = np.asarray(right.fine, dtype=np.float32)
    count = min(left_planes.shape[0], right_planes.shape[0])
    if count < 1:
        return 0.0
    best = -1.0
    for index in range(count):
        best = max(best, slice_similarity(left_planes[index], right_planes[index], params.shift))
        mirror = count - 1 - index
        if mirror != index:
            best = max(
                best,
                slice_similarity(left_planes[index], right_planes[mirror], params.shift),
            )
    return float(min(1.0, max(0.0, best)))


def study_similarity(
    left: Sequence[tuple[str, VolumeDescriptor]],
    right: Sequence[tuple[str, VolumeDescriptor]],
    params: DescriptorParams | None = None,
) -> tuple[float, str, str, bool]:
    """两个检查的相似度：所有序列对的最大值。

    返回 ``(similarity, left_series_uid, right_series_uid, exact)``；``exact=True``
    表示命中中间层像素 SHA1（同一份像素的拷贝），相似度直接取 1.0。
    """
    best = (-1.0, "", "", False)
    for left_uid, left_descriptor in left:
        for right_uid, right_descriptor in right:
            if left_descriptor.exact == right_descriptor.exact:
                return 1.0, str(left_uid), str(right_uid), True
            similarity = slab_similarity(left_descriptor, right_descriptor, params)
            if similarity > best[0]:
                best = (similarity, str(left_uid), str(right_uid), False)
    if best[0] < 0.0:
        return 0.0, "", "", False
    return best
