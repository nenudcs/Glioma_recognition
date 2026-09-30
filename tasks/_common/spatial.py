"""共享的空间变换与逆变换：公共网格 ↔ 原始序列空间（各 Goal 通用）。

规范 §9.6/§5.1：插件必须自己完成逆变换，把掩膜恢复到**目标原图空间**；
Writer 只按正式路径写文件，Validator 负责重读校验。

实现要点：全部通过 **world (RAS) 坐标**桥接，不做任何轴序假设。
直接按体素索引搬运（或假定 XYZ↔ZYX）在非 RAS 存储（LPS、z 翻转）下会**静默错位**，
而这类错误不会报异常，只会让分割指标悄悄变差。
"""
from __future__ import annotations

import numpy as np


def spacing_of(affine: np.ndarray) -> tuple[float, float, float]:
    """从仿射矩阵取出体素尺寸（mm）。"""
    a = np.asarray(affine, dtype=np.float64)
    return tuple(float(np.linalg.norm(a[:3, i])) for i in range(3))


def target_grid(ref_shape: tuple[int, int, int], ref_affine: np.ndarray,
                want: tuple[float, float, float] = (1.0, 1.0, 1.0),
                max_factor: float = 4.0):
    """以参考序列的物理视野为基准，给出目标公共网格 ``(shape, affine)``。

    - 层厚过大的轴**保持原样**：把 5mm 层厚强行插值到 1mm 只会产生虚假细节；
    - 已在目标网格时原样返回，调用方可据此跳过重采样。
    """
    a = np.asarray(ref_affine, dtype=np.float64)
    sp = np.array(spacing_of(a), dtype=np.float64)
    sp[sp <= 0] = 1.0
    tgt = np.where(sp <= max_factor * np.asarray(want, dtype=np.float64),
                   np.asarray(want, dtype=np.float64), sp)
    if np.allclose(tgt, sp, rtol=1e-3, atol=1e-4):
        return tuple(int(x) for x in ref_shape), a
    out_shape = tuple(int(np.ceil(s * f / t)) for s, f, t in zip(ref_shape, sp, tgt))
    out_affine = a.copy()
    for i in range(3):
        out_affine[:3, i] = a[:3, i] / tgt[i] * sp[i]
    return out_shape, out_affine


def resample_to(arr: np.ndarray, src_affine: np.ndarray,
                dst_shape: tuple[int, int, int], dst_affine: np.ndarray,
                order: int = 1, cval: float = 0.0) -> np.ndarray:
    """把 ``arr`` 从其自身空间重采样到目标空间（经世界坐标，任意轴序安全）。"""
    from scipy import ndimage

    a_src = np.asarray(src_affine, dtype=np.float64)
    a_dst = np.asarray(dst_affine, dtype=np.float64)
    if arr.shape == tuple(dst_shape) and np.allclose(a_src, a_dst):
        return arr
    m = np.linalg.inv(a_src) @ a_dst          # dst voxel → src voxel
    return ndimage.affine_transform(arr, m[:3, :3], offset=m[:3, 3],
                                    output_shape=tuple(dst_shape), order=order, cval=cval)


def restore_binary_to_source(mask: np.ndarray, common_affine: np.ndarray,
                             source_affine: np.ndarray,
                             source_shape: tuple[int, int, int]) -> np.ndarray:
    """把公共网格上的二值掩膜恢复到源序列空间。

    **用最近邻（order=0）并重新二值化**：规范要求掩膜体素严格为 ``{0, 1}``，
    线性插值会产生中间灰度值，导致该例分割直接记 0 分。
    """
    out = resample_to(np.asarray(mask).astype(np.float32), common_affine,
                      tuple(int(x) for x in source_shape), source_affine, order=0)
    return (out > 0.5).astype(np.uint8)
