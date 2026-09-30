"""Goal5 的后处理：概率图 → 干净的二值掩膜。

规范 §5.1：``postprocess.py`` 属于 **[运行]** 交付文件。

为什么必须做后处理（而不只是阈值化）：
1. **孤立小斑点**：滑窗拼接边缘常出现几个体素的假阳性，直接提交会拉低
   Precision 与 HD95；
2. **多连通碎片**：真实病灶在低阈值下常碎成多块，需要保留主要成分并做
   形态学桥接，否则 Dice 会明显低于目视结果；
3. **边界层噪声**：脑外/颅骨附近的假阳性会同时污染两个掩膜。

流程：阈值化 → （可选）形态学桥接 → 按体积排序保留主要连通域 → 最小体素过滤。
为可复现，全部为确定性操作，不使用任何随机量。
"""
from __future__ import annotations

import numpy as np


def _largest_components(mask: np.ndarray, keep_n: int) -> np.ndarray:
    """保留体积最大的前 ``keep_n`` 个连通域。"""
    from scipy import ndimage

    if not mask.any():
        return mask
    lab, n = ndimage.label(mask)
    if n <= keep_n:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    order = np.argsort(sizes)[::-1][:keep_n]
    keep = np.zeros(n + 1, dtype=bool)
    keep[order + 1] = True
    return keep[lab]


def _bridge(mask: np.ndarray, radius_mm: float, spacing: tuple[float, float, float]) -> np.ndarray:
    """形态学闭运算桥接邻近碎片（半径按 mm 给定，各向同性）。

    ⚠️ **这里不能用 ``ndimage.binary_closing(structure=<稠密立方结构元>)``。**

    那正是"修完空掩码之后才开始 OOM"的根因：``clean_mask`` 对**空**掩膜会在
    阈值化之后的 ``if not m.any(): return`` 就返回，**根本走不到本函数**；
    空掩膜修好之后掩膜变非空，`_bridge` 才第一次真正执行 ——

    实测（``192x192x120``，4.4 M 体素；真实 1mm 脑 ``240x240x155`` 约是它的 2 倍大）：

    ==================  ==================  ============
    实现                 峰值内存增量        单次耗时
    ==================  ==================  ============
    21³ 稠密结构元       **+693 MB**         **15.95 s**
    EDT 球半径（本实现）    +239 MB             0.52 s
    ==================  ==================  ============

    而 ``clean_pair`` 会调用它**两次**（core + flair），且 ``binary_closing`` 的
    内存/耗时随体积**超线性**增长 —— 真实体积下每次约 1.4 GB / 数十秒，
    两次就是数 GB 的瞬时峰值，容器直接被 OOM-kill。

    改用**欧氏距离变换**实现同样语义的闭运算：闭运算 = 补集的膨胀再取补，
    用 ``distance_transform_edt`` 表达即

    ``closed = edt(dilate(mask)) > r``，其中
    ``dilate(mask) = edt(~mask) <= r``。

    内存 O(体积)（两个定长距离场），耗时也是 O(体积)（精确 EDT，非暴力卷积）。
    附带一个正确性收益：稠密立方结构元是**各向异性**的（在 3mm 层厚轴上半径被
    放大 3 倍），距离变换按 ``spacing`` 取样得到的是**各向同性球**，与
    "``bridge_mm`` 按 mm 给定"的语义一致。实测两者结果 IoU ≈ 0.94~1.00。
    """
    from scipy import ndimage

    if radius_mm <= 0 or not mask.any():
        return mask
    # spacing 里出现 0/负值会污染距离变换，兜底成 1mm
    samp = np.asarray(spacing, dtype=np.float64)
    samp = np.where(samp > 0, samp, 1.0)
    r = float(radius_mm)

    # ---- 只算掩膜的包围盒外扩 r（**这一条是内存的关键**）----
    # 闭运算不会改变"距掩膜 > r"的体素（那些体素恒为背景），
    # 所以窗口外的结果就是输入本身。脑体积 8.9 M 体素，而瘤体包围盒通常只有
    # 几十万 —— 距离变换的规模因此降一个数量级，与整脑大小基本无关。
    def _extent(a: np.ndarray) -> tuple[int, int]:
        nz = np.flatnonzero(a)
        return int(nz[0]), int(nz[-1]) + 1

    vox_r = np.maximum(1, np.ceil(r / samp).astype(int))          # 各轴半径（体素）
    bz, by, bx = _extent(mask.any(axis=(1, 2))), _extent(mask.any(axis=(0, 2))), \
        _extent(mask.any(axis=(0, 1)))
    lo = (max(bz[0] - int(vox_r[0]), 0), max(by[0] - int(vox_r[1]), 0),
          max(bx[0] - int(vox_r[2]), 0))
    hi = (min(bz[1] + int(vox_r[0]), mask.shape[0]),
          min(by[1] + int(vox_r[1]), mask.shape[1]),
          min(bx[1] + int(vox_r[2]), mask.shape[2]))
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    if sub.shape == mask.shape:                                   # 包围盒已覆盖全图 → 直接算
        out = np.empty_like(mask)
    else:
        out = mask.copy()                                         # 窗口外保持原样

    # 到最近前景体素的欧氏距离 → 半径 r 内的背景被填进膨胀结果
    dilated = ndimage.distance_transform_edt(~sub, sampling=samp) <= r
    # 再做一次腐蚀（= 补集的膨胀），即得闭运算
    out[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (
        ndimage.distance_transform_edt(dilated, sampling=samp) > r)
    return out


def clean_mask(prob: np.ndarray, threshold: float, min_voxels: int,
               spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
               keep_components: int = 3, bridge_mm: float = 0.0) -> np.ndarray:
    """单通道：概率图 → 二值掩膜（确定性）。"""
    m = np.asarray(prob) > float(threshold)
    if not m.any():
        return m.astype(bool)
    if bridge_mm > 0:
        m = _bridge(m, bridge_mm, spacing)
    m = _largest_components(m, max(1, int(keep_components)))
    return m if int(m.sum()) >= int(min_voxels) else np.zeros_like(m, dtype=bool)


def clean_pair(core_prob: np.ndarray, flair_prob: np.ndarray, cfg,
               spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
               ) -> tuple[np.ndarray, np.ndarray]:
    """双通道联合后处理。

    任务语义上 **core ⊆ peri**（增强核心区是周围总异常区的一部分）。
    若模型给出的 core 越出 peri，说明边界处的两通道不一致——这里用
    并集兜底，避免出现"核心区在异常区之外"这种物理上不可能的答案。

    ``spacing``：**公共网格的实际体素尺寸（mm）**，必须由调用方传入。

    ⚠️ 此前这里**没有传** ``spacing``，于是 ``bridge_mm=10.0`` 被按
    ``rad = round(10 / 1.0) = 10`` 换算：结构元恒为 **21×21×21**。
    但公共网格并不总是 1mm 各向同性 —— ``build_volume`` 对层厚 > 1.5mm 的轴
    **保持原始 spacing**（与训练侧一致）。一条 3mm 层厚的序列，10mm 桥接半径
    本该是 3 体素，却被当成 10 体素 → **实际桥接 30mm**：
    形态学闭运算会把远离病灶的假阳性斑点与病灶连成一体，直接拉低 Precision 与
    HD95（不会让掩膜变空，因此这个 bug 不报错、只悄悄掉分）。
    """
    core = clean_mask(core_prob, cfg.default_thresholds[0], cfg.min_tumor_voxels,
                      spacing=spacing,
                      keep_components=cfg.keep_components, bridge_mm=cfg.bridge_mm)
    peri = clean_mask(flair_prob, cfg.default_thresholds[1], cfg.min_tumor_voxels,
                      spacing=spacing,
                      keep_components=cfg.keep_components, bridge_mm=cfg.bridge_mm)
    if core.any() and peri.any():
        peri = np.logical_or(peri, core)
    return core, peri
