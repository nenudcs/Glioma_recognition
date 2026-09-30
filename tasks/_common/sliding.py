"""滑窗推理 + TTA + 多折集成（显存友好、全局头保上下文）。

两路设计（关键）：
- **分割**：逐 patch 滑窗（高斯权重叠加），峰值显存 = TTA 批大小 × 1 个 patch，
  可处理任意大视野；
- **全局头**（结构化分类 / 重复影像嵌入 / 特殊影像目标一二）需要**全脑上下文**，
  因此把整个体积降采样到 ``global_size`` 后一次前向，避免"按 patch 平均"造成语义失真。

TTA 语义（**重要，曾存在双重翻折 bug**）：
``_forward_batch`` 内部完成"输入翻转 → 前向 → 输出翻回原方向"，
返回值**始终位于原方向**，调用方不得再次翻转。

集成：多折模型概率平均（分割取均值、分类取均值、嵌入取均值后重新 L2 归一化）。
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
import torch.nn.functional as F

_FLIP_DIM = {"x": 2, "y": 3, "z": 4}


def _gaussian_kernel(patch: tuple) -> torch.Tensor:
    axes = []
    for s in patch:
        x = torch.arange(s, dtype=torch.float32) - (s - 1) / 2.0
        sigma = max(1.0, s / 8.0)
        w = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        axes.append(torch.clamp(w / w.max(), 1e-3, 1.0))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def _windows(shape: tuple, patch: tuple, overlap: float) -> list[tuple[int, int, int]]:
    step = [max(1, int(round(p * (1 - overlap)))) for p in patch]
    grids = []
    for sz, ps, st in zip(shape, patch, step):
        if sz <= ps:
            grids.append([0])
        else:
            g = list(range(0, sz - ps + 1, st))
            if g[-1] != sz - ps:
                g.append(sz - ps)
            grids.append(g)
    return list(itertools.product(*grids))


def _tta_combos(axes: tuple) -> list[tuple]:
    """TTA 组合：(空) + 每个轴单独翻转 + 全部翻转（若 axes ≥ 3 则含两两组合的一半）。"""
    combos: list[tuple] = [()]
    for ax in axes:
        combos.append((ax,))
    if len(axes) >= 2:
        for a, b in itertools.combinations(axes, 2):
            combos.append((a, b))
    if len(axes) >= 3:
        combos.append(tuple(axes))
    return combos


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@torch.inference_mode()
def _forward_batch(models: list, x: torch.Tensor, combos: list[tuple], amp_dtype,
                   tta_batch: int = 1) -> dict:
    """对同一输入按 TTA 组合分组前向；返回**原方向**的 seg/cls/special/embed 均值。

    x: ``[1,C,D,H,W]``（未翻转）。所有返回值均为 TTA + 多模型平均后的结果：
    - seg   : ``[1,2,D,H,W]`` 概率（sigmoid）
    - cls   : list[np.ndarray]，每个字段概率
    - special: ``[2]`` 概率（假人体 / 拼接）
    - embed : ``[E]`` L2 归一化嵌入
    """
    segs: list[torch.Tensor] = []
    specs: list[torch.Tensor] = []
    embs: list[torch.Tensor] = []
    cls_holder: list[list[torch.Tensor]] = []
    for cg in _chunks(combos, max(1, int(tta_batch))):
        xb = torch.cat([_flip(x, c) for c in cg], dim=0)          # [B,C,D,H,W]
        for m in models:
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=xb.is_cuda and amp_dtype is not None):
                out = m(xb)
            seg = torch.sigmoid(out["seg"].float())
            for i, combo in enumerate(cg):
                segs.append(_flip(seg[i:i + 1], combo, back=True))
            specs.append(torch.sigmoid(out["special"].float()))
            embs.append(F.normalize(out["embed"].float(), dim=1))
            cls_now = [torch.softmax(c.float(), 1) if c.shape[1] > 1 else torch.sigmoid(c.float())
                       for c in out["cls"]]
            for i, c in enumerate(cls_now):                       # [B, n]
                while i >= len(cls_holder):
                    cls_holder.append([])
                cls_holder[i].append(c)

    seg = torch.stack(segs).mean(0)
    spec = torch.cat(specs, dim=0).mean(0)                        # 权重 = TTA×模型 均等
    emb = F.normalize(torch.cat(embs, dim=0).mean(0), dim=0)
    cls = [torch.cat(v, dim=0).mean(0) for v in cls_holder]
    return {"seg": seg, "cls": cls, "special": spec, "embed": emb}


def _flip(x: torch.Tensor, combo: tuple, back: bool = False) -> torch.Tensor:
    if back:
        for ax in reversed(combo):
            x = torch.flip(x, dims=[_FLIP_DIM[ax]])
        return x
    for ax in combo:
        x = torch.flip(x, dims=[_FLIP_DIM[ax]])
    return x


@torch.inference_mode()
def predict_volume(models: list, vol: np.ndarray, patch=(96, 96, 96), overlap: float = 0.5,
                   tta_flips: tuple = (), amp_dtype=torch.bfloat16, global_size: int = 96,
                   tta_batch: int = 1, seg_tta_flips: tuple | None = None,
                   device: str | None = None, global_vol: np.ndarray | None = None,
                   crop_brain: bool = True, brain_margin: int = 4) -> dict:
    """``vol: [C,D,H,W] float32`` → ``{seg_prob[2,D,H,W], cls, special, embed}``。

    ``global_vol``：**固定物理尺寸整脑视图**（``[C,G,G,G]``，训练与推理同尺度）。
    为空时退化为"整脑降采样"，但会与训练分布不一致，仅作兜底。

    ``seg_tta_flips``：分割滑窗使用的 TTA 轴（None 时用 ``tta_flips``）。
    分割是耗时主体，可只开 1~2 个轴；全局头很便宜，用满 ``tta_flips``。
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.from_numpy(np.ascontiguousarray(vol))[None].to(dev, torch.float32)
    _c, d, h, w = x0.shape[1:]
    pd, ph, pw = patch
    combos_all = _tta_combos(tuple(tta_flips))
    combos_seg = _tta_combos(tuple(seg_tta_flips if seg_tta_flips is not None else tta_flips))

    # ---- 0) 脑部裁剪（显著减少滑窗数量；结果再贴回原尺寸）----
    off = np.zeros(3, dtype=int)
    if crop_brain:
        mask = (x0[0].abs().sum(0) > 1e-3).cpu().numpy()
        if mask.any():
            idx = np.argwhere(mask)
            lo = np.maximum(idx.min(0) - brain_margin, 0)
            hi = np.minimum(idx.max(0) + brain_margin + 1, np.array([d, h, w]))
            off = lo
            x0 = x0[:, :, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            _c, d, h, w = x0.shape[1:]

    # ---- 1) 分割：逐 patch 滑窗（必要处 pad）----
    d2, h2, w2 = max(d, pd), max(h, ph), max(w, pw)
    x = F.pad(x0, (0, w2 - w, 0, h2 - h, 0, d2 - d)) if (d2, h2, w2) != (d, h, w) else x0
    acc = torch.zeros((1, 2, d2, h2, w2), device=dev, dtype=torch.float32)
    wacc = torch.zeros((1, 1, d2, h2, w2), device=dev, dtype=torch.float32)
    k = _gaussian_kernel(tuple(patch)).to(dev)[None, None]
    for (z, y, xx) in _windows((d2, h2, w2), tuple(patch), overlap):
        sub = x[:, :, z:z + pd, y:y + ph, xx:xx + pw]
        res = _forward_batch(models, sub, combos_seg, amp_dtype, tta_batch=tta_batch)
        seg = res["seg"]                                          # 已在原方向
        acc[:, :, z:z + pd, y:y + ph, xx:xx + pw] += seg * k
        wacc[:, :, z:z + pd, y:y + ph, xx:xx + pw] += k
    seg_roi = (acc / wacc.clamp_min(1e-6))[0, :, :d, :h, :w].float()
    seg_prob = torch.zeros((2, vol.shape[1], vol.shape[2], vol.shape[3]), device=dev)
    seg_prob[:, off[0]:off[0] + d, off[1]:off[1] + h, off[2]:off[2] + w] = seg_roi
    seg_prob = seg_prob.cpu().numpy()

    # ---- 2) 全局头：整脑固定物理尺寸视图一次前向（保上下文且与训练同尺度）----
    if global_vol is not None:
        g = torch.from_numpy(np.ascontiguousarray(global_vol))[None].to(dev, torch.float32)
    else:
        g = F.interpolate(x0, size=(global_size, global_size, global_size),
                          mode="trilinear", align_corners=False)
    res = _forward_batch(models, g, combos_all, amp_dtype, tta_batch=tta_batch)
    return {
        "seg": seg_prob,
        "cls": [c.cpu().numpy() for c in res["cls"]],
        "special": res["special"].cpu().numpy(),
        "embed": res["embed"].cpu().numpy(),
    }
