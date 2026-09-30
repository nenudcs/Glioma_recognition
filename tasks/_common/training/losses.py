"""多任务损失（共享训练工程）。

本工程一个骨干同时服务五个目标，因此损失是**四路加权和**：

    L = w_seg · (Dice + BCE)_seg          # Goal5：core / peri 两通道
      + w_ds  · Σ_mid (Dice + BCE)        # 深监督（加速收敛、稳定浅层）
      + w_cls · Σ_fields BCE/CE           # Goal3 肿瘤 + Goal4 十四个结构化字段
      + w_sp  · BCE_special               # Goal1 真实性 + Goal2-A 拼接
      + w_emb · 配对对比损失               # Goal2-B 重复影像

三点设计考虑（都来自实际训练中踩过的坑）：

1. **分割用 Dice+BCE 组合**：肿瘤体素极度不平衡，纯 BCE 会让模型倾向全预测背景；
   纯 Dice 在早期梯度不稳。组合既保召回又提供稳定梯度。
2. **分类损失必须 mask**：结构化金标准大量缺失（真实数据里 14 个字段往往只标了
   一部分），未标注字段必须从损失里剔除，否则模型会被"强制学 0"污染。
3. **special 头用 label_mask 分路监督**：fake 与 Composition 是两件事，
   一个任务的正样本不能当成另一个任务的负样本（否则头会互相抵消）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LossWeights:
    """各任务损失权重。可被各 Goal 的 ``losses.py`` 覆盖以做单目标微调。"""

    seg: float = 1.0
    ds: float = 0.4
    cls: float = 0.6
    special: float = 0.5
    embed: float = 0.3
    #: 边界加权系数：肿瘤边界体素的 BCE 权重（提升 HD95/边界质量）
    boundary_boost: float = 2.0


@dataclass
class LossBreakdown:
    """分项损失，便于监控"哪一路在退化"。

    ⚠️ ``total`` 必须是**保留计算图的 tensor**，不能用 float：
    若在累加时做 ``float(...)``，梯度链会断开，``total.backward()`` 直接报
    "float object has no attribute 'backward'"——训练根本跑不起来。
    ``parts`` 只用于日志，转成 float 没有副作用。
    """

    total: Any = None
    parts: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value, weight: float = 1.0) -> None:
        v = value * float(weight)                                # 保持 tensor，不 detach
        self.total = v if self.total is None else self.total + v
        # 日志用值：tensor 走 detach，纯 float 直接取（防御性，避免再次断链式写法）
        logged = float(v.detach()) if hasattr(v, "detach") else float(v)
        self.parts[name] = self.parts.get(name, 0.0) + logged

    def backward(self) -> None:
        """对总损失反向传播（无损失时静默跳过）。"""
        if self.total is not None:
            self.total.backward()


def soft_dice_loss(logits, target, eps: float = 1e-6):
    """逐通道 soft Dice（对 batch 与空间维聚合）。"""
    import torch

    p = torch.sigmoid(logits)
    dims = tuple(range(2, p.dim()))
    num = 2 * (p * target).sum(dims) + eps
    den = p.sum(dims) + target.sum(dims) + eps
    return (1 - num / den).mean()


def bce_with_boundary(logits, target, boost: float = 2.0):
    """带边界加权的 BCE。

    边界由 target 的形态学梯度近似（不引入额外依赖时用 max-pool 差分）。
    边界体素在总损失中占比很小却直接决定 HD95，因此显式提权。
    """
    import torch
    import torch.nn.functional as F

    w = torch.ones_like(target)
    if boost > 1.0:
        t = target.float()
        # 3D max-pool 近似膨胀，差分得到边界带
        dil = F.max_pool3d(t, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool3d(-t, kernel_size=3, stride=1, padding=1)
        edge = (dil - ero).clamp(0, 1)
        w = w + (boost - 1.0) * edge
    return F.binary_cross_entropy_with_logits(logits, target.float(), weight=w)


def seg_loss(logits, target, boost: float = 2.0):
    """分割损失：Dice + 边界加权 BCE。"""
    return soft_dice_loss(logits, target) + bce_with_boundary(logits, target, boost)


def classification_loss(logits_list, labels, label_mask, cls_spec):
    """多字段分类损失（按 label_mask 剔除未标注字段）。

    Args:
        logits_list: 每字段一个 tensor（binary → [B,1]；multi → [B,n]）
        labels:      [B, F] 目标值（binary 用 0/1；multi 用类别索引）
        label_mask:  [B, F] 1 表示该字段有标注
        cls_spec:    [(key, n_classes), ...]，n_classes==1 视为二分类
    """
    import torch
    import torch.nn.functional as F

    total = logits_list[0].new_zeros(())
    n_used = 0
    for i, (_key, n_cls) in enumerate(cls_spec):
        m = label_mask[:, i]
        if float(m.sum()) <= 0:
            continue                                             # 本批该字段全缺 → 跳过
        lg = logits_list[i]
        tgt = labels[:, i]
        if n_cls <= 1:
            per = F.binary_cross_entropy_with_logits(
                lg.squeeze(1), tgt.clamp(0, 1), reduction="none")
        else:
            per = F.cross_entropy(lg, tgt.long().clamp(0, n_cls - 1), reduction="none")
        total = total + (per * m).sum() / m.sum().clamp_min(1e-6)
        n_used += 1
    return total / max(1, n_used)


def special_loss(logits, target, label_mask):
    """Goal1/Goal2-A 的检查级二分类损失（按 label_mask 分路监督）。"""
    import torch
    import torch.nn.functional as F

    per = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    m = label_mask
    return (per * m).sum() / m.sum().clamp_min(1e-6)


def embed_pair_loss(emb_a, emb_b, pair_label, margin: float = 0.3):
    """重复影像配对损失：正对拉近、负对推远（cosine + margin）。

    规范指标是 AUC-PR / Recall@10%FPR，属**排序问题**，因此用对比式损失
    而不是二分类交叉熵——后者在极端不平衡（负对远多于正对）下会把
    所有对都压到低分，排序能力反而变差。
    """
    import torch
    import torch.nn.functional as F

    sim = F.cosine_similarity(emb_a, emb_b, dim=1)
    y = pair_label.float()
    pos = y * (1.0 - sim)
    neg = (1.0 - y) * F.relu(sim - margin)
    return (pos + neg).mean()


#: 已告警过的键（避免每个 step 刷屏）
_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    """对"监督信号缺失导致的静默失效"做一次性告警。

    下面几路损失都是 ``if batch.get(...) is not None`` 的形式，缺失时只是
    **安静地跳过**。后果是"以为在训练某个头、实际它从未收到梯度"：
    该头保留随机初始化，推理时照样输出一个 [0,1] 的概率，
    既不会报错也难以从日志察觉。一次性打印成本极低，却能立刻暴露它。
    """
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"[loss][告警] {msg}", flush=True)


def compute_losses(out: dict, batch: dict, cls_spec, weights: LossWeights) -> LossBreakdown:
    """把网络输出与 batch 组装成总损失（供训练循环直接调用）。"""
    bd = LossBreakdown()
    if out.get("seg") is not None and batch.get("target") is not None:
        tgt = batch["target"]
        if tgt.dim() == 4:
            tgt = tgt.unsqueeze(1)
        n_ch = out["seg"].shape[1]
        tgt = tgt[:, :n_ch]
        bd.add("seg", seg_loss(out["seg"], tgt, weights.boundary_boost), weights.seg)
    if out.get("ds") and batch.get("target") is not None:
        tgt = batch["target"]
        if tgt.dim() == 4:
            tgt = tgt.unsqueeze(1)
        # 注意：不能在这里做 float()——那会切断计算图，总损失无法反向传播
        ds = sum(seg_loss(d, _match_ds_target(tgt, d), weights.boundary_boost)
                 for d in out["ds"]) / max(1, len(out["ds"]))
        bd.add("ds", ds, weights.ds)
    if out.get("cls") and batch.get("labels") is not None:
        bd.add("cls", classification_loss(out["cls"], batch["labels"],
                                          batch["label_mask"], cls_spec), weights.cls)
    if out.get("special") is not None:
        if batch.get("special_target") is None:
            _warn_once(
                "special",
                "batch 缺 special_target → special 头（目标一 真实性 / 目标二-A 拼接）"
                "未参与本轮训练。若本次训练**需要**该头却没带上它的监督信号，"
                "它上线后会输出'看起来正常、实际随机'的概率。"
                "（仅做分割/结构化训练时该告警可忽略。）",
            )
        else:
            bd.add("special", special_loss(out["special"], batch["special_target"],
                                           batch["special_mask"]), weights.special)
    if batch.get("pair") is not None and out.get("embed_a") is not None:
        bd.add("embed", embed_pair_loss(out["embed_a"], out["embed_b"],
                                        batch["pair"]), weights.embed)
    elif out.get("embed") is not None:
        _warn_once(
            "embed",
            "batch 缺 pair → 嵌入头（目标二-B 重复影像）未参与本轮训练。"
            "配对监督需要配对数据集（同一检查的两次增强为正对）；"
            "若整轮都没有 pair，嵌入会停留在随机初始化状态，"
            "服务端算出的相似度将失去区分度。"
            "（仅做分割/结构化训练时该告警可忽略。）",
        )
    return bd


def _match_ds_target(tgt, logits):
    """把目标适配到深监督输出的形状：**先下采样空间尺寸，再裁通道**。

    深监督分支的分辨率是主输出的 1/2、1/4…，只裁通道会导致
    "The size of tensor a (8) must match the size of tensor b (16)" 这类形状错误。
    下采样用最近邻：分割目标是二值的，线性插值会造出 0.5 这种中间值。
    """
    import torch.nn.functional as F

    if tuple(tgt.shape[2:]) != tuple(logits.shape[2:]):
        tgt = F.interpolate(tgt, size=tuple(logits.shape[2:]), mode="nearest")
    return tgt[:, :logits.shape[1]] if tgt.shape[1] >= logits.shape[1] else tgt


def summarize(bd: LossBreakdown) -> dict[str, float]:
    """给日志用的紧凑字典（total 转成 float，不参与反向图）。"""
    total = float(bd.total.detach()) if bd.total is not None else 0.0
    return {"total": round(total, 4),
            **{k: round(v, 4) for k, v in sorted(bd.parts.items())}}


def _unused(_x):                                                 # pragma: no cover
    """保留 numpy 引用，避免 linter 误报未使用的导入。"""
    return np is not None
