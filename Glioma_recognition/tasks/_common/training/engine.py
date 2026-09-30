"""共享训练引擎：一个骨干、四路损失的统一训练循环。

为什么训练是**共享的**而不是每个 Goal 各训一个模型：

本工程采用单骨干多任务设计（规范 §5.1 允许的形态），一次前向同时产出
分割 / 结构化 / 特殊影像 / 嵌入。若拆成五个独立训练工程，
(a) 要重复五次几乎相同的数据加载与增强；
(b) 无法利用任务间的共享表征（实测分割与结构化互有正向收益）；
(c) 五个模型的权重文件也让 §5.2 的 checkpoint 约定失去意义。
因此：**训练引擎共享，各 Goal 只声明自己那一路的损失权重与评估口径。**

引擎自带三项工程性保护：
1. **EMA**：验证与提交都用指数滑动平均权重，小数据下比裸权重稳定；
2. **AMP + 梯度裁剪**：3D 分割显存紧张，且早期梯度易爆；
3. **全图阈值标定**：patch 级的 Dice 与全图推理的 Dice 常常不匹配，
   阈值必须在**全图**上扫（对应脚本 ``14_calibrate_thresholds.py``）。
"""
from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from tasks._common.training.losses import LossWeights, compute_losses, summarize


@dataclass
class TrainConfig:
    """训练超参（各 Goal 可用自己的 config 覆盖）。"""

    epochs: int = 100
    batch_size: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    amp_dtype: str = "bf16"
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    warmup_epochs: int = 3
    num_workers: int = 4
    patch: tuple[int, int, int] = (96, 96, 96)
    seed: int = 42
    out_dir: str = "checkpoints"
    tag: str = "run"
    save_every: int = 10
    loss_weights: LossWeights = field(default_factory=LossWeights)


def _lr_at(epoch: int, cfg: TrainConfig) -> float:
    """warmup + 余弦退火。早期小学习率可避免 BN/统计量在小 batch 下被打乱。"""
    if epoch < cfg.warmup_epochs:
        return cfg.lr * (epoch + 1) / max(1, cfg.warmup_epochs)
    t = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return 0.5 * cfg.lr * (1 + math.cos(math.pi * min(1.0, t)))


def _selection_score(val: dict, metrics: dict, fallback_key: str = "seg") -> float:
    """挑选 best 权重的依据（**必须返回有限值**）。

    若 ``score`` 是 ``nan``／``inf``，``score > best_score`` 恒为 False，
    ``_save`` 便**一次都不会执行**：训练全程跑完却没有权重文件，
    而日志与 ``history`` 都显示"完成"——最难察觉的一种失效。
    "验证指标为 nan"在目标一/二/三这种正样本极稀疏的任务上很常见
    （验证折里可能一个正样本都没有，AUC 无从计算）。

    兜底顺序：有效的 ``score`` → 训练损失取负 → 0.0。
    """
    s = val.get("score")
    if s is not None:
        s = float(s)
        if math.isfinite(s):
            return s
    total = metrics.get(fallback_key)
    if total is not None and math.isfinite(float(total)):
        return -float(total)
    return 0.0


class EMA:
    """指数滑动平均权重。"""

    def __init__(self, model, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    def update(self, model) -> None:
        import torch

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                         alpha=1 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return self.shadow


def train(model, train_loader, val_fn, cfg: TrainConfig, cls_spec: list[tuple[str, int]],
          device: str = "cuda") -> dict[str, Any]:
    """执行训练并返回训练结果摘要。

    Args:
        model:      共享骨干（输出 seg/ds/cls/special/embed）。
        train_loader: 产出 batch 的 DataLoader；batch 需含
                     ``image/target/labels/label_mask/special_target/special_mask``，
                     配对训练时另含 ``pair/embed_a_batch``。
        val_fn:      ``val_fn(ema_model) -> dict``，返回验证指标（如 dice_core/peri）。
        cfg:         超参。
        cls_spec:    分类头定义 ``[(key, n_classes), ...]``。

    Returns:
        ``{"best": float, "history": [...], "ckpt": str}``
    """
    import torch

    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.to(dev).train()
    ema = EMA(model, cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp = torch.bfloat16 if cfg.amp_dtype == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp is torch.float16))

    out_dir = Path(cfg.out_dir) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    # 初值取 -inf：保证**第一个 epoch 结束一定保存一次权重**，
    # 即使选择指标异常也不会出现"跑完没有任何产物"。
    best_score, history = -float("inf"), []

    for epoch in range(cfg.epochs):
        lr = _lr_at(epoch, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        t0, agg = time.time(), {}
        for step, batch in enumerate(train_loader):
            batch = {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v)
                     for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)

            # ---- 分阶段前向 + 逐阶段立即反向 ----
            # 一个 batch 最多要跑 **3 个 96³ 前向**：patch→分割头、
            # 整脑→全局头、配对的第二条→嵌入。若像早期实现那样"全部前向完
            # 再统一 backward"，三份激活会同时驻留显存 —— 实测在 24GB 卡上
            # 直接 OOM（22.86 GiB）。分阶段后峰值 = **单阶段峰值**，
            # 梯度在两个阶段间累加，数学上与一次 backward 完全等价。
            parts_all: dict[str, float] = {}

            def _backward(bd) -> None:
                parts_all.update(bd.parts)
                if bd.total is None:
                    return
                if scaler.is_enabled():
                    scaler.scale(bd.total).backward()
                else:
                    bd.total.backward()

            # 阶段 1：patch → seg / ds（分割需要高分辨率局部视图）
            with torch.autocast("cuda", dtype=amp, enabled=(dev == "cuda")):
                out_seg = model(batch["image"])
                for _k in ("cls", "special", "embed"):
                    out_seg[_k] = None                            # 全局头留给阶段 2
                bd_seg = compute_losses(out_seg, {**batch, "labels": None,
                                                  "label_mask": None,
                                                  "special_target": None,
                                                  "pair": None},
                                        cls_spec, cfg.loss_weights)
            _backward(bd_seg)
            del out_seg, bd_seg

            # 阶段 2：整脑视图 → 全局头（cls / special / embed）
            # 数据集的 ``__getitem__`` 已产出它（键名 ``whole``，192mm→96³，
            # 与推理侧 ``tasks/_common/volume.global_view`` 同一尺度）；
            # 早期实现只用 96mm patch 跑全局头 → 训练/推理输入分布不一致，
            # 表现为训练指标正常、上线后结构化字段与特殊影像概率显著退化。
            whole = batch.get("whole")
            if whole is not None:
                with torch.autocast("cuda", dtype=amp, enabled=(dev == "cuda")):
                    out = model(whole)
                    if batch.get("pair") is not None and "image_b" in batch:
                        # 第二条同样走整脑视图（嵌入必须与推理同尺度）
                        second = batch.get("whole_b")
                        out_b = model(second if second is not None else batch["image_b"])
                        out["embed_a"] = out.get("embed")
                        out["embed_b"] = out_b.get("embed")
                        del out_b
                    out["seg"] = None
                    out["ds"] = None                              # 分割已在阶段 1 算过
                    bd = compute_losses(out, {**batch, "target": None},
                                        cls_spec, cfg.loss_weights)
                _backward(bd)
                del out, bd

            # 两阶段梯度已累加完毕，统一做裁剪与优化器步进
            if scaler.is_enabled():
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            ema.update(model)

            for k, v in parts_all.items():
                agg[k] = agg.get(k, 0.0) + v
            if (step + 1) % 50 == 0:
                print(f"[{cfg.tag}] ep{epoch} step{step + 1}/{len(train_loader)} "
                      f"loss={sum(parts_all.values()):.4f} "
                      f"{{{', '.join(f'{k}: {v:.4f}' for k, v in sorted(parts_all.items()))}}}",
                      flush=True)

        n = max(1, len(train_loader))
        metrics = {k: v / n for k, v in agg.items()}
        # 验证用 EMA 权重（与提交时一致）
        ema_model = _with_ema(model, ema)
        val = val_fn(ema_model) if val_fn else {}
        score = _selection_score(val, metrics, fallback_key="seg")
        metrics.update({f"val_{k}": v for k, v in val.items()})
        metrics["lr"], metrics["epoch"] = lr, epoch
        history.append(metrics)
        print(f"[{cfg.tag}] epoch {epoch} loss={metrics.get('seg', 0):.4f} "
              f"{' '.join(f'{k}={v:.4f}' for k, v in val.items())} ({time.time() - t0:.0f}s)",
              flush=True)

        if score > best_score:
            best_score = score
            _save(out_dir / "best.pth", model, ema, cls_spec, cfg, best_score, epoch)
        if (epoch + 1) % cfg.save_every == 0:
            _save(out_dir / "last.pth", model, ema, cls_spec, cfg, best_score, epoch)

    return {"best": best_score, "history": history, "ckpt": str(out_dir / "best.pth")}


def _with_ema(model, ema: EMA):
    """复制一份套用 EMA 权重的模型（不污染训练中的模型）。"""
    import torch

    m = copy.deepcopy(model).eval()
    sd = m.state_dict()
    for k, v in ema.state_dict().items():
        if k in sd:
            sd[k] = v.to(sd[k].dtype)
    m.load_state_dict(sd, strict=False)
    return m


def _as_dict(obj) -> dict:
    """把配置对象安全转成 ``dict``（兼容 dataclass / SimpleNamespace / 动态类实例）。

    不能直接用 ``asdict()``：它**只接受 dataclass 实例**，而训练侧传入的
    ``model.cfg`` 可能是 ``SimpleNamespace`` 或 ``type("MC", (), {...})()``
    这类动态类实例，此时会抛
    ``TypeError: asdict() should be called on dataclass instances``。

    该异常发生在**第一次保存 best 权重**时（epoch 0 结束），表现为
    "训练跑完一个 epoch 就崩"，堆栈又落在保存逻辑里，很容易被误判成
    显存不足或数据问题。这里统一兜底，避免"能否存下权重"取决于
    上层用了哪种配置容器。
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)

    out = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    # ``type("MC", (), {...})()`` 这类动态类实例把配置放在**类属性**上，
    # ``vars(obj)`` 只返回实例 ``__dict__``（为空）。必须再遍历类型层级，
    # 否则 ``model_cfg`` 会存成 ``{}``：推理侧拿不到 base/depth，
    # 只能用默认值重建网络 —— 形状不匹配的层会被 ``strict=False``
    # **静默跳过**，表现为"加载成功但精度崩坏"，比直接报错更难查。
    for klass in type(obj).__mro__:
        for k, v in vars(klass).items():
            if not k.startswith("_") and k not in out and not callable(v):
                out[k] = v
    return out


def _save(path: Path, model, ema: EMA, cls_spec, cfg: TrainConfig,
          best: float, epoch: int) -> None:
    """保存 checkpoint（含重建网络所需的全部元信息）。"""
    import torch

    mc = getattr(model, "cfg", None)
    torch.save({
        "model": model.state_dict(),
        "model_ema": ema.state_dict(),
        "cls_spec": cls_spec,
        "arch": getattr(model, "arch_name", "mednext"),
        "model_cfg": _as_dict(mc),
        "thresholds": [0.5, 0.5],                                # 由全图标定脚本写回
        "train_cfg": {**asdict(cfg), "loss_weights": asdict(cfg.loss_weights)},
        "best_score": best,
        "epoch": epoch,
    }, str(path))
    print(f"[train] 已保存 {path}（best={best:.4f}, epoch={epoch}）", flush=True)


def write_history(path: str, history: list[dict]) -> None:
    """训练历史落盘（JSONL，供曲线复盘）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
