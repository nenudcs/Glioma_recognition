"""统一训练命令行入口。

各 Goal 的 ``train.py`` 只是本入口的薄封装，保证：

    python -m tasks.goal1_authenticity.train --fold 0 --tag g4_fold0
    python -m tasks.goal5_segmentation.train  --fold 0 --tag g4_fold0

都走**同一个**训练引擎与数据管线。这样做的理由：本工程是一个多任务骨干，
一次训练同时优化所有头；若各 Goal 各有一套 ``train.py`` 实现，
"五个目标其实是一个模型"这个事实就会被代码结构掩盖，队友很容易误以为
需要分别训练，从而浪费大量算力。

用法::

    python -m tasks.<goal>.train --fold 0 [--tag NAME] [--epochs N] [--config CFG]

配置文件为 YAML，字段与 :class:`TrainConfig` 对应；未提供的取默认值。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: 各 Goal 的默认损失权重覆盖。共享引擎默认给"全任务均衡"，
#: 这里允许某个 Goal 单独调高自己那一路（用于单目标微调）。
GOAL_LOSS_OVERRIDES: dict[str, dict[str, float]] = {
    "goal1_authenticity": {"special": 1.0, "seg": 0.3},
    "goal2_stitched": {"special": 1.0, "seg": 0.3},
    "goal2_duplicate": {"embed": 1.0},
    "goal3_tumor": {"cls": 1.0},
    "goal4_diagnosis": {"cls": 1.5},
    "goal5_segmentation": {"seg": 1.0, "ds": 0.4},
}


def _load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_argparser(goal: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"训练 {goal}（共享多任务骨干）")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default=None, help="输出目录名，默认 <goal>_fold<N>")
    ap.add_argument("--config", default=None, help="YAML 配置路径")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out-dir", default=None, help="checkpoint 输出根目录")
    ap.add_argument("--patch", type=int, nargs=3, default=None,
                    help="训练 patch 尺寸，如 --patch 96 96 96")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量（调试用）")
    return ap


def run_goal(goal: str, argv: list[str] | None = None) -> int:
    """执行某个 Goal 的训练（内部仍训练整个共享骨干）。"""
    from tasks._common.training.helpers import (
        build_dataloaders,
        build_model_for_training,
        build_val_fn,
    )

    args = build_argparser(goal).parse_args(argv)
    raw = _load_yaml(args.config)
    cfg_kwargs = {k: v for k, v in raw.items() if k in _CONFIG_FIELDS}
    if args.epochs is not None:
        cfg_kwargs["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg_kwargs["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg_kwargs["lr"] = args.lr
    if args.patch:
        cfg_kwargs["patch"] = tuple(args.patch)

    from tasks._common.training.engine import TrainConfig, train, write_history
    from tasks._common.training.losses import LossWeights

    lw = dict(LossWeights().__dict__)
    lw.update((raw.get("loss_weights") or {}))
    lw.update(GOAL_LOSS_OVERRIDES.get(goal, {}))
    cfg = TrainConfig(**cfg_kwargs, loss_weights=LossWeights(**lw),
                      tag=args.tag or f"{goal}_fold{args.fold}",
                      out_dir=args.out_dir or os.environ.get("CKPT_DIR", "checkpoints"))

    print(f"[train] goal={goal} fold={args.fold} tag={cfg.tag} "
          f"epochs={cfg.epochs} lr={cfg.lr} patch={cfg.patch}", flush=True)
    print(f"[train] 损失权重: {lw}", flush=True)

    model, cls_spec = build_model_for_training(raw)
    train_loader, val_loader = build_dataloaders(cfg, fold=args.fold, raw=raw,
                                                limit=args.limit)
    val_fn = build_val_fn(goal, val_loader, raw)

    res = train(model, train_loader, val_fn, cfg, cls_spec, device=args.device)
    write_history(str(Path(cfg.out_dir) / cfg.tag / "history.jsonl"), res["history"])
    print(f"[train] 完成 best={res['best']:.4f} -> {res['ckpt']}", flush=True)
    print(json.dumps({"best": res["best"], "ckpt": res["ckpt"]}, ensure_ascii=False))
    return 0


def run_eval(goal: str, argv: list[str] | None = None) -> int:
    """评估某个 Goal 在指定折验证集上的指标。

    评估口径与训练时的验证**完全一致**（复用同一个 ``build_val_fn``），
    避免"训练时报一个数、评估时报另一个数"这种最难对齐的偏差。

    ⚠️ 这是**轻量、patch 级**的离线评估；面向比赛指标的全图/留一折集成评估
    请用 ``glioma_track4`` 的 ``scripts/16_finalize.sh``（那里是权威口径）。
    """
    import argparse

    from tasks._common.training.helpers import (
        build_datasets,
        build_model_for_training,
        build_val_fn,
    )

    ap = argparse.ArgumentParser(description=f"评估 {goal}（共享多任务骨干）")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default=None, help="默认 <goal>_fold<N>")
    ap.add_argument("--ckpt", default=None,
                    help="权重路径；默认 <out-dir>/<tag>/best.pth")
    ap.add_argument("--out-dir", default=None, help="训练输出根目录（默认 $CKPT_DIR 或 checkpoints）")
    ap.add_argument("--config", default=None, help="YAML 配置（与训练保持一致）")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 例（0 = 全量）")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import torch

    raw = _load_yaml(a.config)
    tag = a.tag or f"{goal}_fold{a.fold}"
    out_dir = Path(a.out_dir or os.environ.get("CKPT_DIR", "checkpoints")) / tag
    ckpt = Path(a.ckpt) if a.ckpt else out_dir / "best.pth"
    if not ckpt.is_file():
        raise SystemExit(
            f"[eval] 找不到权重：{ckpt}\n"
            f"       先训练（python -m tasks.{goal}.train --fold {a.fold}），"
            f"或用 --ckpt 指定路径"
        )

    model, cls_spec = build_model_for_training(raw)
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = ck.get("model_ema") or ck.get("model") or ck
    missing, _ = model.load_state_dict(state, strict=False)
    # 与推理侧同样的判断：骨干缺层必须**明确失败**，否则会静默加载随机层
    core = [k for k in missing if k.startswith(("enc", "dec", "bottleneck", "stem"))]
    if core:
        raise SystemExit(
            f"[eval] 骨干缺失 {len(core)} 层（{core[:3]}…），"
            f"权重与当前配置的结构不匹配"
        )

    dev = a.device if (a.device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.eval().to(dev)

    tr_cfg = raw.get("train") or {}
    patch = tuple(tr_cfg.get("patch", [96, 96, 96]))
    _, val_ds = build_datasets(raw, fold=a.fold, limit=a.limit,
                               seed=int(tr_cfg.get("seed", 42)), patch=patch)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=max(1, a.batch_size), shuffle=False,
        num_workers=max(0, a.num_workers),
    )

    with torch.inference_mode():
        metrics = build_val_fn(goal, val_loader, raw)(model)

    print(f"[eval] goal={goal} fold={a.fold}")
    print(f"[eval] ckpt={ckpt}")
    print(f"[eval] n_val={len(val_ds)}  cls_spec={len(cls_spec)} 项")
    for k, v in sorted((metrics or {}).items()):
        print(f"       {k} = {v}")
    if not metrics:
        print("       （本 Goal 未定义验证指标；请用 16_finalize.sh 做权威评估）")
    return 0


_CONFIG_FIELDS = {
    "epochs", "batch_size", "lr", "weight_decay", "amp_dtype", "grad_clip",
    "ema_decay", "warmup_epochs", "num_workers", "patch", "seed", "save_every",
}
