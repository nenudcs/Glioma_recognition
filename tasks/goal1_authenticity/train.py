"""[研发] 目标一训练入口（规范 §5.1；比赛运行入口不得导入本文件）。

从零训练（默认不加载任何预训练权重），产出写到 ``--out-dir``：

```text
<out-dir>/best.pt         # 验证集检查级 AP 最优权重
<out-dir>/last.pt
<out-dir>/manifest.jsonl  # 本次使用的数据与划分（可复现）
<out-dir>/summary.json
<log-dir>/training.jsonl  # 赛方规范 JSONL 训练日志
```

权重规范位置由 ``migrate_checkpoint.py`` 负责迁移：

```bash
python -m tasks.goal1_authenticity.migrate_checkpoint --source <out-dir>/best.pt --also-last
```
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dataset import (
    SPECIAL_POLICIES,
    OfficialSlices,
    SliceConfig,
    assign_splits,
    collate,
    discover_records,
    find_annotation_root,
    read_manifest,
    summarize,
    write_manifest,
)
from .evaluate import clean_report, metrics_report
from .losses import build_criterion, positive_weight

DEFAULT_WORKSPACE = Path("/2026aicompetition/workspace")


def utc_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def default_data_root() -> Path:
    return Path(os.environ.get("GOAL1_DATA_ROOT", "/2026aicompetition/datasets/training")).expanduser()


def default_run_dir() -> Path:
    raw = os.environ.get("GOAL1_RUN_DIR")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    workspace = Path(os.environ.get("COMPETITION_WORKSPACE", DEFAULT_WORKSPACE)).expanduser()
    return workspace / "goal1_runs"


def default_log_dir() -> Path:
    raw = os.environ.get("GOAL1_LOG_DIR")
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    workspace = Path(os.environ.get("COMPETITION_WORKSPACE", DEFAULT_WORKSPACE)).expanduser()
    return workspace / "logs"


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    atexit.register(handle.close)          # 即使中途异常也关闭句柄（Windows 会锁文件）

    def log(**fields) -> None:
        handle.write(json.dumps(fields, ensure_ascii=False) + "\n")
        handle.flush()

    return log, handle


def topk_mean(logits, k: int):
    k = max(1, min(k, logits.shape[1]))
    return logits.topk(k, dim=1).values.mean(dim=1)


def forward_cases(model, slices):
    """``(B, K, 3, H, W)`` -> 检查级 logits：前一半切片的 logit 均值（与推理口径一致）。"""
    batch, k = slices.shape[0], slices.shape[1]
    flat = slices.reshape(batch * k, *slices.shape[2:])
    logits = model(flat).reshape(batch, k)
    return topk_mean(logits, max(1, k // 2))


def _autocast(torch, device_type: str, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(enabled=enabled)


def _make_scaler(torch, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def evaluate_model(model, loader, device, criterion, torch, *, min_recall: float) -> dict:
    """返回序列级与检查级指标（检查级 = 同一检查多条序列取最大概率）。"""
    model.eval()
    labels, scores = [], []
    case_scores: dict[str, float] = {}
    case_labels: dict[str, float] = {}
    total_loss, batches = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            slices = batch["slices"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            logits = forward_cases(model, slices)
            total_loss += float(criterion(logits, target))
            batches += 1
            probabilities = torch.sigmoid(logits).detach().float().cpu().numpy()
            labels.extend(target.detach().float().cpu().numpy().tolist())
            scores.extend(probabilities.tolist())
            for accession, value, label in zip(batch["accession"], probabilities, target.tolist()):
                case_scores[accession] = max(case_scores.get(accession, 0.0), float(value))
                case_labels[accession] = float(label)

    names = sorted(case_scores)
    return {
        "loss": total_loss / max(batches, 1),
        "volume_metrics": metrics_report(np.asarray(labels), np.asarray(scores), min_recall),
        "case_metrics": metrics_report(
            np.asarray([case_labels[name] for name in names]),
            np.asarray([case_scores[name] for name in names]),
            min_recall,
        ),
        "cases": len(names),
        "volumes": len(labels),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="复用已有 manifest.jsonl")
    parser.add_argument("--out-dir", type=Path, default=default_run_dir())
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--composition", choices=SPECIAL_POLICIES, default="exclude")
    parser.add_argument("--duplicate", choices=SPECIAL_POLICIES, default="exclude")
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--backbone", default="convnext_tiny.fb_in22k_ft_in1k")
    parser.add_argument("--pretrained-backbone", type=int, default=0, choices=(0, 1),
                        help="1 = 用 timm 预训练权重（需联网或本地缓存），0 = 从零随机初始化")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--slices-per-case", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=-1.0)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", type=int, default=1, choices=(0, 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-frequency-branch", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or default_log_dir()
    log, handle = make_logger(Path(log_dir) / "training.jsonl")

    try:
        import timm  # noqa: F401 - 提前检查，缺依赖时给出友好提示
        import torch
        from torch.utils.data import DataLoader

        from .model import AuthenticityNet
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 缺少训练依赖：{type(exc).__name__}: {exc}")
        print(f"        当前解释器：{sys.executable}")
        print("        安装：python -m pip install torch timm numpy nibabel")
        return 2

    set_seed(args.seed, torch)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    if args.manifest:
        records = read_manifest(args.manifest)
        annotation_root = None
        data_source = args.manifest.as_posix()
    else:
        annotation_root = args.annotation_root or find_annotation_root(args.data_root)
        records = discover_records(
            args.data_root,
            annotation_root=annotation_root,
            composition=args.composition,
            duplicate=args.duplicate,
            limit=args.limit_per_class,
        )
        assign_splits(
            records,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        write_manifest(records, out_dir / "manifest.jsonl")
        data_source = f"{args.data_root.as_posix()}/annotation"

    data_summary = summarize(records)
    print(json.dumps(data_summary, ensure_ascii=False, indent=2), flush=True)
    if data_summary["by_split"].get("val", {}).get("positives", 0) == 0:
        print("[WARN] val 集合没有阳性病例，AP 无法计算；调大 --val-fraction 或补数据", flush=True)

    train_config = SliceConfig(
        k=args.slices_per_case,
        size=args.image_size,
        train=not args.no_augment,
        slice_mode="random",
    )
    eval_config = SliceConfig(k=args.slices_per_case, size=args.image_size, train=False, slice_mode="uniform")
    train_set = OfficialSlices(records, "train", train_config, seed=args.seed)
    val_set = OfficialSlices(records, "val", eval_config, seed=args.seed)
    if len(train_set) == 0 or len(val_set) == 0:
        print("[ERROR] train/val 至少一个为空，检查数据扫描结果", flush=True)
        return 3

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, drop_last=False, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=(device.type == "cuda"),
    )

    pretrained_from = f"timm/{args.backbone}" if args.pretrained_backbone else "scratch"
    model = AuthenticityNet(
        backbone=args.backbone,
        pretrained=bool(args.pretrained_backbone),
        frequency_branch=not args.no_frequency_branch,
    ).to(device)
    print(f"device={device} backbone={args.backbone} pretrained_from={pretrained_from}", flush=True)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    pos_weight = positive_weight(train_set.labels(), args.pos_weight)
    criterion = build_criterion(pos_weight, device)
    labels = train_set.labels()
    print(
        f"train volumes={int(labels.size)} positives={int(labels.sum())} pos_weight={pos_weight:.2f}",
        flush=True,
    )

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = _make_scaler(torch, use_amp)
    data_source_train = f"{data_source}/train"
    data_source_val = f"{data_source}/val"

    best_ap, best_epoch, global_step, stale = -1.0, 0, 0, 0
    started = time.time()
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            if args.max_steps and step > args.max_steps:
                break
            slices = batch["slices"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, device.type, use_amp):
                logits = forward_cases(model, slices)
                loss = criterion(logits, target)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            running += float(loss.detach())
            seen += 1
            global_step += 1
            if step % 10 == 0 or step == 1 or args.smoke_test:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch} step {step} loss {running / seen:.4f} lr {current_lr:.2e}", flush=True)
                log(timestamp=utc_now(), epoch=epoch, step=global_step, phase="train", mode="training",
                    loss=round(running / seen, 6), lr=current_lr, data_source=data_source_train,
                    checkpoint=None, pretrained_from=pretrained_from)
        scheduler.step()
        train_loss = running / max(seen, 1)

        last_path = out_dir / "last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "backbone": args.backbone,
                "image_size": args.image_size,
                "slices_per_case": args.slices_per_case,
                "frequency_branch": not args.no_frequency_branch,
                "val_ap": best_ap,
                "pretrained_from": pretrained_from,
                "data_source": data_source,
                "seed": args.seed,
            },
            last_path,
        )

        metrics = evaluate_model(model, val_loader, device, criterion, torch, min_recall=args.min_recall)
        case_report = clean_report(metrics["case_metrics"])
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"val_case_AP={case_report['average_precision']} val_case_partial_AP={case_report['partial_ap']} "
            f"val_case_ROC_AUC={case_report['roc_auc']} (cases={metrics['cases']} volumes={metrics['volumes']})",
            flush=True,
        )
        log(timestamp=utc_now(), epoch=epoch, step=global_step, phase="val", mode="training",
            loss=round(metrics["loss"], 6), lr=optimizer.param_groups[0]["lr"],
            data_source=data_source_val, checkpoint=str(last_path), pretrained_from=pretrained_from,
            ap=case_report["average_precision"], partial_ap=case_report["partial_ap"],
            roc_auc=case_report["roc_auc"])

        score = case_report["average_precision"]
        score = -1.0 if score is None else float(score)
        if score > best_ap:
            best_ap, best_epoch, stale = score, epoch, 0
            best_path = out_dir / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "slices_per_case": args.slices_per_case,
                    "frequency_branch": not args.no_frequency_branch,
                    "val_ap": best_ap,
                    "pretrained_from": pretrained_from,
                    "data_source": data_source,
                    "seed": args.seed,
                },
                best_path,
            )
            print(f"  saved {best_path} (val_case_AP={best_ap:.4f})", flush=True)
            log(timestamp=utc_now(), epoch=epoch, step=global_step, phase="val", mode="training",
                loss=round(metrics["loss"], 6), lr=optimizer.param_groups[0]["lr"],
                data_source=data_source_val, checkpoint=str(best_path),
                pretrained_from=pretrained_from, ap=round(best_ap, 6))
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch} (best AP {best_ap:.4f} @ epoch {best_epoch})", flush=True)
                break
        if args.smoke_test and global_step >= 2:
            print("smoke test finished", flush=True)
            break

    summary = {
        "data_root": str(args.data_root),
        "annotation_root": None if annotation_root is None else str(annotation_root),
        "data_source": data_source,
        "data_summary": data_summary,
        "backbone": args.backbone,
        "pretrained_from": pretrained_from,
        "image_size": args.image_size,
        "slices_per_case": args.slices_per_case,
        "batch_size": args.batch_size,
        "pos_weight": pos_weight,
        "epochs_run": epoch,
        "best_case_ap": best_ap,
        "best_epoch": best_epoch,
        "minutes": round((time.time() - started) / 60, 2),
        "out_dir": str(out_dir),
        "log_file": str(Path(log_dir) / "training.jsonl"),
        "next_step": (
            "python -m tasks.goal1_authenticity.migrate_checkpoint "
            f"--source {out_dir / 'best.pt'} --also-last"
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
