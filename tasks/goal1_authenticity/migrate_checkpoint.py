"""把已有的目标一训练产物迁移到规范路径（运维程序，不参与比赛推理链路）。

规范 §5.2 / §20.1：权重固定放在
``/2026aicompetition/workspace/checkpoint/goal1_authenticity/model.pt``，
且不写入 Git。本程序负责：

1. 校验源文件确实是本项目训练出的 checkpoint（``torch.load`` 后必须含 ``model`` 键）；
2. 复制到规范路径（已存在时默认拒绝，``--force`` 会先备份为 ``model.pt.bak-<时间戳>``）；
3. 写一份 ``model_meta.json``（源路径、SHA256、backbone、image_size、slices_per_case、
   frequency_branch、val_ap、epoch、迁移时间），供日志与复现使用；
4. 可选把 ``last.pt`` 一并复制为 ``model_last.pt``。

用法（在仓库根目录执行）::

    # 先看会做什么（不落盘）
    python -m tasks.goal1_authenticity.migrate_checkpoint --dry-run

    # 指定源与根目录迁移
    python -m tasks.goal1_authenticity.migrate_checkpoint \
        --source /2026aicompetition/workspace/task1_runs/seed42/best.pt \
        --checkpoint-root /2026aicompetition/workspace/checkpoint

    # 覆盖已有权重（自动备份）
    python -m tasks.goal1_authenticity.migrate_checkpoint --source ... --force
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from .checkpoint import (
    META_FILENAME,
    MODEL_FILENAME,
    candidate_sources,
    checkpoint_root,
    goal_dir,
    meta_path,
    model_path,
)

EXIT_OK = 0
EXIT_SOURCE_MISSING = 2
EXIT_INVALID_CHECKPOINT = 3
EXIT_TARGET_EXISTS = 4


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_checkpoint(path: Path) -> dict:
    """校验并读出 checkpoint 里的关键元信息（失败抛 ValueError）。"""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"需要 torch 才能校验权重：{type(exc).__name__}: {exc}") from exc

    try:
        state = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 没有 weights_only
        state = torch.load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"无法读取 checkpoint：{type(exc).__name__}: {exc}") from exc

    if not isinstance(state, dict) or "model" not in state:
        raise ValueError("checkpoint 缺少 'model' 键：不是本项目的训练产物")
    model_state = state["model"]
    if not hasattr(model_state, "keys"):
        raise ValueError("checkpoint['model'] 不是 state_dict")

    return {
        "backbone": state.get("backbone"),
        "image_size": state.get("image_size"),
        "slices_per_case": state.get("slices_per_case"),
        "frequency_branch": state.get("frequency_branch"),
        "val_ap": state.get("val_ap"),
        "epoch": state.get("epoch"),
        "pretrained_from": state.get("pretrained_from"),
        "data_source": state.get("data_source"),
        "seed": state.get("seed"),
        "tensors": len(model_state),
    }


def migrate(
    source: Path,
    *,
    checkpoint_root_path: Path,
    force: bool = False,
    dry_run: bool = False,
    also_last: bool = False,
) -> dict:
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"源权重不存在：{source}")

    info = inspect_checkpoint(source)
    target = model_path(checkpoint_root_path)
    target_dir = goal_dir(checkpoint_root_path)

    report: dict[str, object] = {
        "source": str(source),
        "target": str(target),
        "checkpoint_root": str(checkpoint_root_path),
        "dry_run": dry_run,
        "source_bytes": source.stat().st_size,
        "source_sha256": sha256_of(source),
        **info,
    }

    if target.exists() and not force:
        report["status"] = "target_exists"
        report["hint"] = "目标已存在；确认要覆盖时加 --force（会自动备份）"
        return report

    if dry_run:
        report["status"] = "dry_run"
        return report

    target_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        backup = target.with_name(f"{target.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.copy2(target, backup)
        report["backup"] = str(backup)

    shutil.copy2(source, target)
    report["target_sha256"] = sha256_of(target)
    report["sha256_match"] = report["target_sha256"] == report["source_sha256"]

    last_copied = None
    if also_last:
        last_source = source.with_name("last.pt")
        if last_source.is_file():
            last_target = goal_dir(checkpoint_root_path) / "model_last.pt"
            shutil.copy2(last_source, last_target)
            last_copied = str(last_target)
    report["last_copied"] = last_copied

    meta = {
        "goal": "goal1_authenticity",
        "model_file": MODEL_FILENAME,
        "migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "migrated_by": "tasks.goal1_authenticity.migrate_checkpoint",
        "source": str(source),
        "source_sha256": report["source_sha256"],
        "bytes": report["source_bytes"],
        "backbone": info["backbone"],
        "image_size": info["image_size"],
        "slices_per_case": info["slices_per_case"],
        "frequency_branch": info["frequency_branch"],
        "val_ap": info["val_ap"],
        "epoch": info["epoch"],
        "pretrained_from": info["pretrained_from"],
        "data_source": info["data_source"],
        "seed": info["seed"],
    }
    meta_path(checkpoint_root_path).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report["status"] = "migrated"
    report["meta_file"] = str(meta_path(checkpoint_root_path))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="把目标一训练产物迁移到规范 checkpoint 路径",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=None,
                        help="训练产物路径（默认自动发现 best.pt）")
    parser.add_argument("--search-root", type=Path, action="append", default=None,
                        help="自动发现时的搜索根目录，可重复")
    parser.add_argument("--checkpoint-root", type=Path, default=None,
                        help="checkpoint 根目录（默认按 checkpoint.py 的优先级解析）")
    parser.add_argument("--also-last", action="store_true", help="同时复制 last.pt → model_last.pt")
    parser.add_argument("--force", action="store_true", help="覆盖已有权重（自动备份）")
    parser.add_argument("--dry-run", action="store_true", help="只检查与打印，不落盘")
    args = parser.parse_args(argv)

    root = checkpoint_root(args.checkpoint_root)

    source = args.source
    if source is None:
        candidates = candidate_sources(args.search_root)
        if not candidates:
            print(json.dumps({
                "status": "no_source",
                "checkpoint_root": str(root),
                "hint": "未自动发现 best.pt；请用 --source 指定，或用 --search-root 指定搜索目录",
            }, ensure_ascii=False, indent=2))
            return EXIT_SOURCE_MISSING
        source = candidates[-1]          # 取最新（路径排序最后）的一份
        print(json.dumps({
            "status": "auto_detected",
            "source": str(source),
            "candidates": [str(item) for item in candidates],
        }, ensure_ascii=False, indent=2))

    try:
        report = migrate(
            source,
            checkpoint_root_path=root,
            force=args.force,
            dry_run=args.dry_run,
            also_last=args.also_last,
        )
    except FileNotFoundError as exc:
        print(json.dumps({"status": "source_missing", "error": str(exc)}, ensure_ascii=False, indent=2))
        return EXIT_SOURCE_MISSING
    except ValueError as exc:
        print(json.dumps({"status": "invalid_checkpoint", "error": str(exc)}, ensure_ascii=False, indent=2))
        return EXIT_INVALID_CHECKPOINT

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report.get("status") == "target_exists":
        return EXIT_TARGET_EXISTS
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
