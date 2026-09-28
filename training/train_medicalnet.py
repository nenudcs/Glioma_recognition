#!/usr/bin/env python3
"""Study-level MedicalNet training with accession-safe splits and early stopping."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from training.models.medicalnet import MedicalNet3DClassifier
from training.study_dataset import StudyNiftiDataset, StudyRecord, build_study_records


def split_records(records: list[StudyRecord], *, val_fraction: float, test_fraction: float, seed: int):
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be in [0, 1)")
    rng = random.Random(seed)
    by_label: dict[int, list[StudyRecord]] = {0: [], 1: []}
    for record in records:
        if record.label is not None:
            by_label[int(record.label >= 0.5)].append(record)
    splits = {"train": [], "val": [], "test": []}
    for label_records in by_label.values():
        rng.shuffle(label_records)
        test_count = round(len(label_records) * test_fraction)
        val_count = round(len(label_records) * val_fraction)
        splits["test"].extend(label_records[:test_count])
        splits["val"].extend(label_records[test_count : test_count + val_count])
        splits["train"].extend(label_records[test_count + val_count :])
    for values in splits.values():
        rng.shuffle(values)
    return splits["train"], splits["val"], splits["test"]


def roc_auc(labels: Iterable[float], scores: Iterable[float]) -> float | None:
    y = np.asarray(list(labels), dtype=np.float64)
    s = np.asarray(list(scores), dtype=np.float64)
    if y.size == 0 or np.unique(y).size < 2:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    positives = y.sum()
    negatives = y.size - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = np.flatnonzero(y == 1).astype(np.float64) + 1.0
    return float((ranks.sum() - positives * (positives + 1) / 2) / (positives * negatives))


def evaluate(model, loader, device, criterion) -> dict[str, float | int | None]:
    model.eval()
    total_loss = 0.0
    count = 0
    labels: list[float] = []
    scores: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device)
            target = torch.as_tensor([float(value) for value in batch["label"]], device=device)
            logits = model(image).reshape(-1)
            loss = criterion(logits, target)
            total_loss += float(loss.item()) * image.shape[0]
            count += image.shape[0]
            labels.extend(target.cpu().tolist())
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return {"loss": total_loss / max(count, 1), "roc_auc": roc_auc(labels, scores), "samples": count}


def write_split(path: Path, records: list[StudyRecord]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["AccessionNumber", "label"])
        writer.writeheader()
        writer.writerows({"AccessionNumber": r.accession, "label": r.label} for r in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-index", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--label-column", default="check__glioma_with_label__std")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-shape", nargs=3, type=int, default=(32, 32, 16))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoint_medicalnet"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = build_study_records(
        args.file_index,
        args.labels_csv,
        label_column=args.label_column,
        require_all_modalities=False,
    )
    train_records, val_records, test_records = split_records(
        records, val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed
    )
    if not train_records or not val_records:
        raise RuntimeError("train/val split is empty; check complete four-modality studies")
    args.output.mkdir(parents=True, exist_ok=True)
    write_split(args.output / "train_split.csv", train_records)
    write_split(args.output / "val_split.csv", val_records)
    write_split(args.output / "test_split.csv", test_records)

    shape = tuple(args.target_shape)
    train_set = StudyNiftiDataset(train_records, target_shape=shape, task="classification")
    val_set = StudyNiftiDataset(val_records, target_shape=shape, task="classification")
    test_set = StudyNiftiDataset(test_records, target_shape=shape, task="classification")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalNet3DClassifier(depth=50, in_channels=4, num_classes=1, checkpoint=args.checkpoint).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()
    best_auc = -float("inf")
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            target = torch.as_tensor([float(value) for value in batch["label"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(image).reshape(-1), target)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * image.shape[0]
            seen += image.shape[0]
        train_loss = running / max(seen, 1)
        metrics = evaluate(model, val_loader, device, criterion)
        record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)

        score = metrics["roc_auc"]
        score_value = -float("inf") if score is None else float(score)
        if score_value > best_auc + args.min_delta:
            best_auc = score_value
            best_epoch = epoch
            stale = 0
            torch.save(
                {"model": model.state_dict(), "backend": "medicalnet", "model_depth": 50, "in_channels": 4, "label_column": args.label_column},
                args.output / "best.pt",
            )
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch={epoch}, best_epoch={best_epoch}, best_val_roc_auc={best_auc}")
                break

    best_path = args.output / "best.pt"
    if best_path.is_file():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
    test_metrics = evaluate(model, test_loader, device, criterion) if test_records else None
    summary = {
        "records_with_four_modalities": len(records), "train": len(train_records),
        "val": len(val_records), "test": len(test_records), "target_shape": shape,
        "best_epoch": best_epoch,
        "best_val_roc_auc": None if best_auc == -float("inf") else best_auc,
        "test_metrics": test_metrics, "history": history,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
