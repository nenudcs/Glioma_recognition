"""[研发] 离线评估与批量打分（规范 §5.1；比赛运行入口不得导入本文件）。

提供两类能力：

1. 排序指标（AP / 部分 AUC-PR / ROC-AUC / Recall@FPR）——训练、验证与离线打分共用；
2. 用本包 ``inference.py`` 对任意目录或 manifest 批量打分，产出
   ``authenticity_scores.jsonl`` 与 ``metrics.json``，用于上线前自检。

命令行::

    python -m tasks.goal1_authenticity.evaluate --dataset <NIfTI目录> --out-dir <输出目录>
    python -m tasks.goal1_authenticity.evaluate --manifest <训练清单> --split val --out-dir <输出目录>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import Goal1Config
from .inference import AuthenticityInference

NIFTI_SUFFIXES = (".nii", ".nii.gz")
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels.sum()
    if positives == 0 or positives == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / positives
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def partial_average_precision(labels: np.ndarray, scores: np.ndarray, min_recall: float = 0.5) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels.sum()
    if positives == 0 or positives == labels.size:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / positives
    keep = recall >= min_recall
    if not keep.any():
        return float("nan")
    recall = np.concatenate([[min_recall], recall[keep]])
    precision = np.concatenate([[precision[keep][0]], precision[keep]])
    area = float(_trapezoid(precision, recall))
    return area / max(1e-12, 1.0 - min_recall)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def recall_at_fpr(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.10) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    recall = np.cumsum(labels) / positives
    fpr = np.cumsum(1.0 - labels) / negatives
    valid = fpr <= max_fpr
    return float(recall[valid].max()) if valid.any() else 0.0


def metrics_report(
    labels,
    scores,
    min_recall: float = 0.5,
    max_fpr: float = 0.10,
) -> dict[str, float]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    return {
        "n": int(labels.size),
        "positives": int(labels.sum()),
        "average_precision": average_precision(labels, scores),
        "partial_ap": partial_average_precision(labels, scores, min_recall),
        "partial_ap_min_recall": float(min_recall),
        "roc_auc": roc_auc(labels, scores),
        "recall_at_fpr": recall_at_fpr(labels, scores, max_fpr),
        "recall_at_fpr_limit": float(max_fpr),
    }


def clean_report(report: dict[str, float]) -> dict[str, float | int | None]:
    cleaned: dict[str, float | int | None] = {}
    for key, value in report.items():
        if isinstance(value, float) and np.isnan(value):
            cleaned[key] = None
        elif isinstance(value, float):
            cleaned[key] = round(value, 6)
        else:
            cleaned[key] = value
    return cleaned


# --------------------------------------------------------------------------
# 目录/manifest 扫描
# --------------------------------------------------------------------------
def list_series(case_dir: str | Path) -> list[tuple[str, Path]]:
    """列出一个检查目录下的 ``[(series_uid, path), ...]``（优先同名原文件）。"""
    case_dir = Path(case_dir)
    items: list[tuple[str, Path]] = []
    if not case_dir.is_dir():
        return items
    for series_dir in sorted(path for path in case_dir.iterdir() if path.is_dir()):
        original = next(
            (
                candidate
                for candidate in (series_dir / f"{series_dir.name}{suffix}" for suffix in NIFTI_SUFFIXES)
                if candidate.is_file()
            ),
            None,
        )
        if original is not None:
            items.append((series_dir.name, original))
            continue
        nested = sorted(
            path for path in series_dir.iterdir()
            if path.is_file() and path.name.lower().endswith(NIFTI_SUFFIXES)
        )
        if nested:
            items.append((series_dir.name, nested[0]))
    if items:
        return items
    for path in sorted(case_dir.iterdir()):
        if path.is_file() and path.name.lower().endswith(NIFTI_SUFFIXES):
            stem = path.name
            for suffix in NIFTI_SUFFIXES:
                if stem.lower().endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            items.append((stem, path))
    return items


def _discover(dataset: Path) -> list[tuple[str, list[tuple[str, Path]]]]:
    cases: list[tuple[str, list[tuple[str, Path]]]] = []
    for child in sorted(path for path in dataset.iterdir() if path.is_dir()):
        series = list_series(child)
        if series:
            cases.append((child.name, series))
    if cases:
        return cases
    series = list_series(dataset)
    if series:
        return [(dataset.name or "case", series)]
    return []


def _manifest_label(record: dict) -> float | None:
    for key in ("is_not_human_body", "label"):
        value = record.get(key)
        if value is not None:
            return float(value)
    return None


def _manifest_cases(manifest: Path, data_root: Path, split: str | None = None):
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record_split = str(record.get("split") or "")
        if split and record_split != split:
            continue
        path = Path(record["path"])
        if not path.is_absolute():
            path = data_root / path
        stem = path.name
        for suffix in NIFTI_SUFFIXES:
            if stem.lower().endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        rows.append((str(record.get("accession") or stem), [(stem, path)], _manifest_label(record), record_split))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="目标一离线打分与指标", formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=None, help="manifest 相对路径根目录")
    parser.add_argument("--split", default=None, help="只评分 manifest 中该 split（如 val）")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="临时覆盖 GOAL1_CHECKPOINT")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    config = Goal1Config.from_env()
    if args.device or args.checkpoint:
        config = Goal1Config(
            **{
                **config.__dict__,
                **({"device": args.device} if args.device else {}),
            }
        )
    inference = AuthenticityInference(config)
    if args.checkpoint:
        inference.model_path = args.checkpoint.expanduser()

    # 先校验输入参数（避免路径写错时还去加载 100MB 权重），再加载模型
    if args.manifest:
        data_root = (args.data_root or args.manifest.parent).resolve()
        rows = _manifest_cases(args.manifest, data_root, args.split)
        cases = [(accession, series) for accession, series, _, _ in rows]
        labels = [label for _, _, label, _ in rows]
        splits = [record_split for _, _, _, record_split in rows]
        if not cases:
            parser.error(f"manifest 里没有可打分的记录：{args.manifest}（可用 --split val 只评验证集）")
    else:
        dataset = args.dataset.expanduser().resolve()
        if not dataset.is_dir():
            parser.error(
                f"--dataset 不是目录：{dataset}（可选：先用 verify.sh 的 --make-subset 从训练集切一个小样本）"
            )
        cases = _discover(dataset)
        labels = [None] * len(cases)
        splits = [""] * len(cases)
    if args.limit:
        cases, labels, splits = cases[: args.limit], labels[: args.limit], splits[: args.limit]
    if not cases:
        parser.error("没有找到可打分的 NIfTI")

    inference.load()
    print(json.dumps(inference.describe(), ensure_ascii=False, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.out_dir / "authenticity_scores.jsonl"
    collected_scores: list[float] = []
    collected_labels: list[float] = []

    with scores_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (accession, series) in enumerate(cases, 1):
            started = time.perf_counter()
            probabilities: list[float] = []
            detail = []
            for series_uid, path in series:
                try:
                    value = inference.score_volume(path)
                except Exception as exc:  # noqa: BLE001
                    value = None
                    detail.append({"series_uid": series_uid, "prob": None, "error": f"{type(exc).__name__}: {exc}"})
                else:
                    detail.append({"series_uid": series_uid, "prob": value})
                if value is not None:
                    probabilities.append(value)
            probability = max(probabilities) if probabilities else config_fallback(config)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            case_dir = args.out_dir / accession
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "prediction.json").write_text(
                json.dumps(
                    {"AccessionNumber": accession, "IsNotHumanBodyProb": round(float(probability), 6)},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            record = {
                "AccessionNumber": accession,
                "IsNotHumanBodyProb": round(float(probability), 6),
                "ProcessingTime_ms": elapsed_ms,
                "series": detail,
                "model_version": inference.model_version,
            }
            if labels[index - 1] is not None:
                record["label"] = labels[index - 1]
            if splits[index - 1]:
                record["split"] = splits[index - 1]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            collected_scores.append(float(probability))
            if labels[index - 1] is not None:
                collected_labels.append(float(labels[index - 1]))
            print(
                f"[{index}/{len(cases)}] {accession} IsNotHumanBodyProb={float(probability):.4f} "
                f"series={len(probabilities)}/{len(series)} {elapsed_ms}ms",
                flush=True,
            )

    if collected_labels and len(collected_labels) == len(collected_scores):
        payload = clean_report(metrics_report(collected_labels, collected_scores))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        (args.out_dir / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"scores: {scores_path}", file=sys.stderr)
    return 0


def config_fallback(config: Goal1Config) -> float:
    """所有序列都读不出时的兜底概率（仅离线打分用；线上由 task.py 抛不可降级错误）。"""
    return 0.5


if __name__ == "__main__":
    raise SystemExit(main())
