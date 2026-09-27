"""[研发] 拼接检测的阈值标定与自检（规范 §18.1 / §19.3 的离线评估；不进比赛运行链路）。

用法（在仓库根目录执行）::

    python -m tasks.goal2_stitched.dataset --data-root /2026aicompetition/datasets/training
    python -m tasks.goal2_stitched.evaluate \
        --data-root /2026aicompetition/datasets/training \
        --out-dir /2026aicompetition/workspace/Glioma_task12_runs \
        --target-fpr 0.05 --workers 4

标定口径：正类 = ``annotation/Composition``（拼接影像），负类 = 其余正常影像。
按目标假阳性率在负类分数分位上取阈值，输出 AP / 部分 AP / ROC-AUC / Recall@10%FPR 供比较，
并把 ``threshold`` 写进 ``stitched_calibration.json`` →
``GOAL2_STITCHED_THRESHOLD``。
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import checkpoint as checkpoint_module
from .config import METRICS, STATISTICS, Goal2StitchedConfig
from .dataset import configure_stdout, iter_images, labeled_volumes
from .model import (
    probability_from_score,
    reduce_scores,
    slice_residual_scores,
    volume_score,
)

_TRAPEZOID = getattr(np, "trapezoid", None) or np.trapz


# --------------------------------------------------------------------------
# 指标（纯 numpy，与目标一同一套口径）
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


def partial_average_precision(
    labels: np.ndarray,
    scores: np.ndarray,
    min_recall: float = 0.5,
) -> float:
    """限制在 recall ∈ [min_recall, 1] 的部分 AUC-PR（赛方对目标二的口径）。"""
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
    area = float(_TRAPEZOID(precision, recall))
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
    return float(
        (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


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
    labels: np.ndarray,
    scores: np.ndarray,
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
# 批量打分与标定
# --------------------------------------------------------------------------
def score_paths(
    paths: Sequence[Path],
    statistic: str = "max",
    metric: str = "curvature",
    band: float = 0.1,
    *,
    max_volumes: int = 0,
    workers: int = 1,
) -> list[float]:
    """批量算拼接分数（只返回成功结果）；``workers > 1`` 时多线程读盘。"""
    selected = list(paths)
    if 0 < max_volumes < len(selected):
        selected = selected[:max_volumes]
    if not selected:
        return []

    def one(path: Path) -> float | None:
        try:
            return volume_score(path, statistic, metric, band)
        except Exception:  # noqa: BLE001 - 单个文件读不了就跳过
            return None

    if workers > 1 and len(selected) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, selected))
    else:
        results = [one(path) for path in selected]
    return [float(value) for value in results if value is not None]


def calibrate(
    data_root: str | Path,
    *,
    annotation_root: str | Path | None = None,
    positive_kind: str = "composition",
    statistic: str = "max",
    metric: str = "curvature",
    band: float = 0.1,
    target_fpr: float = 0.05,
    max_volumes: int = 0,
    workers: int = 4,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    """在赛方数据上标定拼接阈值，返回可写入 ``stitched_calibration.json`` 的报告。"""
    volumes = labeled_volumes(
        data_root,
        annotation_root=annotation_root,
        positive_kind=positive_kind,
    )
    positives = score_paths(
        volumes["positives"], statistic, metric, band,
        max_volumes=0,                       # 正类不多，全部打分
        workers=workers,
    )
    negatives = score_paths(
        volumes["negatives"], statistic, metric, band,
        max_volumes=max_volumes,
        workers=workers,
    )
    payload: dict[str, object] = {
        "data_root": str(data_root),
        "positive_kind_requested": volumes["positive_kind_requested"][0],
        "positive_kind": volumes["positive_kind"][0],
        "positive_dir": [str(path) for path in volumes["positive_dir"]],
        "statistic": statistic,
        "metric": metric,
        "band": band,
        "workers": workers,
        "positives": len(positives),
        "negatives": len(negatives),
        "target_fpr": float(target_fpr),
    }
    if not positives or not negatives:
        payload["error"] = "需要 annotation/Composition 正类与正常影像负类"
        return payload

    threshold = float(np.quantile(np.asarray(negatives), 1.0 - target_fpr))
    labels = np.asarray([1.0] * len(positives) + [0.0] * len(negatives))
    scores = np.asarray(positives + negatives)
    payload.update(
        {
            "threshold": round(threshold, 6),
            "negative_fpr_at_threshold": float(
                (np.asarray(negatives) >= threshold).mean()
            ),
            "recall_at_threshold": float((np.asarray(positives) >= threshold).mean()),
            "positive_score_median": float(np.median(positives)),
            "negative_score_median": float(np.median(negatives)),
            "ranking_metrics": clean_report(metrics_report(labels, scores)),
        }
    )
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "stitched_calibration.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def calibration_payload(
    report: Mapping[str, object],
    *,
    source: str = "tasks.goal2_stitched.evaluate",
) -> dict[str, object]:
    """把标定报告压成服务启动时读取的 ``calibration.json``（规范路径）。"""
    return {
        "goal": checkpoint_module.GOAL_NAME,
        "source": source,
        "threshold": report.get("threshold"),
        "metric": report.get("metric"),
        "statistic": report.get("statistic"),
        "band": report.get("band"),
        "target_fpr": report.get("target_fpr"),
        "positives": report.get("positives"),
        "negatives": report.get("negatives"),
        "ranking_metrics": report.get("ranking_metrics"),
    }


def score_dataset(
    dataset: str | Path,
    *,
    statistic: str = "max",
    metric: str = "curvature",
    band: float = 0.1,
    threshold: float = 0.0,
    scale: float = 8.0,
    max_volumes: int = 0,
    workers: int = 4,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    """对任意 NIfTI 目录做离线打分（等价于服务里逐例算 ``IsStitchedProb``）。

    输出 ``stitched_scores.jsonl``：每行一个序列的分数与概率，可用于上线前抽查
    「服务会不会把正常影像判成拼接」。
    """
    root = Path(dataset).expanduser()
    files = iter_images(root)
    if 0 < max_volumes < len(files):
        files = files[:max_volumes]

    def one(path: Path) -> dict[str, object]:
        relative = path.relative_to(root)
        accession = relative.parts[0] if len(relative.parts) > 1 else path.stem
        series_uid = relative.parts[-2] if len(relative.parts) >= 3 else path.stem
        record: dict[str, object] = {
            "path": str(path),
            "accession": accession,
            "series_uid": series_uid,
        }
        try:
            scores = slice_residual_scores(path, metric, band)
            value = float(reduce_scores(scores, statistic))
            record["score"] = round(value, 6)
            record["probability"] = round(
                probability_from_score(value, threshold or 1e-6, scale), 6
            )
        except Exception as exc:  # noqa: BLE001 - 单个文件读不了就记录错误
            record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    if workers > 1 and len(files) > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            records = list(pool.map(one, files))
    else:
        records = [one(path) for path in files]

    ok = [record for record in records if "score" in record]
    values = [float(record["score"]) for record in ok]
    probabilities = [float(record["probability"]) for record in ok]
    payload: dict[str, object] = {
        "dataset": str(root),
        "statistic": statistic,
        "metric": metric,
        "band": band,
        "threshold": threshold,
        "files": len(records),
        "scored": len(ok),
        "failed": len(records) - len(ok),
        "score_median": None if not values else round(float(np.median(values)), 6),
        "score_max": None if not values else round(max(values), 6),
        "flagged": int(sum(1 for value in probabilities if value >= 0.5)),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "stitched_scores.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        payload["scores_file"] = str(out / "stitched_scores.jsonl")
    return payload


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="目标二（拼接）阈值标定与自检")
    parser.add_argument("--data-root", type=Path, default=Path("/2026aicompetition/datasets/training"))
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="对任意 NIfTI 目录做离线打分（不标定）")
    parser.add_argument("--positive-kind", default="composition",
                        help="拼接正类目录名前缀（默认 composition；目录改名时传实际前缀，"
                             "例如 compositing）")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None,
                        help="规范路径根目录，默认 /2026aicompetition/workspace/checkpoint")
    parser.add_argument("--no-write-calibration", action="store_true",
                        help="只打印，不写规范路径 calibration.json")
    parser.add_argument("--statistic", choices=STATISTICS, default="max")
    parser.add_argument("--metric", choices=METRICS, default="curvature")
    parser.add_argument("--band", type=float, default=0.1, help="两端丢弃比例（默认 0.1）")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--max-volumes", type=int, default=0, help="负类抽样上限（0 = 全量）")
    parser.add_argument("--workers", type=int, default=4, help="并行读盘线程数")
    args = parser.parse_args(argv)

    if args.dataset is not None:
        if not args.dataset.is_dir():
            print(json.dumps({"error": f"--dataset 不是目录：{args.dataset}"}, ensure_ascii=False))
            return 2
        payload = score_dataset(
            args.dataset,
            statistic=args.statistic,
            metric=args.metric,
            band=args.band,
            threshold=Goal2StitchedConfig.from_env(
                checkpoint_root=args.checkpoint_root
            ).threshold,
            max_volumes=args.max_volumes,
            workers=args.workers,
            out_dir=args.out_dir,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    payload = calibrate(
        args.data_root,
        annotation_root=args.annotation_root,
        positive_kind=args.positive_kind,
        statistic=args.statistic,
        metric=args.metric,
        band=args.band,
        target_fpr=args.target_fpr,
        max_volumes=args.max_volumes,
        workers=args.workers,
        out_dir=args.out_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if "error" in payload:
        return 2
    if not args.no_write_calibration:
        path = checkpoint_module.write_calibration(
            calibration_payload(payload),
            args.checkpoint_root,
        )
        print(
            "\n✅ 标定结果已写到规范路径：%s\n"
            "   服务启动时会自动读取（也可用 GOAL2_STITCHED_THRESHOLD 覆盖）"
            % path,
            file=sys.stderr,
        )
    print(
        "\n下一步：export GOAL2_STITCHED_THRESHOLD=%s"
        % round(float(payload["threshold"]), 6)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
