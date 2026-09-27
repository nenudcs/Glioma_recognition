"""[研发] 重复影像的图像对级标定与自检（规范 §18.2 / §19.3；不进比赛运行链路）。

用法（在仓库根目录执行）::

    python -m tasks.goal2_duplicate.dataset --data-root /2026aicompetition/datasets/training
    python -m tasks.goal2_duplicate.evaluate \
        --data-root /2026aicompetition/datasets/training \
        --out-dir /2026aicompetition/workspace/Glioma_task12_runs \
        --target-fpr 0.10 --workers 4

评估口径与赛方一致（图像对级）：

* **正类** = ``annotation/duplicate/`` 金标准对；
* **负类** = 随机采样对（易） + 检索命中但非金标准的候选对（难）——只报随机负类会高估指标，
  两套都写在报告里；
* 每例最多保留 Top-200（``GOAL2_DUPLICATE_MAX_PAIRS``），未命中的 pair 按 ``PairProb = 0``；
* 指标：AUC-PR（``average_precision``）、ROC-AUC、Recall@10%FPR、Precision@15%Recall；
* 阈值建议：按目标假阳性率在负类相似度分位上取 ``suggested_center`` →
  ``GOAL2_DUPLICATE_SIM_CENTER``。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import checkpoint as checkpoint_module
from .config import Goal2DuplicateConfig
from .dataset import (
    configure_stdout,
    find_special_root,
    gold_pairs,
    sample_negative_pairs,
    special_dir,
)
from .postprocess import similarity_distribution
from .retrieval import NearDuplicateIndex


# --------------------------------------------------------------------------
# 指标（与目标一/目标二拼接同一套口径）
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


def precision_at_recall(
    labels: np.ndarray,
    scores: np.ndarray,
    min_recall: float = 0.15,
) -> float:
    """召回率达到 ``min_recall`` 时的精确率（赛方 Precision@15%Recall 口径）。"""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = float(labels.sum())
    if positives <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1.0 - labels)
    recall = tp / positives
    valid = np.flatnonzero(recall >= min_recall)
    if valid.size == 0:
        return 0.0
    index = int(valid[0])
    return float(tp[index] / max(tp[index] + fp[index], 1e-12))


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else round(value, 6)


def pair_level_report(
    detected: Mapping[tuple[str, str], float],
    *,
    gold: set[tuple[str, str]],
    negatives: Sequence[tuple[str, str]],
    min_recall: float = 0.15,
    max_fpr: float = 0.10,
) -> dict[str, object]:
    """图像对级指标：正类 = 金标准对，负类 = 采样对（未命中按 0）。"""
    pairs = list(gold) + [pair for pair in negatives if pair not in gold]
    labels = np.asarray([1.0] * len(gold) + [0.0] * len(negatives), dtype=np.float64)
    scores = np.asarray([float(detected.get(pair, 0.0)) for pair in pairs], dtype=np.float64)
    hits = [pair for pair in gold if pair in detected]
    return {
        "pairs_scored": int(labels.size),
        "gold_pairs": len(gold),
        "gold_recall": None if not gold else round(len(hits) / len(gold), 6),
        "negative_pairs": len(negatives),
        "average_precision": _round(average_precision(labels, scores)),
        "roc_auc": _round(roc_auc(labels, scores)),
        "recall_at_fpr": _round(recall_at_fpr(labels, scores, max_fpr)),
        "recall_at_fpr_limit": float(max_fpr),
        "precision_at_recall": _round(precision_at_recall(labels, scores, min_recall)),
        "precision_at_recall_limit": float(min_recall),
    }


# --------------------------------------------------------------------------
# 标定
# --------------------------------------------------------------------------
def evaluate_duplicates(
    data_root: str | Path,
    *,
    annotation_root: str | Path | None = None,
    config: Goal2DuplicateConfig | None = None,
    max_volumes: int = 0,
    workers: int = 4,
    target_fpr: float = 0.10,
    negative_samples: int = 200_000,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    """在全库上跑一遍检索，用金标准 + 采样负类评估图像对级指标并给阈值建议。"""
    config = config or Goal2DuplicateConfig.from_env()
    root = Path(data_root).expanduser()
    pairs, gold_files, duplicate_dir = gold_pairs(root, annotation_root)

    # 训练集的检查号嵌在 annotation/<kind>/<accession>/ 下，而测试集的检查号在数据集根
    # 的一级目录，因此按「检查号所在的那一层」分别扫描后合并到一个索引里。
    annotation = annotation_root or find_special_root(root)
    annotation = Path(annotation).expanduser() if annotation else None
    duplicate_dir_for_scan = special_dir(annotation, "duplicate") if annotation else None
    composition_dir = special_dir(annotation, "composition") if annotation else None
    fake_dir = special_dir(annotation, "fake") if annotation else None
    specials = tuple(
        path for path in (duplicate_dir_for_scan, composition_dir, fake_dir) if path
    )
    scan_roots: list[tuple[Path, tuple[Path, ...]]] = []
    if duplicate_dir_for_scan is not None:
        scan_roots.append((duplicate_dir_for_scan, ()))
    if composition_dir is not None:
        scan_roots.append((composition_dir, ()))
    if annotation is not None:
        scan_roots.append((annotation, specials))
    scan_roots.append((root, tuple([annotation] if annotation else [])))

    index = NearDuplicateIndex(
        config.descriptor_params(),
        mode=config.mode,
        **config.index_kwargs(),
    )
    scans: list[dict[str, object]] = []
    for base, skip_dirs in scan_roots:
        sub_index, report = NearDuplicateIndex.from_dataset(
            base,
            params=config.descriptor_params(),
            max_volumes=max_volumes,
            skip_dirs=skip_dirs,
            workers=workers,
            mode=config.mode,
            **config.index_kwargs(),
        )
        for accession in sub_index.accessions():
            index.add(accession, sub_index.descriptors_of(accession))
        detail = report.as_dict()
        detail["skipped_dirs"] = [str(path) for path in skip_dirs]
        scans.append(detail)

    scan = {
        "roots": [str(path) for path, _ in scan_roots],
        "studies": len(index),
        "volumes": index.volumes,
        "scans": scans,
    }
    detected_pairs = index.all_pairs(max_pairs_per_study=config.max_pairs_per_study)
    detected ={(left, right): probability for left, right, probability, _ in detected_pairs}
    similarities = {(left, right): similarity for left, right, _, similarity in detected_pairs}

    known = set(index.accessions())
    gold_known = [pair for pair in pairs if pair[0] in known and pair[1] in known]
    gold_set = set(gold_known)
    random_negatives = sample_negative_pairs(
        index.accessions(),
        gold_set,
        count=min(negative_samples, max(len(index), 1) * 4),
    )
    hard_negatives = sorted(pair for pair in detected if pair not in gold_set)

    payload: dict[str, object] = {
        "data_root": str(root),
        "duplicate_dir": None if duplicate_dir is None else str(duplicate_dir),
        "config": config.describe(),
        "scan": scan,
        "studies": len(index),
        "volumes": index.volumes,
        "gold_files": [str(path) for path in gold_files],
        "gold_pairs": len(pairs),
        "gold_pairs_in_dataset": len(gold_known),
        "detected_pairs": len(detected),
        "detected_examples": [list(pair) for pair in list(detected)[:10]],
        "missed_examples": [list(pair) for pair in gold_known if pair not in detected][:10],
        "negative_pairs_random": len(random_negatives),
        "negative_pairs_hard": len(hard_negatives),
        "hard_negative_examples": [list(pair) for pair in hard_negatives[:10]],
        "metrics": pair_level_report(
            detected,
            gold=gold_set,
            negatives=sorted(set(random_negatives) | set(hard_negatives)),
        ),
        "metrics_random_negatives_only": pair_level_report(
            detected,
            gold=gold_set,
            negatives=random_negatives,
        ),
        "positive_similarity": similarity_distribution(
            [similarities[pair] for pair in gold_known if pair in similarities]
        ),
        "negative_similarity": similarity_distribution(
            [similarities[pair] for pair in hard_negatives if pair in similarities]
        ),
    }

    negative_sims = [similarities[pair] for pair in hard_negatives if pair in similarities]
    if negative_sims:
        suggested = float(
            np.quantile(np.asarray(negative_sims, dtype=np.float64), 1.0 - target_fpr)
        )
        payload["target_fpr"] = float(target_fpr)
        payload["suggested_center"] = round(suggested, 6)
        if gold_known:
            positive_sims = np.asarray(
                [similarities.get(pair, 0.0) for pair in gold_known], dtype=np.float64
            )
            payload["recall_at_suggested_center"] = round(
                float((positive_sims >= suggested).mean()), 6
            )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "duplicate_evaluation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return payload


def calibration_payload(
    report: Mapping[str, object],
    config: Goal2DuplicateConfig,
    *,
    source: str = "tasks.goal2_duplicate.evaluate",
) -> dict[str, object]:
    """把标定报告压成服务启动时读取的 ``calibration.json``（规范路径）。"""
    return {
        "goal": checkpoint_module.GOAL_NAME,
        "source": source,
        "center": report.get("suggested_center"),
        "scale": config.scale,
        "min_sim": config.min_sim,
        "coarse_floor": config.coarse_floor,
        "top_k": config.top_k,
        "max_pairs_per_study": config.max_pairs_per_study,
        "target_fpr": report.get("target_fpr"),
        "gold_pairs": report.get("gold_pairs_in_dataset"),
        "metrics": report.get("metrics"),
    }


def score_dataset(
    dataset: str | Path,
    *,
    config: Goal2DuplicateConfig | None = None,
    max_volumes: int = 0,
    workers: int = 4,
    out_dir: str | Path | None = None,
) -> dict[str, object]:
    """对任意测试集目录跑一遍重复检索，输出与提交同格式的候选对。

    用途：上线前在没有金标准的真实测试集上抽查——能出多少候选对、最高概率多少、
    有没有把明显不同的病例配成对（等价于服务里写 ``duplicate_pairs.jsonl`` 的那一步）。
    """
    config = config or Goal2DuplicateConfig.from_env()
    root = Path(dataset).expanduser()
    index, scan = NearDuplicateIndex.from_dataset(
        root,
        params=config.descriptor_params(),
        max_volumes=max_volumes,
        workers=workers,
        mode=config.mode,
        **config.index_kwargs(),
    )
    pairs = index.all_pairs(max_pairs_per_study=config.max_pairs_per_study)
    payload: dict[str, object] = {
        "dataset": str(root),
        "studies": len(index),
        "volumes": index.volumes,
        "pairs": len(pairs),
        "scan": scan.as_dict(),
        "top_pairs": [
            {"StudyUID": left, "StudyUID_dup": right, "PairProb": round(probability, 6),
             "similarity": round(similarity, 6)}
            for left, right, probability, similarity in pairs[:10]
        ],
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "duplicate_pairs_offline.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for left, right, probability, _ in pairs:
                handle.write(
                    json.dumps(
                        {
                            "StudyUID": left,
                            "StudyUID_dup": right,
                            "PairProb": round(float(probability), 6),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        payload["pairs_file"] = str(path)
    return payload


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="目标二（重复）图像对级标定与自检")
    parser.add_argument("--data-root", type=Path, default=Path("/2026aicompetition/datasets/training"))
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="对任意测试集目录跑重复检索并输出候选对（不标定）")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None,
                        help="规范路径根目录，默认 /2026aicompetition/workspace/checkpoint")
    parser.add_argument("--no-write-calibration", action="store_true",
                        help="只打印，不写规范路径 calibration.json")
    parser.add_argument("--slices", type=int, default=3)
    parser.add_argument("--coarse-grid", type=int, default=12)
    parser.add_argument("--fine-grid", type=int, default=64)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument("--target-fpr", type=float, default=0.10)
    parser.add_argument("--negative-samples", type=int, default=200_000)
    parser.add_argument("--max-volumes", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4, help="并行读盘线程数（默认 4）")
    args = parser.parse_args(argv)

    config = Goal2DuplicateConfig(
        slices=args.slices,
        coarse_grid=args.coarse_grid,
        fine_grid=args.fine_grid,
        shift=args.shift,
    )

    if args.dataset is not None:
        if not args.dataset.is_dir():
            print(json.dumps({"error": f"--dataset 不是目录：{args.dataset}"}, ensure_ascii=False))
            return 2
        payload = score_dataset(
            args.dataset,
            config=config,
            max_volumes=args.max_volumes,
            workers=args.workers,
            out_dir=args.out_dir,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    payload = evaluate_duplicates(
        args.data_root,
        annotation_root=args.annotation_root,
        config=config,
        max_volumes=args.max_volumes,
        workers=args.workers,
        target_fpr=args.target_fpr,
        negative_samples=args.negative_samples,
        out_dir=args.out_dir,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    center = payload.get("suggested_center")
    if center is not None and not args.no_write_calibration:
        path = checkpoint_module.write_calibration(
            calibration_payload(payload, config),
            args.checkpoint_root,
        )
        print(
            "\n✅ 标定结果已写到规范路径：%s\n"
            "   服务启动时会自动读取（也可用 GOAL2_DUPLICATE_SIM_CENTER 覆盖）" % path,
            file=sys.stderr,
        )
    if center is not None:
        print(
            "\n下一步：export GOAL2_DUPLICATE_SIM_CENTER=%s（当前水平 %s）"
            % (center, config.center)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
