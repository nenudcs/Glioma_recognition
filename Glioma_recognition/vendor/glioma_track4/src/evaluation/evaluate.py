"""离线评估：Dice/NSD/HD95 + 重复影像三项指标 + 阈值扫描。

评估口径（``--split``）：

- ``external``：**官方验证集**（``data/manifest_val.json``，由 ``01_probe.sh --val`` 生成）。
  验证集与训练集无交集 → **全折集成、不做留一**，这是"最终指标"口径。
- ``fold``：折内 val（旧口径；多折 OOF 时逐折调用，评估第 f 折用其余折模型）。
- ``auto``（默认）：验证集清单可用就用 external，否则回退 fold。

用法：
    python -m src.evaluation.evaluate --split auto [--fold 0] [--ckpt ...] [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..data.dataset import load_case_cached, make_targets, resolve_cache_dir
from ..inference.pipeline import GliomaPipeline
from ..inference.duplicate import calibrate, gold_similarities, match_pairs
from ..utils.config import (assert_data_source, external_val_manifest, fold_ckpts,
                            load_config, load_paths, resolve)
from . import metrics as M


def _default_grid() -> list[float]:
    return [round(0.05 * i, 2) for i in range(2, 19)]


def _empty_stats(grid: list[float]) -> list[list[float]]:
    """每阈值一个 ``[Dice 分子, Dice 分母]`` 累加器。"""
    return [[0.0, 0.0] for _ in grid]


def _accumulate(stats: list[list[float]], prob: np.ndarray, gt: np.ndarray,
                grid: list[float]) -> None:
    """把一例的**全图**概率图 + 真值图累计进 ``stats``，随后即可丢弃这两张图。

    ⚠️ 为什么必须流式（这是一次真实 OOM 的根因之一）：
    旧写法是 ``probs.append(res["seg"])`` + ``gts.append(gt)`` —— 而
    ``res["seg"]`` 是**全图体积** ``(2, D, H, W)``（`inference/sliding.py` 的
    `predict_volume` 会贴回原尺寸），``gt`` 同尺寸。按 4 通道体积约 71 MB 反推
    （`dataset.py:649` 的注释），单例 probs+gts ≈ 71 MB；val=666 时
    **≈ 47 GB** —— 32 GB 容器在收尾评估阶段必然 OOM（死在约 380 例处）。
    累计后常驻内存 ``2 通道 × 阈值数 × 2 个浮点数``，**与病例数无关**。
    """
    gt_c = gt > 0.5
    gsum = float(gt_c.sum())
    for i, t in enumerate(grid):
        pr = prob > t
        stats[i][0] += 2.0 * float((pr & gt_c).sum())
        stats[i][1] += float(pr.sum()) + gsum


def _best_from_stats(stats: list[list[float]], grid: list[float]) -> tuple[float, float]:
    """从流式累计里取最优 ``(阈值, Dice)``。

    **与旧的 ``_thr_scan(probs, gts, c)`` 逐位等价**：对任一阈值而言，参与相加的
    用例顺序没变、浮点累加次序完全一致；阈值遍历顺序也仍是 ``grid`` 的顺序
    （所以并列时的取法不变）。
    """
    best = (0.5, -1.0)
    for i, t in enumerate(grid):
        num, den = stats[i]
        d = num / den if den > 0 else 1.0
        if d > best[1]:
            best = (float(t), d)
    return best


def resolve_split(split: str, fold: int = 0) -> tuple[list[dict], dict, str]:
    """按 ``split`` 决定"评哪些病例、用哪份清单" → ``(cases, manifest, tag)``。

    - ``fold``：折内 val（旧口径，多折 OOF 时逐折调用）
    - ``external``：**官方验证集**（manifest_val）。验证集与训练集无交集，
      因此不需要留一折排除任何模型 —— 全折集成就是无泄漏的。
    - ``auto``：验证集清单可用就用 external，否则回退 fold。
    """
    if split == "auto":
        split = "external" if external_val_manifest()[0] is not None else "fold"

    if split == "external":
        man, path = external_val_manifest()
        if man is None:
            raise SystemExit(
                f"[eval] --split external 但验证集清单不可用：{path}\n"
                f"  · 生成：export VAL_ROOT=<验证集目录> && bash scripts/01_probe.sh --val")
        cases = [c for c in man["cases"] if c.get("images")]
        n_gt = sum(1 for c in cases if c.get("masks"))
        print(f"[eval] 外部验证集 {len(cases)} 例（清单 {path}）")
        if n_gt < len(cases):
            # "只参与重复影像评估"这句只在**有**重复金标准时才对。官方验证集实测只给
            # SeriesType.xlsx（没有字段金标准、也没有重复影像金标准）→ 无掩码的例
            # 其实哪个指标都不进；不改会让人以为它们至少还贡献了重复影像那一项。
            _dup = bool((man.get("special") or {}).get("gold_pairs"))
            print(f"[eval] ⚠️ {len(cases) - n_gt} 例无掩码 → "
                  + ("只参与重复影像评估，不计入分割指标" if _dup else
                     "不计入分割指标；验证集也没有重复影像金标准 → 这些例不参与任何指标"))
        return cases, man, "external"

    paths = load_paths()
    with open(resolve(paths["folds"]), encoding="utf-8") as f:
        folds = json.load(f)
    with open(resolve(paths["manifest"]), encoding="utf-8") as f:
        man = json.load(f)
    assert_data_source(man, phase="train")                        # 数据源合规闸门
    by_acc = {c["accession"]: c for c in man["cases"]}
    return [by_acc[a] for a in folds[str(fold)]["val"] if a in by_acc], man, "fold"


def eval_cases(cases: list[dict], man: dict, ckpt: str | list[str], tag: str = "fold",
               fold: int | None = None, limit: int | None = None) -> dict:
    """评估给定病例集合；``ckpt`` 传**列表**即为多折集成（含 TTA 后的均值融合）。

    分割指标只统计**有掩码**的病例（验证集清单可能只给图不给掩码）；
    无掩码的病例仍参与目标二（重复影像）——它只需要图像与前向特征。
    """
    if limit:
        cases = cases[:limit]
    pre = load_config("preprocess.yaml")
    ckpts = [ckpt] if isinstance(ckpt, str) else [str(c) for c in ckpt]
    pipe = GliomaPipeline(ckpts)
    # 阈值扫描**流式累计**（不再把每例的全图概率体/真值体留在内存里；见 _accumulate）
    rows, embeds, feats = [], {}, {}
    thr_grid = _default_grid()
    thr_stats = [_empty_stats(thr_grid), _empty_stats(thr_grid)]
    n_thr = 0
    from ..inference.pipeline import postprocess
    from ..inference.writer import case_fingerprint

    cache_dir = resolve_cache_dir()
    n_no_gt = 0
    for case in cases:
        acc = case["accession"]
        try:
            vol, aff, masks = load_case_cached(case, pre, cache_dir)
            res = pipe.predict_prob(case, vol)
            embeds[acc] = np.asarray(res["embed"], np.float32)
            feats[acc] = case_fingerprint(case)
            if not masks:
                n_no_gt += 1
                continue
            gt = make_targets(masks, vol.shape[1:])
            spacing = tuple(float(np.linalg.norm(aff[:3, i])) for i in range(3))
            pc, pp = postprocess(res["seg"][0] > pipe.thresholds[0],
                                 res["seg"][1] > pipe.thresholds[1],
                                 int(pre["inference"]["min_tumor_voxels"]), spacing)
            # 累计后立即丢弃这两张全图（只留统计量）——这是 32 GB 容器能跑完的关键
            for _c in range(2):
                _accumulate(thr_stats[_c], res["seg"][_c], gt[_c], thr_grid)
            n_thr += 1
            rows.append({"accession": acc,
                         "dice_core": M.dice(pc, gt[0] > 0.5),
                         "dice_peri": M.dice(pp, gt[1] > 0.5),
                         "nsd_core": M.nsd(pc, gt[0] > 0.5, spacing),
                         "nsd_peri": M.nsd(pp, gt[1] > 0.5, spacing),
                         "hd95_peri": M.hd95(pp, gt[1] > 0.5, spacing)})
        except Exception as e:                                    # noqa: BLE001
            print(f"[eval] ✗ {acc}: {e}")

    rep: dict = {"split": tag, "fold": fold, "n": len(rows),
                 "thresholds": pipe.thresholds}
    if n_no_gt:
        rep["n_no_mask"] = n_no_gt
        print(f"[eval] ⚠️ {n_no_gt} 例无掩码，未计入分割指标（有效 n={len(rows)}）")
    if rows:
        for k in ("dice_core", "dice_peri", "nsd_core", "nsd_peri"):
            rep[k] = float(np.mean([r[k] for r in rows]))
        rep["dice_mean"] = 0.5 * (rep["dice_core"] + rep["dice_peri"])
        rep["hd95_peri"] = float(np.nanmean([r["hd95_peri"] for r in rows]))
        if n_thr:
            rep["thr_scan_core"] = _best_from_stats(thr_stats[0], thr_grid)
            rep["thr_scan_peri"] = _best_from_stats(thr_stats[1], thr_grid)

    gold = {tuple(sorted(p)) for p in (man.get("special", {}).get("gold_pairs") or [])}
    if gold and embeds:
        pos, neg = gold_similarities(embeds, [list(g) for g in gold])
        calib = calibrate(pos, neg)
        w_fp = float(pre["inference"].get("fp_weight", 0.5))
        for w in (0.0, w_fp):
            pairs = match_pairs(embeds, calib, topk=int(pre["inference"]["duplicate_topk"]),
                                feats=feats, w_fp=w)
            rep[f"duplicate_w{w}"] = M.duplicate_report(pairs, gold, sorted(embeds))
        rep["calib"] = calib
    else:
        # 静默少几个键，会让人以为"重复影像那一项跑过了、只是没打印"。
        # 官方验证集实测只给 SeriesType.xlsx → 没有重复影像金标准，这一项**算不了**。
        _why = ("没有重复影像金标准（special.gold_pairs 为空；官方验证集实测只给 "
                "SeriesType.xlsx）" if not gold else "没有可用嵌入")
        print(f"[eval] ℹ️ 跳过重复影像指标：{_why}")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=("auto", "fold", "external"), default="auto",
                    help="auto（默认）=有验证集清单就评官方验证集，否则评折内 val；"
                         "fold=强制折内 val；external=强制官方验证集（全折集成，不做留一）")
    ap.add_argument("--ckpt", default=None,
                    help="权重路径；**逗号分隔即为多折集成**，如 g4_fold0/best.pth,g4_fold1/best.pth"
                         "（--split external 且未指定时=自动取全部折权重）")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    paths = load_paths()
    split = a.split
    if split == "auto":
        split = "external" if external_val_manifest()[0] is not None else "fold"
    raw = a.ckpt
    if not raw:
        if split == "external":
            ck = fold_ckpts(paths)
            if not ck:
                raise SystemExit("[eval] --split external 需要折权重，但 checkpoints/ 下没有 "
                                 "g4_fold*/best.pth；请先训练或显式 --ckpt")
            raw = ",".join(ck)
        else:
            raw = os.path.join(paths["checkpoints_dir"], f"g4_fold{a.fold}", "best.pth")
    ckpts: list[str] = []
    for c in str(raw).split(","):
        c = c.strip()
        if c:
            ckpts.append(c if os.path.isabs(c) else resolve(c))
    for c in ckpts:
        if not os.path.exists(c):
            raise SystemExit(f"[eval] 权重不存在: {c}")
    print(f"[eval] split={split}"
          f"{f'  fold={a.fold}' if split == 'fold' else '（官方验证集）'}  "
          f"权重 {len(ckpts)} 个{'（多折集成）' if len(ckpts) > 1 else ''}", flush=True)
    for c in ckpts:
        print(f"        · {c}", flush=True)
    if split == "external" and len(ckpts) == 1:
        print("[eval] ⚠️ 官方验证集与训练集无交集 → **多折集成**才是最终口径的数字；"
              "单折结果只适合快速自检。", flush=True)
    cases, man, tag = resolve_split(split, a.fold)
    eval_cases(cases, man, ckpts, tag=tag,
               fold=(a.fold if tag == "fold" else None), limit=a.limit)


if __name__ == "__main__":
    main()
