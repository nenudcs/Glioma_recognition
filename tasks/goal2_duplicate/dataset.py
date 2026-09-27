"""[研发] 重复影像的金标准与负类采样（标定/自检用，**不进比赛运行链路**）。

赛方金标准（``annotation/duplicate/`` 下的非 NIfTI 文件）每行一对检查号::

    src_img, desc_img
    GLIOMA_001, GLIOMA_007

``(A, B)`` 与 ``(B, A)`` 视为同一检查影像对；未出现在提交文件里的 pair 在评测时按
``PairProb = 0`` 处理，因此图像对级 AUC-PR 的负类是「除金标准对以外的全部 pair」——
本地标定时用「随机采样对 + 检索命中的难负类」近似，两者都在报告里分开给出。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

# 目录扫描规则与目标二（拼接）共用一份实现，避免两处漂移
from tasks.goal2_stitched.dataset import (  # noqa: F401 - 研发侧复用
    SPECIAL_KINDS,
    configure_stdout,
    candidate_kind_dirs,
    find_special_root,
    inspect_layout,
    is_image,
    iter_images,
    looks_like_code_repo,
    special_dir,
)


def parse_gold_pairs(path: str | Path) -> list[tuple[str, str]]:
    """解析 ``src_img, desc_img`` 金标准（每行一对检查号，顺序无关）。"""
    pairs: list[tuple[str, str]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        text = line.strip().replace("\t", ",")
        if not text or text.startswith("#"):
            continue
        parts = [item.strip() for item in text.split(",") if item.strip()]
        if len(parts) >= 2 and parts[0].lower() not in {"src_img", "studyuid"}:
            left, right = str(parts[0]), str(parts[1])
            pairs.append((left, right) if left <= right else (right, left))
    return pairs


# 金标准是文本表格（csv/txt/tsv），必须排除 NIfTI 影像与掩膜，
# 否则 ``*.nii.gz`` 的 ``suffix`` 是 ``.gz``，会被误当成文本读进解析器。
GOLD_SUFFIXES = (".csv", ".txt", ".tsv", ".list", ".json", ".xlsx")
GOLD_NAME_HINTS = ("mask", "seg", "label", "roi")


def is_gold_file(path: Path) -> bool:
    """是否为金标准文件（排除影像、掩膜与隐藏文件）。"""
    name = path.name.lower()
    if not path.is_file() or name.startswith("."):
        return False
    if name.endswith((".nii", ".nii.gz", ".nii.bz2")):
        return False
    if any(hint in name for hint in GOLD_NAME_HINTS):
        return False
    return path.suffix.lower() in GOLD_SUFFIXES


def gold_pairs_from_dir(directory: str | Path) -> tuple[list[tuple[str, str]], list[Path]]:
    """从 ``annotation/duplicate`` 目录里找出全部金标准文件并解析。"""
    base = Path(directory)
    files = [path for path in sorted(base.rglob("*")) if is_gold_file(path)]
    pairs: list[tuple[str, str]] = []
    for path in files:
        pairs.extend(parse_gold_pairs(path))
    return pairs, files


def gold_pairs(data_root: str | Path, annotation_root: str | Path | None = None):
    """定位 ``annotation/duplicate`` 并返回 ``(pairs, files, duplicate_dir)``。"""
    root = Path(data_root).expanduser()
    ann = Path(annotation_root).expanduser() if annotation_root else find_special_root(root)
    duplicate_dir = special_dir(ann, "duplicate") if ann else None
    if duplicate_dir is None:
        return [], [], None
    pairs, files = gold_pairs_from_dir(duplicate_dir)
    return pairs, files, duplicate_dir


def sample_negative_pairs(
    accessions: Sequence[str],
    gold: set[tuple[str, str]],
    *,
    count: int,
    seed: int = 2026,
) -> list[tuple[str, str]]:
    """随机采样非金标准 pair（负类的一部分；另一部分是检索命中的难负类）。"""
    total = len(accessions)
    if total < 2 or count <= 0:
        return []
    rng = np.random.default_rng(seed)
    sampled: set[tuple[str, str]] = set()
    attempts = 0
    max_attempts = count * 50
    while len(sampled) < count and attempts < max_attempts:
        attempts += 1
        left, right = rng.choice(total, size=2, replace=False)
        pair = tuple(sorted((accessions[int(left)], accessions[int(right)])))
        if pair[0] == pair[1] or pair in gold or pair in sampled:
            continue
        sampled.add(pair)
    return sorted(sampled)


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="目标二（重复）金标准体检（不跑检索）")
    parser.add_argument("--data-root", type=Path, default=Path("/2026aicompetition/datasets/training"))
    parser.add_argument("--annotation-root", type=Path, default=None)
    args = parser.parse_args(argv)

    raw_root = str(args.data_root).strip()
    resolved_root = Path(raw_root).expanduser()
    if raw_root in ("", ".") or not resolved_root.is_dir() or looks_like_code_repo(resolved_root.resolve()):
        print(
            json.dumps(
                {
                    "error": f"--data-root 无效或指向代码仓库：{raw_root!r}",
                    "hint": "$DATA_ROOT 多半是空的（新 shell/新 pod 后 export 会丢）。"
                    "先 `echo \"[$DATA_ROOT]\"`，再用绝对路径重跑："
                    "--data-root /2026aicompetition/datasets/training",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    pairs, files, duplicate_dir = gold_pairs(args.data_root, args.annotation_root)
    layout = inspect_layout(args.data_root, annotation_root=args.annotation_root)
    payload = {
        "data_root": str(args.data_root),
        "duplicate_dir": None if duplicate_dir is None else str(duplicate_dir),
        "gold_files": [str(path) for path in files],
        "gold_pairs": len(pairs),
        "unique_studies": len({item for pair in pairs for item in pair}),
        "examples": [list(pair) for pair in pairs[:5]],
        "special_dirs": layout["special_dirs"],
        "special_image_counts": layout["special_image_counts"],
        "annotation_children": layout["annotation_children"],
        "data_root_children": layout["data_root_children"],
        "candidate_dirs": {
            kind: candidate_kind_dirs(args.data_root, kind) for kind in ("duplicate",)
        },
    }
    if duplicate_dir is None:
        payload["hints"] = [
            "未找到名字以 duplicate 开头的目录：金标准文件（每行 src_img,desc_img）通常放在那里；"
            "用 candidate_dirs 找到真实名字，或用 --annotation-root 指定上一层"
        ]
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if pairs else 2


if __name__ == "__main__":
    raise SystemExit(main())
