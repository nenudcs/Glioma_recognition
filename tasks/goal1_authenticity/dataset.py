"""[研发] 训练数据扫描、标签与切片数据集（规范 §5.1；比赛运行入口不得导入本文件）。

沿用赛方《公共数据集格式说明》的目录约定（赛道四）：

```text
<data_root>/
├── annotation/
│   ├── fake/          → 目标一正类（假人体/非人体）
│   ├── Composition/   → 目标二·拼接正类（目标一默认排除）
│   ├── duplicate/     → 目标二·重复（含金标准，目标一默认排除）
│   └── <检查号>/…     → 正常影像 = 目标一负类
└── <其它目录>/…       → 正常影像（同样是负类）
```

还有 ``SeriesType.xlsx``（若存在）由 ``data/loader.py`` 在推理侧解析，训练侧不依赖它。

命令行（数据体检，不需要 GPU）::

    python -m tasks.goal1_authenticity.dataset --data-root /2026aicompetition/datasets/training
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .augmentations import AugmentConfig, augment_slices
from .preprocess import volume_to_slices

NIFTI_SUFFIXES = (".nii", ".nii.gz")
MASK_HINTS = ("mask", "seg", "label", "roi")
SPECIAL_KINDS = ("fake", "composition", "duplicate")
SPECIAL_POLICIES = ("exclude", "negative")


@dataclass
class Record:
    """一个训练样本 = 一个检查的一条序列。"""

    path: str
    accession: str
    series_uid: str
    label: int
    group: str
    split: str = "train"

    def as_dict(self) -> dict:
        return asdict(self)


def _stem(path: Path) -> str:
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def is_image(path: Path) -> bool:
    if not path.name.lower().endswith(NIFTI_SUFFIXES):
        return False
    name = path.name.lower()
    return not any(hint in name for hint in MASK_HINTS)


def iter_images(root: Path, skip_dirs: Sequence[Path] = ()) -> list[Path]:
    """递归收集输入影像（跳过 ``skip_dirs`` 子树）。"""
    if not root.is_dir():
        return []
    skips = [Path(item).resolve() for item in skip_dirs if item is not None]
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_image(path):
            continue
        resolved = path.resolve()
        if any(_is_within(resolved, skip) for skip in skips):
            continue
        found.append(path)
    return found


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def special_dir(base: Path | None, kind: str) -> Path | None:
    """在 ``base`` 下按大小写不敏感前缀找 ``fake`` / ``composition`` / ``duplicate``。"""
    if base is None or not base.is_dir():
        return None
    for child in sorted(path for path in base.iterdir() if path.is_dir()):
        if child.name.strip().lower().startswith(kind):
            return child
    return None


def find_special_root(data_root: Path, kinds: Sequence[str] = SPECIAL_KINDS) -> Path | None:
    """定位含任一特殊目录的标注根目录（``data_root`` 自身或其子目录）。"""
    if not data_root.is_dir():
        return None
    candidates = [data_root, data_root / "annotation"]
    candidates += sorted(path for path in data_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if any(special_dir(candidate, kind) is not None for kind in kinds):
            return candidate
    for child in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for grandchild in sorted(path for path in child.iterdir() if path.is_dir()):
            if any(special_dir(grandchild, kind) is not None for kind in kinds):
                return grandchild
    return None


def find_annotation_root(data_root: Path) -> Path | None:
    """含 ``fake`` 目录的标注根（目标一训练必须存在）。"""
    return find_special_root(data_root, kinds=("fake",))


def _identity(path: Path, base: Path) -> tuple[str, str]:
    try:
        relative = path.relative_to(base)
    except ValueError:
        relative = Path(path.name)
    parts = relative.parts
    if len(parts) >= 3:
        return parts[0], parts[-2]
    if len(parts) == 2:
        return parts[0], _stem(path)
    return _stem(path), _stem(path)


def discover_records(
    data_root: Path,
    *,
    annotation_root: Path | None = None,
    composition: str = "exclude",
    duplicate: str = "exclude",
    limit: int = 0,
) -> list[Record]:
    """扫描赛方数据 → 带标签样本；``limit`` 为每类上限（调试用）。"""
    if composition not in SPECIAL_POLICIES or duplicate not in SPECIAL_POLICIES:
        raise ValueError(f"策略必须是 {SPECIAL_POLICIES} 之一")
    if not data_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{data_root}")

    root = data_root.resolve()
    ann = (annotation_root or find_annotation_root(root))
    if ann is not None:
        ann = ann.resolve()

    fake_dir = special_dir(ann, "fake")
    composition_dir = special_dir(ann, "composition")
    duplicate_dir = special_dir(ann, "duplicate")
    if fake_dir is None:
        raise FileNotFoundError(
            f"在 {root} 下找不到 fake 标注目录；如布局不同请传 --annotation-root"
        )

    records: list[Record] = []
    for path in iter_images(fake_dir):
        accession, series_uid = _identity(path, fake_dir)
        records.append(Record(str(path), accession, series_uid, 1, "fake"))

    seen: set[str] = set()
    negative_roots: list[tuple[Path, tuple[Path, ...]]] = []
    if ann is not None:
        negative_roots.append(
            (ann, tuple(item for item in (fake_dir, composition_dir, duplicate_dir) if item))
        )
    negative_roots.append(
        (root, tuple(item for item in (ann, fake_dir, composition_dir, duplicate_dir) if item))
    )
    for base, skips in negative_roots:
        for path in iter_images(base, skip_dirs=skips):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            accession, series_uid = _identity(path, base)
            records.append(Record(str(path), accession, series_uid, 0, "normal"))

    for kind, directory, policy in (
        ("composition", composition_dir, composition),
        ("duplicate", duplicate_dir, duplicate),
    ):
        if directory is None or policy != "negative":
            continue
        for path in iter_images(directory):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            accession, series_uid = _identity(path, directory)
            records.append(Record(str(path), accession, series_uid, 0, kind))

    records.sort(key=lambda item: (item.label, item.group, item.path))
    if limit:
        from collections import Counter

        kept: list[Record] = []
        quota: Counter = Counter()
        for record in records:
            if quota[record.label] < limit:
                kept.append(record)
                quota[record.label] += 1
        records = kept
    return records


def assign_splits(
    records: list[Record],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> list[Record]:
    """按检查号分层切分（同一检查不会跨集合）。"""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val/test 比例非法")
    rng = random.Random(seed)
    cases: dict[int, dict[str, list[Record]]] = {}
    for record in records:
        cases.setdefault(record.label, {}).setdefault(record.accession, []).append(record)

    for label, grouped in cases.items():
        accessions = sorted(grouped)
        rng.shuffle(accessions)
        total = len(accessions)
        n_test = int(round(total * test_fraction)) if test_fraction else 0
        n_val = int(round(total * val_fraction)) if val_fraction else 0
        if total > 1:
            n_val = max(1, min(n_val, total - 1 - n_test))
            n_test = max(0, min(n_test, total - 1 - n_val))
        else:
            n_val = n_test = 0
        for index, accession in enumerate(accessions):
            if index < n_test:
                split = "test"
            elif index < n_test + n_val:
                split = "val"
            else:
                split = "train"
            for record in grouped[accession]:
                record.split = split
    return records


def summarize(records: Sequence[Record]) -> dict:
    records = list(records)
    by_split: dict[str, dict[str, object]] = {}
    for split in ("train", "val", "test"):
        subset = [record for record in records if record.split == split]
        if not subset:
            continue
        by_split[split] = {
            "volumes": len(subset),
            "cases": len({record.accession for record in subset}),
            "positives": sum(record.label for record in subset),
        }
    groups: dict[str, int] = {}
    for record in records:
        groups[record.group] = groups.get(record.group, 0) + 1
    return {
        "volumes": len(records),
        "cases": len({record.accession for record in records}),
        "positives": sum(record.label for record in records),
        "negatives": sum(1 for record in records if record.label == 0),
        "by_group": groups,
        "by_split": by_split,
    }


def write_manifest(records: Sequence[Record], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False) + "\n")
    return path


def read_manifest(path: Path) -> list[Record]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(Record(**json.loads(line)))
    return records


@dataclass
class SliceConfig:
    """切片几何 + 增广参数（训练侧）。"""

    k: int = 16
    size: int = 224
    min_std: float = 0.05
    train: bool = False
    slice_mode: str = "random"
    augment: AugmentConfig = field(default_factory=AugmentConfig)


class OfficialSlices:
    """一个样本 = 一个检查的一条序列（K 张 2.5D 切片）。"""

    def __init__(
        self,
        records: Sequence[Record],
        split: str,
        config: SliceConfig,
        seed: int = 0,
    ) -> None:
        self.records = [record for record in records if record.split == split]
        self.split = split
        self.config = config
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def labels(self) -> np.ndarray:
        return np.asarray([record.label for record in self.records], dtype=np.float32)

    def __getitem__(self, index: int) -> dict:
        import torch

        record = self.records[index]
        config = self.config
        rng = random.Random(f"{self.seed}:{self.split}:{self.epoch}:{index}")
        mode = config.slice_mode
        if not config.train and mode == "random":
            mode = "uniform"
        augment = None
        if config.train:
            augment = lambda images, generator: augment_slices(images, generator, config.augment)  # noqa: E731
        slices = volume_to_slices(
            record.path,
            config.k,
            config.size,
            config.min_std,
            mode=mode,
            rng=rng if mode != "uniform" else None,
            augment=augment,
        )
        return {
            "slices": torch.from_numpy(slices),
            "label": float(record.label),
            "accession": record.accession,
            "series_uid": record.series_uid,
            "group": record.group,
            "path": record.path,
        }


def collate(batch: list[dict]) -> dict:
    import torch

    return {
        "slices": torch.stack([item["slices"] for item in batch]),
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "accession": [item["accession"] for item in batch],
        "series_uid": [item["series_uid"] for item in batch],
        "group": [item["group"] for item in batch],
        "path": [item["path"] for item in batch],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="目标一训练数据体检（不训练）")
    parser.add_argument("--data-root", type=Path,
                        default=Path("/2026aicompetition/datasets/training"))
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="写出 manifest.jsonl")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--composition", choices=SPECIAL_POLICIES, default="exclude")
    parser.add_argument("--duplicate", choices=SPECIAL_POLICIES, default="exclude")
    parser.add_argument("--limit-per-class", type=int, default=0)
    args = parser.parse_args(argv)

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
    summary = summarize(records)
    summary["data_root"] = str(args.data_root)
    summary["annotation_root"] = None if annotation_root is None else str(annotation_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.out_dir:
        print(f"manifest: {write_manifest(records, args.out_dir / 'manifest.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
