#!/usr/bin/env python3
"""Lazy PyTorch Dataset/DataLoader for the generated file index.

The input should be file_index.csv from scan_training_files.py.  It contains
one row per SeriesType sample and paths discovered without loading pixels.
NIfTI arrays are loaded only inside __getitem__, so the whole dataset is not
held in memory.  Use batch_size=1 unless all volumes have been resampled to a
common shape.

Examples:
    python brain_nifti_dataloader.py \
        --index file_check/file_index.csv \
        --task classification --max-samples 2

    python brain_nifti_dataloader.py \
        --index file_check/file_index.csv \
        --task segmentation --mask-name 水肿 --max-samples 2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Iterable

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from training.resize import resize_volume


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def normalize_mask_name(name: str) -> str:
    return name.strip().replace("_", "")


def mask_label_from_path(path: Path) -> str:
    name = path.name
    suffix = "_mask.nii.gz"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    # Remove the numeric component used by names such as 水肿_2_mask.nii.gz.
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]
    return name


def find_mask_paths(series_dir: Path) -> list[Path]:
    return sorted(
        path for path in series_dir.glob("*.nii.gz") if path.name.endswith("_mask.nii.gz")
    )


def choose_mask_path(mask_paths: Iterable[Path], requested: str) -> Path | None:
    paths = list(mask_paths)
    if not paths:
        return None
    if requested == "any":
        return paths[0]
    priority_options = {
        "t2_auto": ["水肿", "全肿瘤", "瘤体"],
        "t1ce_auto": ["肿瘤瘤体", "瘤体", "全肿瘤"],
    }
    requested_norm = normalize_mask_name(requested)
    names = {
        path: normalize_mask_name(mask_label_from_path(path))
        for path in paths
    }
    if requested in priority_options:
        for candidate in priority_options[requested]:
            candidate_norm = normalize_mask_name(candidate)
            exact = [path for path, name in names.items() if name == candidate_norm]
            if exact:
                return exact[0]
        return None
    exact = [path for path, name in names.items() if name == requested_norm]
    return exact[0] if exact else None


def load_nifti(path: Path, *, is_mask: bool = False) -> torch.Tensor:
    image = nib.load(str(path))
    # get_fdata is called per sample, not during Dataset construction.
    array = np.asarray(image.dataobj, dtype=np.float32)
    array = np.nan_to_num(array, copy=False)
    if is_mask:
        array = (array > 0).astype(np.float32, copy=False)
    # Add a channel dimension. The remaining dimensions preserve NIfTI order.
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


class BrainNiftiDataset(Dataset):
    """One row per SeriesType sample, with lazy image and mask loading."""

    def __init__(
        self,
        index_csv: str | Path,
        *,
        task: str = "classification",
        mask_name: str = "any",
        label_column: str | None = None,
        labels_csv: str | Path | None = None,
        input_channels: int = 1,
        target_shape: tuple[int, int, int] | None = None,
        require_label: bool = False,
        require_mask: bool | None = None,
    ) -> None:
        self.index_csv = Path(index_csv)
        self.task = task
        self.mask_name = mask_name
        self.label_column = label_column
        self.require_label = require_label
        if input_channels < 1:
            raise ValueError("input_channels must be >= 1")
        self.input_channels = input_channels
        self.target_shape = target_shape
        if require_mask is None:
            require_mask = task == "segmentation"
        self.require_mask = require_mask

        rows = read_csv(self.index_csv)
        if labels_csv is not None:
            label_rows = read_csv(Path(labels_csv))
            label_map = {
                (r.get("AccessionNumber", ""), r.get("SeriesUid", "")): r
                for r in label_rows
            }
            for row in rows:
                source = label_map.get(
                    (row.get("AccessionNumber", ""), row.get("SeriesUid", "")),
                    {},
                )
                for key, value in source.items():
                    row.setdefault(key, value)
        filtered: list[dict[str, str]] = []
        for row in rows:
            series_path = Path(row.get("series_path", ""))
            series_exists = as_bool(row.get("series_file_exists", "")) and series_path.is_file()
            if not series_exists:
                continue
            mask_paths = find_mask_paths(Path(row.get("series_dir", "")))
            selected_mask = choose_mask_path(mask_paths, mask_name) if self.require_mask else None
            if self.require_mask and selected_mask is None:
                continue
            if self.require_label and (
                label_column is None or not row.get(label_column, "").strip()
            ):
                continue
            row["_selected_mask_path"] = str(selected_mask) if selected_mask else ""
            filtered.append(row)
        self.rows = filtered

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["series_path"])
        image = load_nifti(image_path)
        if self.input_channels > 1:
            image = image.repeat(self.input_channels, 1, 1, 1)
        if self.target_shape is not None:
            image = resize_volume(image, self.target_shape, is_mask=False)
        item: dict[str, Any] = {
            "image": image,
            "AccessionNumber": row.get("AccessionNumber", ""),
            "SeriesUid": row.get("SeriesUid", ""),
            "SeriesType": row.get("SeriesType", ""),
            "image_path": str(image_path),
        }
        if row.get("_selected_mask_path", ""):
            mask_path = Path(row["_selected_mask_path"])
            item["mask"] = load_nifti(mask_path, is_mask=True)
            if self.target_shape is not None:
                item["mask"] = resize_volume(item["mask"], self.target_shape, is_mask=True)
            item["mask_path"] = str(mask_path)
        if self.label_column:
            item["label"] = row.get(self.label_column, "")
        return item


def build_dataloader(
    index_csv: str | Path,
    *,
    task: str = "classification",
    mask_name: str = "any",
    label_column: str | None = None,
    labels_csv: str | Path | None = None,
    require_label: bool = False,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    target_shape: tuple[int, int, int] | None = None,
) -> tuple[BrainNiftiDataset, DataLoader]:
    dataset = BrainNiftiDataset(
        index_csv,
        task=task,
        mask_name=mask_name,
        label_column=label_column,
        labels_csv=labels_csv,
        target_shape=target_shape,
        require_label=require_label,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="file_index.csv from scan_training_files.py")
    parser.add_argument("--task", choices=["classification", "segmentation"], default="classification")
    parser.add_argument(
        "--mask-name",
        default="any",
        help="any, 水肿, 瘤体, 肿瘤瘤体, 全肿瘤, t2_auto, or t1ce_auto",
    )
    parser.add_argument("--label-column", help="Optional label column already present in file_index.csv")
    parser.add_argument("--labels-csv", type=Path, help="Optional series_merged.csv containing label columns")
    parser.add_argument("--require-label", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max-samples", type=int, default=1)
    args = parser.parse_args()

    dataset, loader = build_dataloader(
        args.index,
        task=args.task,
        mask_name=args.mask_name,
        label_column=args.label_column,
        labels_csv=args.labels_csv,
        require_label=args.require_label,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
    )
    print({"dataset_length": len(dataset), "task": args.task, "mask_name": args.mask_name})
    for count, batch in enumerate(loader):
        print(
            {
                "batch": count,
                "image_shape": tuple(batch["image"].shape),
                "has_mask": "mask" in batch,
                "mask_shape": tuple(batch["mask"].shape) if "mask" in batch else None,
                "AccessionNumber": batch["AccessionNumber"],
                "SeriesUid": batch["SeriesUid"],
            }
        )
        if count + 1 >= args.max_samples:
            break


if __name__ == "__main__":
    main()
