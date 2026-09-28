#!/usr/bin/env python3
"""Check image and mask files for the SeriesType sample index.

Expected layout:
    DATA_ROOT/<AccessionNumber>/<SeriesUid>/<SeriesUid>.nii.gz
    DATA_ROOT/<AccessionNumber>/<SeriesUid>/<ROI>_<number>_mask.nii.gz

This script only checks paths and names. It does not load NIfTI pixels.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scan(index_path: Path, data_root: Path, output_dir: Path) -> dict[str, Any]:
    rows = read_index(index_path)
    if not data_root.exists():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    result: list[dict[str, Any]] = []
    mask_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    for row in rows:
        accession = row.get("AccessionNumber", "")
        series_uid = row.get("SeriesUid", "")
        series_dir = data_root / accession / series_uid
        all_nii = sorted(series_dir.glob("*.nii.gz")) if series_dir.is_dir() else []
        mask_paths = [p for p in all_nii if p.name.endswith("_mask.nii.gz")]
        series_candidates = [p for p in all_nii if not p.name.endswith("_mask.nii.gz")]
        series_path = series_candidates[0] if len(series_candidates) == 1 else None
        mask_names = []
        for path in mask_paths:
            match = re.match(r"^(.*?)(?:_\d+)?_mask\.nii\.gz$", path.name)
            name = match.group(1) if match else path.name
            mask_names.append(name)
            mask_counter[name] += 1

        if not accession or not series_uid:
            status = "invalid_key"
        elif not series_dir.is_dir():
            status = "series_directory_missing"
        elif not series_candidates:
            status = "series_file_missing"
        elif len(series_candidates) > 1:
            status = "series_file_ambiguous"
        elif not mask_paths:
            status = "series_found_mask_missing"
        else:
            status = "series_and_mask_found"
        status_counter[status] += 1
        result.append(
            {
                "AccessionNumber": accession,
                "SeriesUid": series_uid,
                "SeriesType": row.get("SeriesType", ""),
                "has_check_label": row.get("has_check_label", ""),
                "has_series_label": row.get("has_series_label", ""),
                "has_roi_label": row.get("has_roi_label", ""),
                "series_dir": str(series_dir),
                "series_path": str(series_path) if series_path is not None else "",
                "series_file_exists": str(series_path is not None).lower(),
                "mask_count": len(mask_paths),
                "mask_names": "|".join(mask_names),
                "status": status,
            }
        )

    write_csv(output_dir / "file_index.csv", result)
    missing = [row for row in result if row["status"] not in {"series_and_mask_found"}]
    write_csv(output_dir / "file_missing_or_incomplete.csv", missing)
    summary = {
        "index": str(index_path),
        "data_root": str(data_root),
        "base_rows": len(result),
        "status_counts": dict(status_counter),
        "series_file_found": sum(bool(row["series_file_exists"] == "true") for row in result),
        "series_file_missing": sum(row["status"] == "series_file_missing" for row in result),
        "series_file_ambiguous": sum(row["status"] == "series_file_ambiguous" for row in result),
        "mask_file_total": sum(int(row["mask_count"]) for row in result),
        "mask_any_found": sum(int(row["mask_count"]) > 0 for row in result),
        "mask_missing": sum(int(row["mask_count"]) == 0 for row in result),
        "mask_name_counts": dict(mask_counter),
        "notes": [
            "This is a path/name check; NIfTI pixels were not loaded.",
            "A missing file is distinct from an annotation key missing in Excel.",
            "All base rows remain in file_index.csv, including incomplete rows.",
        ],
    }
    (output_dir / "file_qc.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="sample_index.csv from read_training_annotations.py")
    parser.add_argument("--data-root", type=Path, required=True, help="training directory containing accession/series folders")
    parser.add_argument("--out-dir", type=Path, default=Path("file_check"))
    args = parser.parse_args()
    print(json.dumps(scan(args.index, args.data_root, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
