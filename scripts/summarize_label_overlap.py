#!/usr/bin/env python3
"""Summarize label availability for every SeriesType sample.

Input is sample_index.csv produced by read_training_annotations.py.  The
script counts the three annotation layers independently and by intersection:
check-level, sequence-level, and ROI-level.  It does not count raw ROI rows as
samples; every count is based on one AccessionNumber + SeriesUid row.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


LAYERS = ("check", "series", "roi")


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else ["combination", "count"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    for row in rows:
        for layer in LAYERS:
            row[f"has_{layer}_label"] = truthy(row.get(f"has_{layer}_label", ""))
        row["label_count"] = sum(bool(row[f"has_{layer}_label"]) for layer in LAYERS)

    combo_counter: Counter[str] = Counter()
    count_counter: Counter[str] = Counter()
    for row in rows:
        present = [layer for layer in LAYERS if row[f"has_{layer}_label"]]
        missing = [layer for layer in LAYERS if not row[f"has_{layer}_label"]]
        combo_counter["+".join(present) if present else "none"] += 1
        count_counter[str(len(present))] += 1

    return {
        "base_samples": len(rows),
        "layer_counts": {
            layer: sum(bool(row[f"has_{layer}_label"]) for row in rows)
            for layer in LAYERS
        },
        "missing_by_layer": {
            layer: sum(not bool(row[f"has_{layer}_label"]) for row in rows)
            for layer in LAYERS
        },
        "by_number_of_layers": {
            "all_three": count_counter["3"],
            "exactly_two": count_counter["2"],
            "exactly_one": count_counter["1"],
            "none": count_counter["0"],
        },
        "by_combination": dict(combo_counter),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="sample_index.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("label_overlap"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.index)
    summary = summarize(rows)
    (args.out_dir / "label_overlap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(
        args.out_dir / "label_overlap_samples.csv",
        [
            {
                "AccessionNumber": row.get("AccessionNumber", ""),
                "SeriesUid": row.get("SeriesUid", ""),
                "has_check_label": row["has_check_label"],
                "has_series_label": row["has_series_label"],
                "has_roi_label": row["has_roi_label"],
                "label_count": row["label_count"],
            }
            for row in rows
        ],
    )
    combo_rows = [
        {"combination": key, "count": value}
        for key, value in sorted(summary["by_combination"].items())
    ]
    write_csv(args.out_dir / "label_overlap_combinations.csv", combo_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
