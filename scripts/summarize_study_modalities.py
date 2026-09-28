#!/usr/bin/env python3
"""Count available T1/T1CE/T2/FLAIR combinations by AccessionNumber."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from training.study_dataset import MODALITIES, canonical_modality


def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="file_index.csv")
    parser.add_argument("--out", type=Path, default=Path("study_modalities.json"))
    args = parser.parse_args()
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in read_rows(args.index):
        if row.get("status") not in {"series_and_mask_found", "series_found_mask_missing"}:
            continue
        modality = canonical_modality(row.get("SeriesType", ""))
        accession = row.get("AccessionNumber", "")
        if accession and modality:
            grouped[accession].add(modality)
    combos = Counter("+".join(modality for modality in MODALITIES if modality in values) for values in grouped.values())
    summary = {
        "studies": len(grouped),
        "combination_counts": dict(sorted(combos.items(), key=lambda item: (-item[1], item[0]))),
        "per_modality": {modality: sum(modality in values for values in grouped.values()) for modality in MODALITIES},
        "complete_four_modality": sum(all(modality in values for modality in MODALITIES) for values in grouped.values()),
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
