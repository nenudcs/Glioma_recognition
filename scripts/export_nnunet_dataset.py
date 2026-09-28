#!/usr/bin/env python3
"""Export one four-modality nnUNet case per AccessionNumber."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from training.study_dataset import MODALITIES, build_study_records


def materialize(source: Path, destination: Path, link: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    if link:
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def write_zero_like(reference: Path, destination: Path) -> None:
    image = nib.load(str(reference))
    zeros = np.zeros(image.shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(zeros, image.affine, image.header), str(destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="file_index.csv")
    parser.add_argument("--labels-csv", type=Path, help="series_merged.csv; used for classification labels only")
    parser.add_argument("--out-dir", type=Path, required=True, help="nnUNet dataset folder")
    parser.add_argument("--target", choices=["abnormal", "core"], default="abnormal")
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="GliomaSegmentation")
    parser.add_argument("--link", action="store_true", help="symlink instead of copying NIfTI files")
    args = parser.parse_args()

    records = build_study_records(args.index, args.labels_csv, require_all_modalities=False, require_label=False)
    images = args.out_dir / "imagesTr"
    labels = args.out_dir / "labelsTr"
    selected = []
    for record in records:
        mask = record.abnormal_mask if args.target == "abnormal" else record.core_mask
        if not mask or not Path(mask).is_file():
            continue
        case_id = f"case_{len(selected):06d}"
        reference = next((Path(path) for path in record.modalities.values() if path), None)
        if reference is None:
            continue
        for channel, modality in enumerate(MODALITIES):
            destination = images / f"{case_id}_{channel:04d}.nii.gz"
            if record.modalities.get(modality):
                materialize(Path(record.modalities[modality]), destination, args.link)
            else:
                write_zero_like(reference, destination)
        materialize(Path(mask), labels / f"{case_id}.nii.gz", args.link)
        selected.append({"case": case_id, "AccessionNumber": record.accession, "mask": args.target})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = {
        "channel_names": {str(i): modality.upper() for i, modality in enumerate(MODALITIES)},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(selected),
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
        "description": f"Four-modality study export; target={args.target}",
    }
    (args.out_dir / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "case_map.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "cases": len(selected), "target": args.target}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
