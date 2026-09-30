from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Series:
    """One 3-D image in original output space.

    ``image`` axis order is the order represented by ``affine``. The affine is
    always voxel-to-RAS for NIfTI output. Task preprocessing must not mutate
    this object or replace the original-space affine.
    """

    series_uid: str
    modality: str | None
    image: np.ndarray
    affine: np.ndarray
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image.ndim != 3:
            raise ValueError(
                f"series {self.series_uid!r} must be 3-D, got {self.image.shape}"
            )
        if self.affine.shape != (4, 4):
            raise ValueError("affine must be a 4x4 matrix")


@dataclass(frozen=True)
class Study:
    accession_number: str
    series: tuple[Series, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.accession_number:
            raise ValueError("accession_number cannot be empty")
        if not self.series:
            raise ValueError(f"study {self.accession_number!r} has no image series")
        uids = [item.series_uid for item in self.series]
        if len(uids) != len(set(uids)):
            raise ValueError(f"study {self.accession_number!r} has duplicate series UIDs")

    def series_by_uid(self, series_uid: str) -> Series:
        for item in self.series:
            if item.series_uid == series_uid:
                return item
        raise KeyError(f"unknown series UID {series_uid!r}")


@dataclass(frozen=True)
class CompetitionDataset:
    dataset_path: Path
    studies: tuple[Study, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.studies:
            raise ValueError("dataset has no studies")
        accessions = [study.accession_number for study in self.studies]
        if len(accessions) != len(set(accessions)):
            raise ValueError("dataset has duplicate accession numbers")

