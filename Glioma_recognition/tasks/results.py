from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Goal1Result:
    not_human_probability: float


@dataclass(frozen=True)
class StitchedResult:
    stitched_probability: float


@dataclass(frozen=True)
class Goal3Result:
    tumor_probability: float


@dataclass(frozen=True)
class CategoricalResult:
    predicted: str | int | None
    probabilities: dict[str, float]


@dataclass(frozen=True)
class BinaryResult:
    present: bool
    probability: float


@dataclass(frozen=True)
class Goal4Result:
    location: str
    morphology: CategoricalResult
    who_grade: CategoricalResult
    enhancement: BinaryResult
    enhancement_pattern: CategoricalResult
    necrosis: BinaryResult
    cystic_change: BinaryResult
    hemorrhage: BinaryResult
    calcification: BinaryResult
    margin_clear: BinaryResult
    lobulation: BinaryResult
    signal_t2wi: CategoricalResult
    signal_flair: CategoricalResult
    conclusion: str
    attention_map_uri: str | None = None


@dataclass(frozen=True)
class Goal5Result:
    core_mask: np.ndarray
    core_source_series_uid: str
    flair_mask: np.ndarray
    flair_source_series_uid: str


@dataclass(frozen=True)
class DuplicatePair:
    left_accession: str
    right_accession: str
    probability: float


@dataclass(frozen=True)
class DuplicateResult:
    pairs: tuple[DuplicatePair, ...]

