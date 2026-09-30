from __future__ import annotations

import numpy as np

from core.exceptions import MissingSeriesError
from data.structures import Series
from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.results import (
    BinaryResult,
    CategoricalResult,
    Goal1Result,
    Goal3Result,
    Goal4Result,
    Goal5Result,
    StitchedResult,
)


class DummyGoal1Task(StudyTask[Goal1Result]):
    name = "goal1"

    def predict(self, context: PipelineContext) -> Goal1Result:
        return Goal1Result(not_human_probability=0.0)


class DummyStitchedTask(StudyTask[StitchedResult]):
    name = "goal2_stitched"

    def predict(self, context: PipelineContext) -> StitchedResult:
        return StitchedResult(stitched_probability=0.0)


class DummyGoal3Task(StudyTask[Goal3Result]):
    name = "goal3"

    def predict(self, context: PipelineContext) -> Goal3Result:
        return Goal3Result(tumor_probability=0.0)


class DummyGoal4Task(StudyTask[Goal4Result]):
    name = "goal4"

    def predict(self, context: PipelineContext) -> Goal4Result:
        negative = BinaryResult(present=False, probability=0.0)
        return Goal4Result(
            location="Other",
            morphology=CategoricalResult(
                predicted="Regular",
                probabilities={"Regular": 1.0, "Irregular": 0.0},
            ),
            who_grade=CategoricalResult(
                predicted=None,
                probabilities={"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0},
            ),
            enhancement=negative,
            enhancement_pattern=CategoricalResult(
                predicted="None",
                probabilities={
                    "None": 1.0,
                    "Ring": 0.0,
                    "RimEnhancing": 0.0,
                    "Nodular": 0.0,
                    "GroundGlass": 0.0,
                    "Gyriform": 0.0,
                    "Multifocal": 0.0,
                    "Other": 0.0,
                },
            ),
            necrosis=negative,
            cystic_change=negative,
            hemorrhage=negative,
            calcification=negative,
            margin_clear=negative,
            lobulation=negative,
            signal_t2wi=CategoricalResult(
                predicted="Iso",
                probabilities={"Low": 0.0, "Iso": 1.0, "High": 0.0},
            ),
            signal_flair=CategoricalResult(
                predicted="Iso",
                probabilities={"Low": 0.0, "Iso": 1.0, "High": 0.0},
            ),
            conclusion="Dummy baseline: no diagnostic conclusion.",
        )


class DummyGoal5Task(StudyTask[Goal5Result]):
    name = "goal5"

    def predict(self, context: PipelineContext) -> Goal5Result:
        if not context.study.series:
            raise MissingSeriesError(
                f"study {context.study.accession_number!r} has no series"
            )
        core_source = _select_series(context.study.series, ("t1ce", "t1+c", "t1 enhanced", "t1"))
        flair_source = _select_series(context.study.series, ("flair", "t2flair", "t2"))
        return Goal5Result(
            core_mask=np.zeros(core_source.image.shape, dtype=np.uint8),
            core_source_series_uid=core_source.series_uid,
            flair_mask=np.zeros(flair_source.image.shape, dtype=np.uint8),
            flair_source_series_uid=flair_source.series_uid,
        )


def _select_series(series: tuple[Series, ...], hints: tuple[str, ...]) -> Series:
    for hint in hints:
        for item in series:
            description = " ".join(
                (
                    item.modality or "",
                    str(item.metadata.get("SeriesDescription", "")),
                    item.series_uid,
                )
            ).lower()
            if hint in description:
                return item
    return series[0]

