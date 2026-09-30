from __future__ import annotations

from data.structures import Study
from pipeline.context import PipelineContext
from tasks.base import DatasetTask
from tasks.results import DuplicatePair, DuplicateResult


class DummyDuplicateTask(DatasetTask[DuplicateResult]):
    name = "goal2_duplicate"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._accessions: list[str] = []

    def update(self, study: Study, context: PipelineContext) -> None:
        self._accessions.append(study.accession_number)
        self._accessions = sorted(self._accessions)[:2]

    def finalize(self) -> DuplicateResult:
        if len(self._accessions) < 2:
            return DuplicateResult(pairs=())
        return DuplicateResult(
            pairs=(
                DuplicatePair(
                    left_accession=self._accessions[0],
                    right_accession=self._accessions[1],
                    probability=0.0,
                ),
            )
        )
