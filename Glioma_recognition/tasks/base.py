from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from data.structures import Study

if False:  # pragma: no cover - imports for type checkers without a cycle
    from pipeline.context import PipelineContext


ResultT = TypeVar("ResultT")


class StudyTask(ABC, Generic[ResultT]):
    name: str

    def load_model(self) -> None:
        """Load weights once. Dummy and stateless tasks may do nothing."""

    @abstractmethod
    def predict(self, context: "PipelineContext") -> ResultT:
        raise NotImplementedError


class DatasetTask(ABC, Generic[ResultT]):
    name: str

    def load_model(self) -> None:
        """Load weights once. Dummy and stateless tasks may do nothing."""

    def reset(self) -> None:
        """Start a new evaluation and discard state from the previous one."""

    @abstractmethod
    def update(
        self,
        study: Study,
        context: "PipelineContext",
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> ResultT:
        raise NotImplementedError
