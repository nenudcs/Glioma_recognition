"""Official Goal 3 competition Task."""

import torch

from pipeline.context import PipelineContext
from tasks.goal3.model import Goal3Model
from tasks.results import Goal3Result
from tasks.goal_common import _TorchTask
from tasks.base import StudyTask


class Goal3Task(_TorchTask, StudyTask[Goal3Result]):
    name = "goal3"

    def load_model(self) -> None:
        self._finish_load(Goal3Model())

    def predict(self, context: PipelineContext) -> Goal3Result:
        if self.model is None:
            raise RuntimeError("Goal3 model has not been loaded")
        with torch.inference_mode():
            probability = torch.sigmoid(self.model(self._input(context))).item()
        return Goal3Result(tumor_probability=float(probability))


TorchGoal3Task = Goal3Task
__all__ = ["Goal3Task", "TorchGoal3Task"]
