"""Official Goal 4 competition Task."""

import torch

from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.goal4.model import Goal4Model
from tasks.results import Goal4Result
from tasks.goal_common import _TorchTask, _goal4_result


class Goal4Task(_TorchTask, StudyTask[Goal4Result]):
    name = "goal4"

    def load_model(self) -> None:
        self._finish_load(Goal4Model())

    def predict(self, context: PipelineContext) -> Goal4Result:
        if self.model is None:
            raise RuntimeError("Goal4 model has not been loaded")
        with torch.inference_mode():
            outputs = self.model(self._input(context))
        return _goal4_result(outputs)


TorchGoal4Task = Goal4Task
__all__ = ["Goal4Task", "TorchGoal4Task"]
