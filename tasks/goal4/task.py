"""Official Goal 4 competition Task."""

import torch
import os

from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.goal4.model import Goal4Model
from tasks.results import Goal4Result
from tasks.goal_common import _TorchTask, _goal4_result, _select_series


class Goal4Task(_TorchTask, StudyTask[Goal4Result]):
    name = "goal4"

    def load_model(self) -> None:
        self._finish_load(Goal4Model(in_channels=int(os.environ.get("GOAL4_IN_CHANNELS", "1"))))

    def predict(self, context: PipelineContext) -> Goal4Result:
        if self.model is None:
            raise RuntimeError("Goal4 model has not been loaded")
        selected = _select_series(context.study.series, ("t1ce", "flair", "t2", "t1"))
        with torch.inference_mode():
            outputs = self.model(self._input_single(selected))
        return _goal4_result(outputs)


TorchGoal4Task = Goal4Task
__all__ = ["Goal4Task", "TorchGoal4Task"]
