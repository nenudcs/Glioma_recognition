"""Official Goal 5 competition Task."""

import torch

from pipeline.context import PipelineContext
from tasks.base import StudyTask
from tasks.goal5.model import Goal5Model
from tasks.results import Goal5Result
from tasks.goal_common import _TorchTask, _restore_mask, _select_series


class Goal5Task(_TorchTask, StudyTask[Goal5Result]):
    name = "goal5"

    def load_model(self) -> None:
        self._finish_load(Goal5Model())

    def predict(self, context: PipelineContext) -> Goal5Result:
        if self.model is None:
            raise RuntimeError("Goal5 model has not been loaded")
        t1ce = _select_series(context.study.series, ("t1ce", "t1+c", "t1 enhanced", "t1"))
        flair = _select_series(context.study.series, ("flair", "t2flair", "t2"))
        with torch.inference_mode():
            logits = self.model(self._input(context))
            masks = (torch.sigmoid(logits) >= 0.5).to(torch.uint8).cpu().numpy()[0]
        return Goal5Result(
            core_mask=_restore_mask(masks[0], t1ce.image.shape),
            core_source_series_uid=t1ce.series_uid,
            flair_mask=_restore_mask(masks[1], flair.image.shape),
            flair_source_series_uid=flair.series_uid,
        )


TorchGoal5Task = Goal5Task
__all__ = ["Goal5Task", "TorchGoal5Task"]
