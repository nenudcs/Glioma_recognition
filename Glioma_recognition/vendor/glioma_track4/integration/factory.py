"""团队 Pipeline 工厂入口（``COMPETITION_PIPELINE_FACTORY`` 指向本模块）。

团队侧只需设置：

```bash
export COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline
```

或直接指向本模块（要求本工程在 ``sys.path`` 上）：

```bash
export COMPETITION_PIPELINE_FACTORY=integration.factory:build_pipeline
```

绑定顺序与团队 Dummy 基线**完全一致**：goal1 → goal2_stitched → goal3 → goal5 → goal4，
另加 dataset 级 goal2_duplicate。

``GLIOMA_GOALS`` 控制启用哪些真实插件（未启用的用团队 Dummy 补位），便于按规范
"每次只替换一个插件并跑完整回归"。
"""
from __future__ import annotations

import os

#: 默认全部启用
DEFAULT_GOALS = "goal1,goal2_stitched,goal3,goal5,goal4,goal2_duplicate"


def build_pipeline():
    """构建团队 ``InferencePipeline``（真实插件 + Dummy 补位）。"""
    from pipeline.inference import InferencePipeline, StudyTaskBinding

    from tasks.dummy.dataset_tasks import DummyDuplicateTask
    from tasks.dummy.study_tasks import (
        DummyGoal1Task,
        DummyGoal3Task,
        DummyGoal4Task,
        DummyGoal5Task,
        DummyStitchedTask,
    )

    from .tasks import (
        AuthenticityTask,
        DiagnosisTask,
        DuplicateTask,
        SegmentationTask,
        StitchedTask,
        TumorTask,
    )

    enabled = {g.strip() for g in
               os.environ.get("GLIOMA_GOALS", DEFAULT_GOALS).split(",") if g.strip()}

    def pick(key: str, real, dummy):
        return real() if key in enabled else dummy()

    study_tasks = (
        StudyTaskBinding("goal1", pick("goal1", AuthenticityTask, DummyGoal1Task)),
        StudyTaskBinding("goal2_stitched", pick("goal2_stitched", StitchedTask, DummyStitchedTask)),
        StudyTaskBinding("goal3", pick("goal3", TumorTask, DummyGoal3Task)),
        StudyTaskBinding("goal5", pick("goal5", SegmentationTask, DummyGoal5Task)),
        StudyTaskBinding("goal4", pick("goal4", DiagnosisTask, DummyGoal4Task)),
    )
    duplicate_task = pick("goal2_duplicate", DuplicateTask, DummyDuplicateTask)
    return InferencePipeline(study_tasks=study_tasks, duplicate_task=duplicate_task)
