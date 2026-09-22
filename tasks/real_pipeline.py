"""真实插件注册入口（规范 §5.1 / §8.1 / §15.1）。

当前只接入 **Goal1 authenticity**，其余目标继续使用 Dummy（替换策略：一次只替换一个插件，
替换后跑完整回归）。执行顺序按规范 §8.1 推荐顺序：Goal1 → Goal2 stitched → Goal3 → Goal5 → Goal4。

用法（不改管线代码，只用环境变量注入）::

    export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
    export GOAL1_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal1_authenticity/model.pt
    ./start.sh

若 ``core/config.py`` 将来提供 checkpoint 根目录（规范 §5.2），
``Goal1Config`` 会自动优先使用它，本文件无需修改。
"""
from __future__ import annotations

from pipeline.inference import InferencePipeline, StudyTaskBinding
from tasks.dummy.dataset_tasks import DummyDuplicateTask
from tasks.dummy.study_tasks import (
    DummyGoal3Task,
    DummyGoal4Task,
    DummyGoal5Task,
    DummyStitchedTask,
)
from tasks.goal1_authenticity.config import Goal1Config
from tasks.goal1_authenticity.task import Goal1AuthenticityTask


def build_pipeline(goal1_config: Goal1Config | None = None) -> InferencePipeline:
    """规范顺序的完整管线：真实 Goal1 + 其余 Dummy 占位。"""
    return InferencePipeline(
        study_tasks=(
            StudyTaskBinding("goal1", Goal1AuthenticityTask(goal1_config)),
            StudyTaskBinding("goal2_stitched", DummyStitchedTask()),
            StudyTaskBinding("goal3", DummyGoal3Task()),
            StudyTaskBinding("goal5", DummyGoal5Task()),
            StudyTaskBinding("goal4", DummyGoal4Task()),
        ),
        duplicate_task=DummyDuplicateTask(),
    )
