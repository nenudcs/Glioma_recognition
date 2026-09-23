"""真实插件注册入口（规范 §5.1 / §8.1 / §15.1）。

当前接入 Goal1 authenticity 和 Goal3/4/5 模型，Goal2 stitched 与重复影像仍使用 Dummy。
执行顺序按规范 §8.1 推荐顺序：Goal1 → Goal2 stitched → Goal3 → Goal5 → Goal4。

用法（不改管线代码，只用环境变量注入）::

    export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
    export GOAL1_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal1_authenticity/model.pt
    export GOAL3_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal3.pt
    export GOAL4_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal4.pt
    export GOAL5_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal5.pt
    ./start.sh

若 ``core/config.py`` 将来提供 checkpoint 根目录（规范 §5.2），
``Goal1Config`` 会自动优先使用它，本文件无需修改。
"""
from __future__ import annotations

import os
from pathlib import Path

from pipeline.inference import InferencePipeline, StudyTaskBinding
from tasks.dummy.dataset_tasks import DummyDuplicateTask
from tasks.dummy.study_tasks import (
    DummyStitchedTask,
)
from tasks.goal1_authenticity.config import Goal1Config
from tasks.goal1_authenticity.task import Goal1AuthenticityTask
from tasks.goal3.task import Goal3Task
from tasks.goal4.task import Goal4Task
from tasks.goal5.task import Goal5Task


def build_pipeline(goal1_config: Goal1Config | None = None) -> InferencePipeline:
    """规范顺序的完整管线：Goal1/3/4/5 真实 Task，其余保留 Dummy。"""
    checkpoint_root = Path(
        os.environ.get(
            "COMPETITION_CHECKPOINT_ROOT",
            "/2026aicompetition/workspace/checkpoint",
        )
    )
    return InferencePipeline(
        study_tasks=(
            StudyTaskBinding("goal1", Goal1AuthenticityTask(goal1_config)),
            StudyTaskBinding("goal2_stitched", DummyStitchedTask()),
            StudyTaskBinding("goal3", Goal3Task(_checkpoint("GOAL3_CHECKPOINT", checkpoint_root / "goal3.pt"))),
            StudyTaskBinding("goal5", Goal5Task(_checkpoint("GOAL5_CHECKPOINT", checkpoint_root / "goal5.pt"))),
            StudyTaskBinding("goal4", Goal4Task(_checkpoint("GOAL4_CHECKPOINT", checkpoint_root / "goal4.pt"))),
        ),
        duplicate_task=DummyDuplicateTask(),
    )


def _checkpoint(variable: str, default: Path) -> str | None:
    configured = os.environ.get(variable)
    if configured:
        return configured
    return str(default) if default.is_file() else None
