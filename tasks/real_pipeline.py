"""真实插件注册入口（规范 §5.1 / §8.1 / §15.1）。

当前已接入 Goal1、Goal2、Goal3、Goal4、Goal5。执行顺序按规范 §8.1 推荐顺序：
Goal1 → Goal2 stitched → Goal3 → Goal5 → Goal4。

用法（不改管线代码，只用环境变量注入）::

    export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
    export GOAL1_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal1_authenticity/model.pt
    export GOAL2_STITCHED_THRESHOLD=<标定结果>
    export GOAL2_DUPLICATE_SIM_CENTER=<标定结果>
    ./start.sh

两条任务二检测都在 ``goal2_stitched`` 这个检查级位置完成（重复检测的逐例探针由
``Goal2StitchedTask`` 携带），因此在写 ``prediction.json`` 之前就能决定是否对下游
``goal3/goal4/goal5`` 上闸；成对结果由数据集级 ``Goal2DuplicateRecorder`` 汇总。

若 ``core/config.py`` 将来提供 checkpoint 根目录（规范 §5.2），
``Goal1Config`` 会自动优先使用它，本文件无需修改。
"""
from __future__ import annotations

from pipeline.inference import InferencePipeline, StudyTaskBinding
import numpy as np

from tasks.dummy.study_tasks import DummyGoal4Task
from tasks.goal1_authenticity.config import Goal1Config
from tasks.goal1_authenticity.task import Goal1AuthenticityTask
from tasks.goal2_duplicate.config import Goal2DuplicateConfig
from tasks.goal2_duplicate.inference import DuplicateProbe
from tasks.goal2_duplicate.task import Goal2DuplicateRecorder
from tasks.goal2_stitched.config import Goal2StitchedConfig
from tasks.goal2_stitched.gating import GatedStudyTask, gated_fields
from tasks.goal2_stitched.task import Goal2StitchedTask
from tasks.goal3.task import Goal3Task
from tasks.goal4.task import Goal4Task
from tasks.goal5.task import Goal5Task
from tasks.results import Goal3Result, Goal5Result


def _neutral_goal3(_context) -> Goal3Result:
    """Neutral value used when Goal2's gate skips Goal3."""
    return Goal3Result(tumor_probability=0.5)


def _neutral_goal4(context):
    """Reuse the format-defined Goal4 neutral result without running a model."""
    return DummyGoal4Task().predict(context)


def _neutral_goal5(context) -> Goal5Result:
    """Emit binary zero masks in the exact source-series geometries."""
    from tasks.goal_common import _select_series

    core = _select_series(context.study.series, ("t1ce", "t1+c", "t1 enhanced", "t1"))
    flair = _select_series(context.study.series, ("flair", "t2flair", "t2"))
    return Goal5Result(
        core_mask=np.zeros(core.image.shape, dtype=np.uint8),
        core_source_series_uid=core.series_uid,
        flair_mask=np.zeros(flair.image.shape, dtype=np.uint8),
        flair_source_series_uid=flair.series_uid,
    )


def build_pipeline(
    goal1_config: Goal1Config | None = None,
    goal2_config: Goal2StitchedConfig | None = None,
    duplicate_config: Goal2DuplicateConfig | None = None,
    *,
    gated_fields_override: tuple[str, ...] | None = None,
) -> InferencePipeline:
    """Build the complete runtime pipeline from trained model adapters."""
    duplicate = duplicate_config or Goal2DuplicateConfig.from_env()
    fields = gated_fields() if gated_fields_override is None else tuple(gated_fields_override)
    probe = DuplicateProbe(duplicate) if duplicate.enabled else None

    goal3 = Goal3Task()
    goal4 = Goal4Task()
    goal5 = Goal5Task()
    return InferencePipeline(
        study_tasks=(
            StudyTaskBinding("goal1", Goal1AuthenticityTask(goal1_config)),
            StudyTaskBinding(
                "goal2_stitched",
                Goal2StitchedTask(goal2_config, duplicate_probe=probe),
            ),
            StudyTaskBinding(
                "goal3",
                GatedStudyTask(goal3, "goal3", _neutral_goal3, gated_fields=fields),
            ),
            StudyTaskBinding(
                "goal5",
                GatedStudyTask(goal5, "goal5", _neutral_goal5, gated_fields=fields),
            ),
            StudyTaskBinding(
                "goal4",
                GatedStudyTask(goal4, "goal4", _neutral_goal4, gated_fields=fields),
            ),
        ),
        duplicate_task=Goal2DuplicateRecorder(duplicate, probe=probe),
    )
