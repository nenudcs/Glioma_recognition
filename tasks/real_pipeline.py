"""真实插件注册入口（规范 §5.1 / §8.1 / §15.1）。

当前已接入 **Goal1 authenticity** 与 **Goal2（拼接 + 重复）**，其余目标继续使用 Dummy
（替换策略：一次只替换一个插件，替换后跑完整回归）。执行顺序按规范 §8.1 推荐顺序：
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
from tasks.dummy.study_tasks import (
    DummyGoal3Task,
    DummyGoal4Task,
    DummyGoal5Task,
)
from tasks.goal1_authenticity.config import Goal1Config
from tasks.goal1_authenticity.task import Goal1AuthenticityTask
from tasks.goal2_duplicate.config import Goal2DuplicateConfig
from tasks.goal2_duplicate.inference import DuplicateProbe
from tasks.goal2_duplicate.task import Goal2DuplicateRecorder
from tasks.goal2_stitched.config import Goal2StitchedConfig
from tasks.goal2_stitched.gating import GatedStudyTask, gated_fields
from tasks.goal2_stitched.task import Goal2StitchedTask


def build_pipeline(
    goal1_config: Goal1Config | None = None,
    goal2_config: Goal2StitchedConfig | None = None,
    duplicate_config: Goal2DuplicateConfig | None = None,
    *,
    gated_fields_override: tuple[str, ...] | None = None,
) -> InferencePipeline:
    """规范顺序的完整管线：真实 Goal1/Goal2 + 其余 Dummy 占位。"""
    duplicate = duplicate_config or Goal2DuplicateConfig.from_env()
    fields = gated_fields() if gated_fields_override is None else tuple(gated_fields_override)
    probe = DuplicateProbe(duplicate) if duplicate.enabled else None

    goal3 = DummyGoal3Task()
    goal4 = DummyGoal4Task()
    goal5 = DummyGoal5Task()
    return InferencePipeline(
        study_tasks=(
            StudyTaskBinding("goal1", Goal1AuthenticityTask(goal1_config)),
            StudyTaskBinding(
                "goal2_stitched",
                Goal2StitchedTask(goal2_config, duplicate_probe=probe),
            ),
            StudyTaskBinding(
                "goal3",
                GatedStudyTask(goal3, "goal3", goal3.predict, gated_fields=fields),
            ),
            StudyTaskBinding(
                "goal5",
                GatedStudyTask(goal5, "goal5", goal5.predict, gated_fields=fields),
            ),
            StudyTaskBinding(
                "goal4",
                GatedStudyTask(goal4, "goal4", goal4.predict, gated_fields=fields),
            ),
        ),
        duplicate_task=Goal2DuplicateRecorder(duplicate, probe=probe),
    )
