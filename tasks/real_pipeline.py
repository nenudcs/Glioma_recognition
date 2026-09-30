"""真实插件注册入口（规范 §5.1 / §8.1 / §15.1）。

**全真实管线**：Goal1 / Goal2（官方实现）+ Goal3 / Goal5 / Goal4（本队实现），
执行顺序按规范 §8.1 推荐：Goal1 → Goal2 stitched → Goal3 → Goal5 → Goal4。

用法（不改管线代码，只用环境变量注入）::

    export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
    export GOAL1_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal1_authenticity/model.pt
    export GOAL2_STITCHED_THRESHOLD=<标定结果>
    export GOAL2_DUPLICATE_SIM_CENTER=<标定结果>
    export COMPETITION_CHECKPOINT_ROOT=/2026aicompetition/workspace/checkpoint  # Goal3/4/5 权重根
    ./start.sh

Goal3/4/5（本队实现）的说明：

- 共享一个多任务骨干（``tasks/_common/backbone_runner.py``）：同一 Study 的
  骨干前向**只算一次**，三个头复用结果（若各 Goal 各自前向会白白慢 3 倍）；
- 权重按 ``$COMPETITION_CHECKPOINT_ROOT`` 下的相对路径解析：
  ``goal3_tumor/model.pt`` / ``goal4_diagnosis/model.pt`` /
  ``goal5_segmentation/{core,flair}.pt``（与训练侧导出约定一致）；
- 模态识别三层兜底（关键词 → 官方 ``SeriesType.xlsx`` → 体素统计判别），
  见 ``data/series_selector.py`` 与 ``data/voxel_modality.py``；
- ``GLIOMA_GOALS`` 可临时把某些 Goal 退回 Dummy（逗号分隔白名单，默认全真实），
  用于规范 §15.1「每次只替换一个插件」的对照实验。

两条任务二检测都在 ``goal2_stitched`` 这个检查级位置完成（重复检测的逐例探针由
``Goal2StitchedTask`` 携带），因此在写 ``prediction.json`` 之前就能决定是否对下游
``goal3/goal4/goal5`` 上闸；成对结果由数据集级 ``Goal2DuplicateRecorder`` 汇总。
``core/config.py`` 提供 ``checkpoint_root``（规范 §5.2），Goal3/4/5 从它解析权重。
"""
from __future__ import annotations

import os

from core.config import Settings
from pipeline.inference import InferencePipeline, StudyTaskBinding
from tasks.dummy.dataset_tasks import DummyDuplicateTask
from tasks.dummy.study_tasks import DummyGoal3Task, DummyGoal4Task, DummyGoal5Task
from tasks.goal1_authenticity.config import Goal1Config
from tasks.goal1_authenticity.task import Goal1AuthenticityTask
from tasks.goal2_duplicate.config import Goal2DuplicateConfig
from tasks.goal2_duplicate.inference import DuplicateProbe
from tasks.goal2_duplicate.task import Goal2DuplicateRecorder
from tasks.goal2_stitched.config import Goal2StitchedConfig
from tasks.goal2_stitched.gating import GatedStudyTask, gated_fields
from tasks.goal2_stitched.task import Goal2StitchedTask
from tasks.goal3.task import TumorTask
from tasks.goal4.task import DiagnosisTask
from tasks.goal5.task import Goal5Task


def _enabled(key: str) -> bool:
    """``GLIOMA_GOALS`` 白名单里是否包含本 Goal（默认全真实）。

    设了 ``GLIOMA_GOALS=goal3,goal5`` 就只有列出的用真实实现、其余退回 Dummy ——
    用于规范 §15.1「每次只替换一个插件并运行完整回归」的对照实验。
    """
    raw = os.environ.get("GLIOMA_GOALS", "").strip()
    if not raw:
        return True
    return key in {g.strip() for g in raw.split(",") if g.strip()}


def _pick_goal(key: str, build_real, build_dummy):
    if _enabled(key):
        return build_real(), "real"
    return build_dummy(), "dummy"


def build_pipeline(
    goal1_config: Goal1Config | None = None,
    goal2_config: Goal2StitchedConfig | None = None,
    duplicate_config: Goal2DuplicateConfig | None = None,
    *,
    gated_fields_override: tuple[str, ...] | None = None,
) -> InferencePipeline:
    """规范顺序的完整管线：Goal1/2（官方实现）+ Goal3/5/4（本队实现）。"""
    settings = Settings.from_env()

    duplicate = duplicate_config or Goal2DuplicateConfig.from_env()
    fields = gated_fields() if gated_fields_override is None else tuple(gated_fields_override)
    probe = DuplicateProbe(duplicate) if duplicate.enabled else None

    goal3, g3_kind = _pick_goal(
        "goal3", lambda: TumorTask(settings=settings), lambda: DummyGoal3Task())
    goal5, g5_kind = _pick_goal(
        "goal5", lambda: Goal5Task(settings=settings), lambda: DummyGoal5Task())
    goal4, g4_kind = _pick_goal(
        "goal4", lambda: DiagnosisTask(settings=settings), lambda: DummyGoal4Task())

    kinds = {"goal3": g3_kind, "goal5": g5_kind, "goal4": g4_kind}
    real = [k for k in ("goal3", "goal5", "goal4") if kinds[k] == "real"]
    dummy = [k for k in ("goal3", "goal5", "goal4") if kinds[k] == "dummy"]
    print(
        f"[real_pipeline] Goal3/5/4 → 真实 {len(real)}/3: {','.join(real) or '无'}"
        + (f"  ⚠️ Dummy 补位: {','.join(dummy)}" if dummy else "  ✓ 全真实"),
        flush=True,
    )

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
