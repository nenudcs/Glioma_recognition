"""真实 Goal 插件包的 Pipeline 注册入口（规范 §5.1、§17.1）。

与 ``tasks/glioma/pipeline.py``（指向算法工程 ``integration/`` 的过渡桥接）的区别：
本模块直接注册 ``tasks/goalX`` 下的**真实插件包**，不依赖外部算法工程，
是目标架构下的正式接入方式。

启用方式::

    export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline
    export COMPETITION_CHECKPOINT_ROOT=/path/to/checkpoint

绑定顺序与团队 Dummy 基线一致：goal1 → goal2_stitched → goal3 → goal5 → goal4，
另加数据集级 goal2_duplicate（规范 §8.1 推荐顺序）。

``GLIOMA_GOALS`` 环境变量控制启用哪些真实插件，未启用的用 Dummy 补位——
这是规范 §15.1 "每次只替换一个插件并运行完整回归"的直接支持。
"""
from __future__ import annotations

import os

from pipeline.inference import InferencePipeline, StudyTaskBinding

DEFAULT_GOALS = "goal1,goal2_stitched,goal3,goal5,goal4,goal2_duplicate"

#: 各插件在启用时的构建函数（延迟导入：任何单个插件的导入错误不应拖垮服务启动）
_BUILDERS = {
    "goal1": ("tasks.goal1_authenticity.task", "Goal1Task"),
    "goal2_stitched": ("tasks.goal2_stitched.task", "StitchedTask"),
    "goal3": ("tasks.goal3_tumor.task", "TumorTask"),
    "goal5": ("tasks.goal5_segmentation.task", "Goal5Task"),
    "goal4": ("tasks.goal4_diagnosis.task", "DiagnosisTask"),
    "goal2_duplicate": ("tasks.goal2_duplicate.task", "DuplicateTask"),
}

_DUMMY = {
    "goal1": ("tasks.dummy.study_tasks", "DummyGoal1Task"),
    "goal2_stitched": ("tasks.dummy.study_tasks", "DummyStitchedTask"),
    "goal3": ("tasks.dummy.study_tasks", "DummyGoal3Task"),
    "goal5": ("tasks.dummy.study_tasks", "DummyGoal5Task"),
    "goal4": ("tasks.dummy.study_tasks", "DummyGoal4Task"),
    "goal2_duplicate": ("tasks.dummy.dataset_tasks", "DummyDuplicateTask"),
}


def _load(module_name: str, class_name: str):
    import importlib

    return getattr(importlib.import_module(module_name), class_name)


def _pick(key: str, enabled: set[str], device: str):
    """启用则构建真实插件，否则用 Dummy 补位。"""
    module_name, class_name = (_BUILDERS if key in enabled else _DUMMY)[key]
    cls = _load(module_name, class_name)
    try:
        return cls(device=device)
    except TypeError:
        return cls()                                              # Dummy 不接受 device


def build_pipeline() -> InferencePipeline:
    """构建注册了 Goal1~5 真实插件的团队 ``InferencePipeline``。"""
    enabled = {g.strip() for g in
               os.environ.get("GLIOMA_GOALS", DEFAULT_GOALS).split(",") if g.strip()}
    device = os.environ.get("GLIOMA_DEVICE", "cuda")

    study_tasks = (
        StudyTaskBinding("goal1", _pick("goal1", enabled, device)),
        StudyTaskBinding("goal2_stitched", _pick("goal2_stitched", enabled, device)),
        StudyTaskBinding("goal3", _pick("goal3", enabled, device)),
        StudyTaskBinding("goal5", _pick("goal5", enabled, device)),
        StudyTaskBinding("goal4", _pick("goal4", enabled, device)),
    )
    duplicate_task = _pick("goal2_duplicate", enabled, device)

    # staged rollout（规范 §15.1「每次只替换一个插件」）最危险的失误是
    # "以为全开了、其实只有 GLIOMA_GOALS 里列的那几个是真的" —— 打印成一行，一眼可见。
    all_keys = tuple(_BUILDERS)
    real = [k for k in all_keys if k in enabled]
    dummy = [k for k in all_keys if k not in enabled]
    print(
        f"[real_pipeline] GLIOMA_GOALS={','.join(sorted(enabled)) or '(空)'} → "
        f"真实插件 {len(real)}/{len(all_keys)}: {','.join(real) or '无'}"
        + (f"  ⚠️ Dummy 补位: {','.join(dummy)}" if dummy else "  ✓ 无 Dummy 补位"),
        flush=True,
    )
    return InferencePipeline(study_tasks=study_tasks, duplicate_task=duplicate_task)
