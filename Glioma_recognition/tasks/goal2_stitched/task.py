"""Goal2-A 影像规范性审核 · 拼接影像检测（规范 §9.2 的 StudyTask）。

输出 ``IsStitchedProb``：检查级拼接概率。取共享骨干 ``special`` 头的通道 1。
（Goal2-B 重复影像检测是 ``DatasetTask``，见 ``tasks/goal2_duplicate``。）
"""
from __future__ import annotations

from typing import Any

from core.config import Settings
from tasks._common.single_head import SingleHeadStudyTask
from tasks.goal2_stitched.config import Goal2StitchedConfig
from tasks.goal2_stitched.postprocess import to_check_level
from tasks.results import StitchedResult

#: 本 Goal 的推理期配置（规范 §5.1 把"配置"列为独立交付项）
_CONFIG = Goal2StitchedConfig()


class StitchedTask(SingleHeadStudyTask):
    """拼接影像检查级概率。"""

    name = "goal2_stitched"
    head = ("special", 1)
    # 取值与原先的类属性默认值一致，仅收敛到 config.py 单点维护
    ckpt_rel = _CONFIG.ckpt_rel
    in_channels = _CONFIG.in_channels
    arch = _CONFIG.arch
    global_size = _CONFIG.global_size
    global_size_mm = _CONFIG.global_size_mm
    tta_flips = _CONFIG.tta_flips
    tta_batch = _CONFIG.tta_batch
    common_spacing = _CONFIG.common_spacing

    def build_result(self, probability: float) -> StitchedResult:
        return StitchedResult(stitched_probability=to_check_level(probability))


def build_task(settings: Settings | None = None, **kwargs: Any) -> StitchedTask:
    """工厂入口。"""
    return StitchedTask(settings=settings, **kwargs)
