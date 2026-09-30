"""Goal1 影像真实性识别：区分真人体 / 假人体 / 非人体（规范 §9.1）。

输出 ``IsNotHumanBodyProb`` —— "假人体或非人体"的**检查级**概率
（规范 §3.3 已确认该语义；字段中文解释与评分细则的边界以细则为准）。

本工程是多任务共享骨干，Goal1 取 ``special`` 头的通道 0；
``postprocess.py`` 完成"序列级 → 检查级"的聚合。
"""
from __future__ import annotations

from typing import Any

from core.config import Settings
from tasks._common.single_head import SingleHeadStudyTask
from tasks.goal1_authenticity.config import Goal1Config
from tasks.goal1_authenticity.postprocess import to_check_level
from tasks.results import Goal1Result

#: 本 Goal 的推理期配置（规范 §5.1 把"配置"列为独立交付项，单点声明在 config.py）
_CONFIG = Goal1Config()


class Goal1Task(SingleHeadStudyTask):
    """假人体 / 非人体 检查级概率。"""

    name = "goal1_authenticity"
    head = ("special", 0)
    # 下列参数从 config.py 读取：取值与原先的类属性默认值完全一致，
    # 只是把"可调项"收敛到一处，便于契约测试**静态**校验权重路径。
    ckpt_rel = _CONFIG.ckpt_rel
    in_channels = _CONFIG.in_channels
    arch = _CONFIG.arch
    global_size = _CONFIG.global_size
    global_size_mm = _CONFIG.global_size_mm
    tta_flips = _CONFIG.tta_flips
    tta_batch = _CONFIG.tta_batch
    common_spacing = _CONFIG.common_spacing

    def build_result(self, probability: float) -> Goal1Result:
        return Goal1Result(not_human_probability=to_check_level(probability))


def build_task(settings: Settings | None = None, **kwargs: Any) -> Goal1Task:
    """工厂入口。"""
    return Goal1Task(settings=settings, **kwargs)
