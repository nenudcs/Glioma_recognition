"""Goal3 病灶识别：区分胶质瘤与非肿瘤性病变（规范 §9.4）。

输出 ``TumorProbability``（ROC-AUC 评估）。取共享骨干中名为 ``TumorProbability``
的分类头。

**已知限制（须如实记录在 README）**：本地与公开训练数据里**没有非肿瘤性病变
（脑梗死/脑脓肿等）负样本**，因此该头在当前数据上无法有效训练，
AUC 也无法在本地评估。链路口径已就绪，待补充负样本后即可启用。
"""
from __future__ import annotations

from typing import Any

from core.config import Settings
from tasks._common.single_head import SingleHeadStudyTask
from tasks.goal3_tumor.config import Goal3Config
from tasks.goal3_tumor.postprocess import to_probability
from tasks.results import Goal3Result

#: 本 Goal 的推理期配置（规范 §5.1 把"配置"列为独立交付项）
_CONFIG = Goal3Config()


class TumorTask(SingleHeadStudyTask):
    """胶质瘤检查级概率。"""

    name = "goal3_tumor"
    head = ("cls", "TumorProbability")
    # 取值与原先的类属性默认值一致，仅收敛到 config.py 单点维护
    ckpt_rel = _CONFIG.ckpt_rel
    in_channels = _CONFIG.in_channels
    arch = _CONFIG.arch
    global_size = _CONFIG.global_size
    global_size_mm = _CONFIG.global_size_mm
    tta_flips = _CONFIG.tta_flips
    tta_batch = _CONFIG.tta_batch
    common_spacing = _CONFIG.common_spacing

    def build_result(self, probability: float) -> Goal3Result:
        return Goal3Result(tumor_probability=to_probability(probability))


def build_task(settings: Settings | None = None, **kwargs: Any) -> TumorTask:
    """工厂入口。"""
    return TumorTask(settings=settings, **kwargs)
