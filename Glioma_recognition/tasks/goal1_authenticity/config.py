"""Goal1 影像真实性识别 的插件配置（规范 §5.1 把"配置"列为独立交付项）。

规范 §5.2：本文件只声明**相对路径**，绝对根目录由 ``core.config.Settings``
的 ``ckpt_root`` 提供。

把这些参数从 ``task.py`` 的类属性收敛到这里，收益与 Goal4/Goal5 相同：

* 调参不必改动任务逻辑；
* 契约测试可以**静态**校验权重路径（无需导入 Task 类，更不必加载权重）；
* 推理依赖的骨干超参（``in_channels``/``arch``/整脑视图尺度）与训练侧
  有了一处明确的对照点——这几项一旦与训练不一致，就会静默加载出
  结构不符的网络或让全局头的输入分布漂移。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Goal1Config:
    """Goal1 推理期参数（默认值与 ``SingleHeadStudyTask`` 的基线一致）。"""

    #: 权重相对路径（规范 §5.2：绝对根目录由 ``Settings.ckpt_root`` 提供）
    ckpt_rel: str = "goal1_authenticity/model.pt"
    #: 共享骨干输入通道数（必须与训练一致，否则加载会明确失败）
    in_channels: int = 4
    #: 骨干结构：mednext / resunet
    arch: str = "mednext"
    #: 整脑视图：物理边长（mm）与输出尺寸——**训练与推理必须一致**
    global_size: int = 96
    global_size_mm: float = 192.0
    #: 推理期 TTA 翻折轴
    tta_flips: tuple[str, ...] = ("x", "y")
    tta_batch: int = 2
    #: 公共网格 spacing（1mm）
    common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
