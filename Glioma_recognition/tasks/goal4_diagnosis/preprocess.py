"""Goal4 辅助诊断 的预处理入口。

实现已提取到 ``tasks._common.volume`` **单点维护**（规范 §5.1 禁止每个 Goal
各写一套 NIfTI 读取），本文件保留为该 Goal 的稳定导入路径。

Goal4 只读**整脑视图**（``global_view``）：它的 14 个字段都是检查级的
（位置、分级、强化形态…），用局部 patch 判断会丢失关键证据，且必须与
推理侧使用同一物理尺度——``size_mm=192 → out=96`` 与训练侧完全一致。
"""
from __future__ import annotations

from tasks._common.volume import (  # noqa: F401
    CHANNEL_ORDER,
    PreparedVolume,
    build_volume,
    global_view,
    zscore,
)

__all__ = ["CHANNEL_ORDER", "PreparedVolume", "build_volume", "global_view", "zscore"]
