"""Goal2-B 的预处理入口。

实现复用 ``tasks._common.volume``（规范 §5.1 禁止每个 Goal 各写一套 NIfTI 读取）。
重复影像任务对几何特别敏感，因此指纹里显式保留 spacing 与原点。
"""
from __future__ import annotations

from tasks._common.volume import (  # noqa: F401
    CHANNEL_ORDER,
    PreparedVolume,
    build_volume,
    global_view,
)

__all__ = ["CHANNEL_ORDER", "PreparedVolume", "build_volume", "global_view"]
