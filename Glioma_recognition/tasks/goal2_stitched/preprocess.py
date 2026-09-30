"""Goal2-A 拼接影像检测 的预处理入口。

实现已提取到 ``tasks._common.volume`` **单点维护**（规范 §5.1 禁止每个 Goal 各写一套
NIfTI 读取），本文件保留为该 Goal 的稳定导入路径。
"""
from __future__ import annotations

from tasks._common.volume import CHANNEL_ORDER, PreparedVolume, build_volume, global_view, zscore

__all__ = ["CHANNEL_ORDER", "PreparedVolume", "build_volume", "global_view", "zscore"]
