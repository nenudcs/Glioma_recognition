"""Goal5 的空间变换入口。

实现已提取到 ``tasks._common.spatial`` **单点维护**（Goal1~5 都需要同样的
世界坐标重采样与逆变换，规范 §5.1 禁止每个 Goal 各写一套）。
本文件保留为 Goal5 的稳定导入路径，便于将来独立替换实现。
"""
from __future__ import annotations

from tasks._common.spatial import (  # noqa: F401
    resample_to,
    restore_binary_to_source,
    spacing_of,
    target_grid,
)

__all__ = ["resample_to", "restore_binary_to_source", "spacing_of", "target_grid"]
