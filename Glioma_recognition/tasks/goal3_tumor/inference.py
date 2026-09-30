"""Goal3 病灶识别 的推理入口：权重加载 + 共享骨干前向。

组合逻辑在 ``tasks._common.single_head.SingleHeadStudyTask`` 中单点实现
（本工程是共享多任务骨干，各 Goal 只取不同的输出头）；
本文件保留为该 Goal 的稳定导入路径，便于将来替换为独立模型。
"""
from __future__ import annotations

from tasks._common.factory import build_shared_backbone  # noqa: F401
from tasks._common.single_head import SingleHeadStudyTask  # noqa: F401

__all__ = ["SingleHeadStudyTask", "build_shared_backbone"]
