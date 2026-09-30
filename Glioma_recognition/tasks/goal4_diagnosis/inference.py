"""Goal4 辅助诊断 的推理入口：权重加载 + 共享骨干前向。

组合逻辑在 ``tasks._common.backbone_runner.BackboneRunner`` 中**单点实现**
（本工程是共享多任务骨干，各 Goal 只取不同的输出头，见规范 §5.1）；
本文件保留为该 Goal 的稳定导入路径，便于将来替换为独立模型。

与 Goal1/2-A/3 的差别只有一个：Goal4 读的是 **14 个分类头**而不是单个
``special`` / ``cls`` 通道，因此它必须校验权重里头的数量足够
（见 ``task.DiagnosisTask._collect``）——少一个头就静默用默认值填充，
这类失效不会报错，只能靠显式校验拦住。
"""
from __future__ import annotations

from tasks._common.backbone_runner import BackboneRunner  # noqa: F401
from tasks._common.factory import build_shared_backbone  # noqa: F401

__all__ = ["BackboneRunner", "build_shared_backbone"]
