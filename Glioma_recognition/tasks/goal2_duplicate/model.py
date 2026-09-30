"""Goal2-B 的嵌入模型（规范 §5.2 中的 ``goal2_duplicate/encoder.pt``）。

本工程是共享多任务骨干，嵌入是其中一个输出头（``embed``），
因此这里不重复定义网络，只声明"重复影像任务使用 embed 头"这一语义。

若要换成独立编码器（例如专门的对比学习编码器），只需让
``build_encoder()`` 返回该网络，并把 ``DuplicateConfig.ckpt_rel`` 指向它的权重。
"""
from __future__ import annotations

from tasks._common.factory import build_shared_backbone

#: 嵌入维度由骨干决定，创建后可通过 ``embed_dim(model)`` 查询
EMBED_HEAD = "embed"


def build_encoder(ck: dict, in_channels: int = 4):
    """按 checkpoint 元信息重建编码器（与训练结构同源）。"""
    return build_shared_backbone(ck, None, in_channels)


def embed_dim() -> int:
    """嵌入向量维度（与训练时一致；此处为常量以便契约测试断言）。"""
    return 128
