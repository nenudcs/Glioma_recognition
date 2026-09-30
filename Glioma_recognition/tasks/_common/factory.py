"""共享骨干工厂：按 checkpoint 内记录的模型超参重建网络。

规范 §5.2 要求权重只在服务初始化时加载一次；本模块把"从 checkpoint 元信息
重建同构骨干"这件事收敛到一处，避免各 Goal 各写一份 ``build_model`` 参数列表
（那些参数一旦在某处漏改，就会静默加载出结构不符的网络）。

**必须从 checkpoint 读结构超参**，而不是从配置读：训练时的 ``base``/``depth``
等可能与当前配置不同，用配置重建会得到形状不匹配、``load_state_dict`` 静默跳过的结果。
"""
from __future__ import annotations

from typing import Any


def build_shared_backbone(ck: dict[str, Any], arch: str | None = None,
                          in_channels: int = 4):
    """按 checkpoint 中记录的 ``model_cfg`` 重建骨干。

    Args:
        ck: ``torch.load`` 得到的 checkpoint 字典。
        arch: 覆盖 checkpoint 里的 arch（一般不需要）。
        in_channels: 输入通道数（需与训练一致）。

    Returns:
        与训练同构的 ``nn.Module``（未加载权重，由调用方 load_state_dict）。
    """
    from tasks._common.mednext import MedNeXtNet
    from tasks._common.unet3d import GliomaNet

    mc = ck.get("model_cfg") or {}
    a = str(arch or ck.get("arch") or mc.get("arch") or "mednext")
    spec = ck.get("cls_spec") or []

    if a == "resunet":
        return GliomaNet(in_ch=in_channels, base=int(mc.get("base", 32)), cls_spec=spec)

    return MedNeXtNet(
        in_ch=in_channels,
        base=int(mc.get("base", 32)),
        depth=int(mc.get("depth", 4)),
        cls_spec=spec,
        blocks_per_stage=int(mc.get("blocks_per_stage", 2)),
        k=int(mc.get("k", 3)),
        expand=int(mc.get("expand", 2)),
        aniso_z=bool(mc.get("aniso_z", False)),
        max_ch=int(mc.get("max_ch", 320)),
        plain_stages=int(mc.get("plain_stages", 0)),
        dec_blocks=int(mc.get("dec_blocks", 1)),
    )
