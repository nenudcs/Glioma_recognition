"""3D 残差 U-Net + 多任务头（赛道4）。

输出设计（一次前向给出全部所需）：
- ``seg``   : [B,2,D,H,W]  logits —— 通道0=core（T1C 核心区）、通道1=peri（FLAIR/T2 总异常区）
- ``ds``    : 深监督的中间尺度 seg logits 列表
- ``cls``   : 结构化字段 logits 列表（binary → 1 logit；single → n_classes logits）
- ``embed`` : [B,E] L2 归一化嵌入（重复影像匹配用）
- ``special``: [B,2] logits —— 0=假人体(IsNotHumanBodyProb)、1=拼接(IsStitchedProb)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_norm_act(ci: int, co: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(ci, co, k, padding=k // 2, bias=False),
        nn.InstanceNorm3d(co, affine=True),
        nn.LeakyReLU(0.01, inplace=True),
    )


class ResBlock(nn.Module):
    """预激活残差块（3D）。"""

    def __init__(self, ci: int, co: int, stride: int = 1):
        super().__init__()
        self.c1 = conv_norm_act(ci, co)
        self.c2 = nn.Sequential(
            nn.Conv3d(co, co, 3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(co, affine=True),
        )
        self.skip = (nn.Sequential(nn.Conv3d(ci, co, 1, stride=stride, bias=False),
                                   nn.InstanceNorm3d(co, affine=True))
                     if (ci != co or stride != 1) else nn.Identity())
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        h = self.c2(self.c1(x))
        return self.act(h + self.skip(x))


class Down(nn.Module):
    def __init__(self, ci: int, co: int):
        super().__init__()
        self.block = ResBlock(ci, co, stride=2)

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, ci: int, skip_ch: int, co: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(ci, co, 2, stride=2)
        self.block = ResBlock(co + skip_ch, co)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:                     # 尺寸对齐（奇数尺寸）
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class GliomaNet(nn.Module):
    def __init__(self, in_ch: int = 4, base: int = 32, depth: int = 4,
                 cls_spec: list[tuple[str, int]] | None = None, embed_dim: int = 128,
                 deep_supervision: bool = True):
        super().__init__()
        cls_spec = cls_spec or []
        self.deep_supervision = deep_supervision
        chs = [base * (2 ** min(i, 3)) for i in range(depth)]      # 32,64,128,256

        self.stem = conv_norm_act(in_ch, base)
        self.downs = nn.ModuleList()
        self.encs = nn.ModuleList()
        ci = base
        for i in range(depth):
            co = chs[i]
            self.downs.append(Down(ci, co) if i > 0 else nn.Identity())
            self.encs.append(ResBlock(co if i > 0 else ci, co))
            ci = co

        self.ups = nn.ModuleList()
        dec_chs = list(reversed(chs[:-1]))
        for i, dc in enumerate(dec_chs):
            skip_ch = chs[depth - 2 - i]
            self.ups.append(Up(ci, skip_ch, dc))
            ci = dc

        self.seg_head = nn.Conv3d(ci, 2, 1)                        # core / peri
        if deep_supervision:
            self.ds_heads = nn.ModuleList([nn.Conv3d(c, 2, 1) for c in dec_chs[:-1]])

        self.pool = nn.AdaptiveAvgPool3d(1)
        feat = ci + max(chs)
        self.cls_heads = nn.ModuleList([nn.Linear(feat, n) for _n, n in cls_spec])
        self.field_names = [n for n, _ in cls_spec]
        self.embed_head = nn.Sequential(nn.Linear(feat, embed_dim), nn.GELU(),
                                        nn.Linear(embed_dim, embed_dim))
        self.special_head = nn.Linear(feat, 2)                     # 假人体 / 拼接

    def forward(self, x: torch.Tensor) -> dict:
        h = self.stem(x)
        skips = []
        for i, (down, enc) in enumerate(zip(self.downs, self.encs)):
            h = enc(down(h))
            if i < len(self.encs) - 1:
                skips.append(h)

        bottleneck = h
        ds_logits = []
        for i, up in enumerate(self.ups):
            skip = skips[-1 - i]
            h = up(h, skip)
            if self.deep_supervision and i < len(self.ups) - 1:
                ds_logits.append(self.ds_heads[i](h))

        seg = self.seg_head(h)
        pooled = torch.cat([self.pool(h).flatten(1), self.pool(bottleneck).flatten(1)], dim=1)
        embed = F.normalize(self.embed_head(pooled), dim=1)
        return {
            "seg": seg,
            "ds": ds_logits,
            "cls": [head(pooled) for head in self.cls_heads],
            "embed": embed,
            "special": self.special_head(pooled),
        }


def build_model(cls_spec: list[tuple[str, int]], in_ch: int = 4, base: int = 32,
                arch: str = "mednext", depth: int = 4, blocks_per_stage: int = 2,
                k: int = 3, expand: int = 2, dropout: float = 0.0,
                aniso_z: bool = False, max_ch: int = 320, plain_stages: int = 0,
                dec_blocks: int = 1) -> nn.Module:
    """骨干工厂：``arch="mednext"``（默认，强）或 ``"resunet"``（备选，轻）。

    容量档位（见 ``README.md`` §6.2 训练配置）：
    - ``plain_stages``：最深的 N 级改用普通 3×3×3 卷积（**参数大增、FLOPs 几乎不变**）；
    - ``max_ch``：通道上限，配合 ``depth=5`` 得到 320 通道瓶颈；
    - ``expand``：倒瓶颈膨胀比；``dec_blocks``：解码器每级 block 数。

    所有档位的输出契约完全一致（seg/ds/cls/embed/special），训练与推理代码无感知。
    """
    if arch == "resunet":
        return GliomaNet(in_ch=in_ch, base=base, cls_spec=cls_spec)
    from .mednext import MedNeXtNet
    return MedNeXtNet(in_ch=in_ch, base=base, depth=depth, cls_spec=cls_spec,
                      blocks_per_stage=blocks_per_stage, k=k, expand=expand,
                      dropout=dropout, aniso_z=aniso_z, max_ch=max_ch,
                      plain_stages=plain_stages, dec_blocks=dec_blocks)


def cls_spec_from_config(labels_cfg: dict) -> list[tuple[str, int]]:
    """labels.yaml → [(字段名, 输出维度)]；binary → 1。"""
    spec = []
    for f in labels_cfg["fields"]:
        n = 1 if f["type"] == "binary" else len(f["classes"])
        spec.append((f["key"], n))
    return spec
