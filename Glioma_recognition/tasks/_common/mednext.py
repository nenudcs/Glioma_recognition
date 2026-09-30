"""MedNeXt 风格的 3D 医学分割骨干（自研实现，仅使用公开设计思想）。

为什么换骨干（对齐赛事"自建模型组"规则）：
- 规范把 MedNeXt / nnU-Net / 3D UX-Net 列为**允许的基线模型**，自研 MedNeXt 风格
  骨干完全合规，且比小型 ResUNet 在 BraTS 类任务上 Dice 高 3~6 个点；
- 关键机制（公开论文设计，此处独立编码实现）：
  1. **深度可分离大核卷积**（5×5×5 / 3×3×3）：3D 感受野大但参数少，显存友好；
  2. **GRN（Global Response Normalization）**：抑制通道冗余，小数据更稳；
  3. **各向异性下采样**：脑 MRI 层厚方向分辨率低，z 轴下采样更激进（可配）。
- 同时保留原 ResUNet 作为 `arch="resunet"` 备选，权重与配置可切换。

输出契约与 `unet3d.GliomaNet` 完全一致（seg/ds/cls/embed/special），
推理与训练代码无需感知骨干差异。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(ch: int) -> int:
    """GroupNorm 组数：取能整除通道数的最大值（≤8），避免整除错误。"""
    for g in (8, 4, 2):
        if ch % g == 0:
            return g
    return 1


class GRN(nn.Module):
    """ConvNeXt-V2 的 Global Response Normalization（3D 版）。"""

    def __init__(self, ch: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, ch, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, ch, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * nx) + self.beta


class MedNeXtBlock(nn.Module):
    """深度可分离大核卷积 + 倒瓶颈 + GRN 的残差块。"""

    def __init__(self, ch: int, k: int = 3, expand: int = 2, drop: float = 0.0):
        super().__init__()
        self.dw = nn.Conv3d(ch, ch, k, padding=k // 2, groups=ch, bias=False)
        self.norm = nn.GroupNorm(_groups(ch), ch)
        self.pw1 = nn.Conv3d(ch, ch * expand, 1, bias=False)
        self.pw2 = nn.Conv3d(ch * expand, ch, 1, bias=False)
        self.act = nn.GELU()
        self.grn = GRN(ch)
        self.drop = nn.Dropout3d(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        h = self.norm(h)
        h = self.pw1(h)
        h = self.act(h)
        h = self.pw2(h)
        h = self.grn(h)
        h = self.drop(h)
        return self.act(x + h)


class PlainResBlock(nn.Module):
    """普通 3×3×3 卷积残差块（**仅用于最低分辨率的若干级**）。

    为什么这样设计：深度可分离卷积的参数/算力比很低，而参数量是表达力的重要来源。
    在低分辨率层（如 12³/6³ 体素、256 通道）换成普通卷积，可让参数量增加数倍
    而 FLOPs（即训练时间）只增加几个百分点 —— 这是"不放慢训练就放大容量"的
    最高性价比位置，也正是 nnU-Net 参数量（~30M）的主要来源。
    """

    def __init__(self, ch: int, expand: int = 1, dropout: float = 0.0):
        super().__init__()
        mid = ch * max(1, int(expand))
        self.c1 = nn.Conv3d(ch, mid, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(_groups(mid), mid)
        self.c2 = nn.Conv3d(mid, ch, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(_groups(ch), ch)
        self.act = nn.GELU()
        self.grn = GRN(ch)
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.n1(self.c1(x)))
        h = self.n2(self.c2(h))
        h = self.grn(h)
        return self.act(x + self.drop(h))


class DownBlock(nn.Module):
    """下采样：stride-2 卷积 + 归一化（各向异性：z 轴可额外压缩）。"""

    def __init__(self, ci: int, co: int, k: int = 2, stride=(2, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(ci, co, k, stride=stride, bias=False)
        self.norm = nn.GroupNorm(_groups(co), co)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    """上采样（trilinear + 1×1 投影）+ skip 拼接 + 通道融合 + 残差块。

    注意：拼接后的通道数为 ``co + skip_ch``，必须先用 1×1 融合回 ``co``
    再交给 MedNeXtBlock（后者保持通道数不变）。
    """

    def __init__(self, ci: int, skip_ch: int, co: int, n_blocks: int = 1,
                 k: int = 3, expand: int = 2, dropout: float = 0.0, plain: bool = False):
        super().__init__()
        self.proj = nn.Conv3d(ci, co, 1, bias=False)
        self.norm = nn.GroupNorm(_groups(co), co)
        self.fuse = nn.Conv3d(co + skip_ch, co, 1, bias=False)
        self.fuse_norm = nn.GroupNorm(_groups(co), co)
        blk = (lambda c: PlainResBlock(c, 1, dropout)) if plain \
            else (lambda c: MedNeXtBlock(c, k, expand, dropout))
        self.blocks = nn.Sequential(*[blk(co) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = self.norm(self.proj(x))
        h = self.fuse_norm(self.fuse(torch.cat([x, skip], dim=1)))
        return self.blocks(h)


class MedNeXtNet(nn.Module):
    """MedNeXt 风格编码-解码骨干 + 赛道四多任务头。

    - ``seg``：``[B,2,D,H,W]`` logits（通道 0=core/T1C 核心区，1=peri/FLAIR-T2 总异常区）
    - ``ds`` ：深监督中间尺度 logits
    - ``cls``：结构化字段 logits（binary → 1；single → n_classes）
    - ``embed``：L2 归一化 128 维嵌入（重复影像匹配）
    - ``special``：``[B,2]`` 假人体 / 拼接 logits（目标一/二）
    """

    def __init__(self, in_ch: int = 4, base: int = 32, depth: int = 4,
                 cls_spec: list[tuple[str, int]] | None = None, embed_dim: int = 128,
                 deep_supervision: bool = True, blocks_per_stage: int = 2,
                 k: int = 3, expand: int = 2, dropout: float = 0.0,
                 aniso_z: bool = False, max_ch: int = 320,
                 plain_stages: int = 0, dec_blocks: int = 1):
        """``plain_stages``：最深的 N 级用**普通 3×3×3 卷积**（低分辨率处参数/算力比最高）;
        ``max_ch``：通道上限（配合 ``depth=5`` 得到 MedNeXt 风格的 320 通道瓶颈）;
        ``dec_blocks``：解码器每级的 block 数。
        """
        super().__init__()
        cls_spec = cls_spec or []
        self.deep_supervision = deep_supervision
        # min(base·2^i, max_ch)：depth=4/base=32/max_ch=320 时与旧版 [32,64,128,256] 完全一致
        chs = [min(base * (2 ** i), max_ch) for i in range(depth)]

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, chs[0], 3, padding=1, bias=False),
            nn.GroupNorm(_groups(chs[0]), chs[0]), nn.GELU())

        # ---- 编码器 ----
        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev = chs[0]
        for i in range(depth):
            co = chs[i]
            use_plain = i >= depth - int(plain_stages) and plain_stages > 0
            mk = (lambda c, first: PlainResBlock(c, 1, dropout)) if use_plain else \
                (lambda c, first: MedNeXtBlock(prev if (first and i == 0) else c,
                                               k, expand, dropout))
            if i == 0:
                self.downs.append(nn.Identity())
            else:
                stride = (1, 2, 2) if (aniso_z and i == 1) else (2, 2, 2)
                self.downs.append(DownBlock(prev, co, k=2, stride=stride))
            self.enc_stages.append(nn.Sequential(
                *[mk(co, j == 0) for j in range(blocks_per_stage)]))
            prev = co

        # ---- 解码器 ----
        self.ups = nn.ModuleList()
        dec_chs = list(reversed(chs[:-1]))
        ci = chs[-1]
        for i, dc in enumerate(dec_chs):
            skip_ch = chs[depth - 2 - i]
            plain_here = (i < int(plain_stages) - 1)              # 与编码器对称的深层
            self.ups.append(UpBlock(ci, skip_ch, dc, n_blocks=int(dec_blocks), k=k,
                                    expand=expand, dropout=dropout, plain=plain_here))
            ci = dc

        self.seg_head = nn.Conv3d(ci, 2, 1)
        if deep_supervision:
            self.ds_heads = nn.ModuleList([nn.Conv3d(c, 2, 1) for c in dec_chs[:-1]])

        # ---- 全局头（分类 / 嵌入 / 特殊影像）----
        self.pool = nn.AdaptiveAvgPool3d(1)
        feat = ci + chs[-1]
        self.cls_heads = nn.ModuleList([nn.Linear(feat, n) for _n, n in cls_spec])
        self.field_names = [n for n, _ in cls_spec]
        self.embed_head = nn.Sequential(nn.Linear(feat, embed_dim), nn.GELU(),
                                        nn.Linear(embed_dim, embed_dim))
        self.special_head = nn.Sequential(nn.Linear(feat, 64), nn.GELU(),
                                          nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> dict:
        h = self.stem(x)
        skips = []
        bottleneck = h
        for i, (down, enc) in enumerate(zip(self.downs, self.enc_stages)):
            h = enc(down(h))
            if i < len(self.enc_stages) - 1:
                skips.append(h)
            else:
                bottleneck = h

        ds_logits = []
        for i, up in enumerate(self.ups):
            h = up(h, skips[-1 - i])
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
            "pooled": pooled,
        }
