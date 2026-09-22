"""[运行] 目标一网络定义（规范 §5.1：model.py 只定义网络）。

2-D 预训练骨干（ConvNeXt-Tiny）+ FFT 幅度谱分支，逐切片打分；
检查级聚合由 ``postprocess.py`` 完成，权重由 ``inference.py`` 加载。

本文件不扫描比赛目录、不写结果、不发送回调。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_BACKBONE


class FrequencyBranch(nn.Module):
    """FFT 幅度谱分支：暴露上采样/插值伪影（合成假体与真实扫描差异明显）。"""

    def __init__(self, out_dim: int = 64, spectrum_size: int = 32):
        super().__init__()
        self.spectrum_size = spectrum_size
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft2(x, norm="ortho").abs()
        spectrum = torch.log1p(spectrum)
        spectrum = F.adaptive_avg_pool2d(spectrum, (self.spectrum_size, self.spectrum_size))
        return self.net(spectrum).flatten(1)


class AuthenticityNet(nn.Module):
    """切片级打分器；检查级分数由 postprocess 聚合（先平均 logit 再 sigmoid）。"""

    def __init__(
        self,
        backbone: str = DEFAULT_BACKBONE,
        pretrained: bool = True,
        dropout: float = 0.2,
        frequency_branch: bool = True,
        frequency_dim: int = 64,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        import timm  # 延迟导入：缺 timm 时本模块仍可被静态检查

        self.backbone_name = backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        features = int(self.backbone.num_features)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.backbone.eval()
        self.frequency_branch = FrequencyBranch(frequency_dim) if frequency_branch else None
        head_input = features + (frequency_dim if frequency_branch else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_input, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: (N, 3, H, W) -> 切片级 logits (N,)。"""
        features = self.backbone(images)
        if self.frequency_branch is not None:
            gray = images.mean(dim=1, keepdim=True)
            features = torch.cat([features, self.frequency_branch(gray)], dim=1)
        return self.head(features).squeeze(1)
