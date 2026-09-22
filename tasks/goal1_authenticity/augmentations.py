"""[研发] 数据增强（规范 §5.1；比赛运行入口不得导入本文件）。

只作用于「一层切片图像堆」``(K, H, W)``：翻转、亮度抖动、低频偏置场、模糊、噪声、gamma。
训练时随机启用，验证/推理阶段不使用。
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from .preprocess import resize2d


@dataclass
class AugmentConfig:
    flip_prob: float = 0.5
    intensity_jitter: float = 0.1
    bias_prob: float = 0.3
    bias_strength: float = 0.25
    blur_prob: float = 0.2
    noise_prob: float = 0.3
    noise_sigma: float = 0.05
    gamma_prob: float = 0.2


def _box_blur(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    acc = np.zeros_like(image)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy:dy + image.shape[0], dx:dx + image.shape[1]]
    return (acc / 9.0).astype(np.float32)


def _low_frequency_field(shape: tuple[int, int], rng: random.Random, cells: int = 4) -> np.ndarray:
    small = np.array(
        [[rng.uniform(0.7, 1.3) for _ in range(cells)] for _ in range(cells)],
        dtype=np.float32,
    )
    return resize2d(small, shape[0])


def augment_slices(
    images: np.ndarray,
    rng: random.Random,
    config: AugmentConfig | None = None,
) -> np.ndarray:
    """``images``: ``(K, H, W)`` float32 → 同形状增强结果。"""
    config = config or AugmentConfig()
    if rng.random() < config.flip_prob:
        images = images[:, :, ::-1]
    if rng.random() < config.intensity_jitter:
        images = images * rng.uniform(1 - config.intensity_jitter, 1 + config.intensity_jitter)
    if rng.random() < config.bias_prob:
        field = _low_frequency_field(images.shape[1:], rng)
        images = images * (1.0 + config.bias_strength * (field - 1.0))
    if rng.random() < config.blur_prob:
        images = np.stack([_box_blur(image) for image in images])
    if rng.random() < config.noise_prob:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, config.noise_sigma, images.shape
        )
        images = images + np.asarray(noise, dtype=np.float32)
    if rng.random() < config.gamma_prob:
        gamma = rng.uniform(0.7, 1.4)
        lo, hi = float(images.min()), float(images.max())
        if hi > lo:
            images = np.clip((images - lo) / (hi - lo), 0, 1) ** gamma * (hi - lo) + lo
    return np.ascontiguousarray(images, dtype=np.float32)
