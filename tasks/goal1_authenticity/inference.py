"""[运行] 权重加载与纯推理（规范 §5.1）。

职责边界：

* 从 ``Goal1Config.resolved_model_path``（规范路径 ``…/checkpoint/goal1_authenticity/model.pt``）
  读取一次权重，之后只做前向；
* 切片级 logits → 「取 ``K // 2`` 个最高 logit 求均值再 sigmoid」得到**序列级概率**
  （与训练侧验证口径一致）；
* 不做检查级聚合（那是 ``postprocess.py`` 的职责），不写任何比赛输出。
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import DEFAULT_BACKBONE, Goal1Config
from .preprocess import to_float_volume, volume_to_slices

logger = logging.getLogger(__name__)


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class AuthenticityInference:
    """单权重推理器（规范路径唯一、加载一次）。"""

    def __init__(self, config: Goal1Config | None = None) -> None:
        self.config = config or Goal1Config.from_env()
        self.model_path = self.config.resolved_model_path
        self.device = None
        self.backbone = DEFAULT_BACKBONE
        self.image_size = self.config.image_size or 224
        self.slices_per_case = self.config.slices_per_case or 16
        self.frequency_branch = True
        self.val_ap = None
        self.epoch = None
        self.model_version = "unloaded"
        self._net = None
        self._torch = None

    # -- 加载 ---------------------------------------------------------------
    def load(self) -> None:
        """读取权重并构建网络；失败时抛异常（由 ``task.py`` 决定是否降级）。"""
        import torch

        from .model import AuthenticityNet

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"权重不存在：{self.model_path}（规范路径 = <checkpoint_root>/"
                f"{self.config.checkpoint_relative_path}；可用 GOAL1_CHECKPOINT 临时覆盖）"
            )

        state = self._read_checkpoint(torch, self.model_path)
        self.backbone = str(state.get("backbone") or DEFAULT_BACKBONE)
        self.frequency_branch = bool(state.get("frequency_branch", True))
        self.image_size = int(state.get("image_size") or self.config.image_size or 224)
        self.slices_per_case = int(
            state.get("slices_per_case") or self.config.slices_per_case or 16
        )
        self.val_ap = state.get("val_ap")
        self.epoch = state.get("epoch")

        net = AuthenticityNet(
            backbone=self.backbone,
            pretrained=False,                      # 权重来自本地文件，推理不联网
            frequency_branch=self.frequency_branch,
        )
        missing, unexpected = net.load_state_dict(state["model"], strict=False)
        if missing or unexpected:
            raise ValueError(
                "checkpoint 与网络定义不匹配："
                f"missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}"
            )

        self._torch = torch
        self.device = self._resolve_device(torch, self.config.device)
        self._net = net.to(self.device).eval()
        digest = sha256_of(self.model_path)[:12]
        self.model_version = f"goal1_authenticity/{self.model_path.name}@sha256:{digest}"
        logger.info(
            "goal1_authenticity: 权重就绪 path=%s backbone=%s image_size=%d slices=%d "
            "val_ap=%s device=%s version=%s",
            self.model_path,
            self.backbone,
            self.image_size,
            self.slices_per_case,
            self.val_ap,
            self.device,
            self.model_version,
        )

    @staticmethod
    def _read_checkpoint(torch, path: Path) -> dict:
        try:
            state = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:  # torch < 2.0 没有 weights_only
            state = torch.load(str(path), map_location="cpu")
        if not isinstance(state, dict) or "model" not in state:
            raise ValueError(f"checkpoint 结构不符合本项目约定：{path}")
        return state

    @staticmethod
    def _resolve_device(torch, device: str):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device)

    @property
    def ready(self) -> bool:
        return self._net is not None

    # -- 推理 ---------------------------------------------------------------
    def _logits(self, stack: np.ndarray):
        torch = self._torch
        flat = torch.from_numpy(np.ascontiguousarray(stack))
        outputs = []
        for start in range(0, flat.shape[0], self.config.batch_slices):
            chunk = flat[start:start + self.config.batch_slices].to(self.device, non_blocking=True)
            outputs.append(self._net(chunk).detach().float().cpu())
        return torch.cat(outputs)

    def score_volume(self, volume: np.ndarray | str | Path) -> float | None:
        """一个 3-D 体数据 -> 序列级 ``P(非人体/伪造)``；空数据返回 ``None``。"""
        if self._net is None:
            raise RuntimeError("score_volume 已调用但权重尚未加载（应先执行 load_model()）")
        array = to_float_volume(volume)
        if array.size == 0:
            return None
        stack = volume_to_slices(
            array,
            self.slices_per_case,
            self.image_size,
            self.config.min_std,
        )
        if stack.shape[0] == 0:
            return None
        with self._torch.inference_mode():
            logits = self._logits(stack)
            top_k = max(1, logits.shape[0] // 2)
            return float(self._torch.sigmoid(logits.topk(top_k).values.mean()))

    def score_series(self, series_images: Sequence[np.ndarray]) -> list[float]:
        """便捷入口：按顺序给多条序列打分（异常由调用方处理）。"""
        return [
            value
            for value in (self.score_volume(image) for image in series_images)
            if value is not None
        ]

    def describe(self) -> dict[str, object]:
        return {
            "model_path": str(self.model_path),
            "model_version": self.model_version,
            "backbone": self.backbone,
            "image_size": self.image_size,
            "slices_per_case": self.slices_per_case,
            "frequency_branch": self.frequency_branch,
            "val_ap": self.val_ap,
            "epoch": self.epoch,
            "device": None if self.device is None else str(self.device),
            **self.config.describe(),
        }
