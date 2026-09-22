"""[运行] 目标一配置（规范 §5.2 / §20.1）。

规范要求：权重只声明**相对路径**，checkpoint 根目录由 ``core/config.py`` 提供。
当前 ``core/config.py`` 尚未提供该根目录（规范 §26 差距清单），所以这里：

* 相对路径固定为 ``goal1_authenticity/model.pt``（见 ``checkpoint.relative_model_path()``）；
* 根目录由 ``checkpoint.checkpoint_root()`` 按「显式参数 → core.config → 环境变量 → 默认」
  的优先级解析，将来 core 支持后无需改本文件；
* 过渡期允许 ``GOAL1_CHECKPOINT`` 直接指向一个具体的 .pt 文件（仅用于迁移前后自测）。

所有可调项都可通过环境变量覆盖，便于容器内不改代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from . import checkpoint as checkpoint_module

DEFAULT_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"
SERIES_AGGREGATIONS = ("max", "mean")


def _env_str(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    return default if raw is None or not raw.strip() else int(raw)


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    return default if raw is None or not raw.strip() else float(raw)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Goal1Config:
    """目标一运行配置。

    ``checkpoint_root`` 为 ``None`` 时按 ``checkpoint.checkpoint_root()`` 的优先级解析；
    ``slices_per_case`` / ``image_size`` 为 0 表示沿用权重文件里记录的训练值。
    """

    checkpoint_root: Path | None = None
    checkpoint_relative_path: Path = field(default_factory=checkpoint_module.relative_model_path)
    device: str = "auto"
    batch_slices: int = 32
    slices_per_case: int = 0
    image_size: int = 0
    min_std: float = 0.05
    series_aggregation: str = "max"
    strict: bool = False

    def __post_init__(self) -> None:
        if self.batch_slices < 1:
            raise ValueError("batch_slices 必须 >= 1")
        if self.slices_per_case < 0 or self.image_size < 0:
            raise ValueError("slices_per_case / image_size 必须 >= 0")
        if self.series_aggregation not in SERIES_AGGREGATIONS:
            raise ValueError(f"series_aggregation 必须是 {SERIES_AGGREGATIONS} 之一")

    # -- 路径 ---------------------------------------------------------------
    @property
    def resolved_model_path(self) -> Path:
        """最终权重文件路径：``GOAL1_CHECKPOINT`` 覆盖 > 规范相对路径。"""
        override = os.environ.get("GOAL1_CHECKPOINT")
        if override and override.strip():
            return Path(override.strip()).expanduser()
        root = checkpoint_module.checkpoint_root(self.checkpoint_root)
        return root / self.checkpoint_relative_path

    @property
    def resolved_checkpoint_root(self) -> Path:
        return checkpoint_module.checkpoint_root(self.checkpoint_root)

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Goal1Config":
        env = os.environ if environ is None else environ
        raw_root = env.get("GOAL1_CHECKPOINT_ROOT") or env.get("CHECKPOINT_ROOT")
        return cls(
            checkpoint_root=Path(raw_root.strip()).expanduser() if raw_root and raw_root.strip() else None,
            checkpoint_relative_path=Path(
                _env_str(env, "GOAL1_CHECKPOINT_RELATIVE", str(checkpoint_module.relative_model_path()))
            ),
            device=_env_str(env, "GOAL1_DEVICE", "auto"),
            batch_slices=_env_int(env, "GOAL1_BATCH_SLICES", 32),
            slices_per_case=_env_int(env, "GOAL1_SLICES_PER_CASE", 0),
            image_size=_env_int(env, "GOAL1_IMAGE_SIZE", 0),
            min_std=_env_float(env, "GOAL1_MIN_STD", 0.05),
            series_aggregation=_env_str(env, "GOAL1_SERIES_AGGREGATION", "max"),
            strict=_env_bool(env, "GOAL1_STRICT", False),
        )

    def describe(self) -> dict[str, object]:
        return {
            "checkpoint_root": str(self.resolved_checkpoint_root),
            "checkpoint_relative_path": str(self.checkpoint_relative_path),
            "model_path": str(self.resolved_model_path),
            "device": self.device,
            "batch_slices": self.batch_slices,
            "slices_per_case": self.slices_per_case or "from-checkpoint",
            "image_size": self.image_size or "from-checkpoint",
            "min_std": self.min_std,
            "series_aggregation": self.series_aggregation,
            "strict": self.strict,
        }
