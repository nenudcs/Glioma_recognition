"""[运行] 目标二（拼接影像）配置（规范 §5.2 / §20.1）。

拼接检测是**规则式**检测器，没有需要加载的权重，因此配置只有「几何口径 + 判定阈值 +
概率映射斜率」三项。所有可调项都走环境变量，容器里不改代码：

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| ``GOAL2_STITCHED`` | ``1`` | 是否启用拼接检测 |
| ``GOAL2_STITCHED_METRIC`` | ``curvature`` | 单层差异口径：``curvature`` / ``adjacent`` / ``local`` |
| ``GOAL2_STITCHED_STAT`` | ``max`` | 层分数汇总：``max`` / ``p99`` / ``mean`` |
| ``GOAL2_STITCHED_BAND`` | ``0.1`` | 层方向两端各丢弃的比例 |
| ``GOAL2_STITCHED_THRESHOLD`` | ``0.16`` | 判定阈值，**必须用赛方标定结果替换** |
| ``GOAL2_STITCHED_SCALE`` | ``8.0`` | 概率映射斜率（分数=阈值时概率 0.5） |
| ``GOAL2_STITCHED_GATE`` | ``1`` | 命中拼接后是否对下游任务上闸 |

阈值来源优先级（与 Goal1 的权重路径同一套约定）：

1. 显式传入的构造参数；
2. ``GOAL2_STITCHED_*`` 环境变量；
3. 规范路径 ``<checkpoint_root>/goal2_stitched/calibration.json``
   （由 ``tasks.goal2_stitched.evaluate`` 写出）；
4. 内置默认值。

也就是说：标定一次、把 calibration.json 放到规范路径后，服务不需要再 export 阈值。
标定命令与口径见 ``tasks/goal2_stitched/evaluate.py`` 与本包 README。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import checkpoint as checkpoint_module

STATISTICS = ("max", "p99", "mean")
METRICS = ("curvature", "adjacent", "local")

# 本地（真实体数据 + 人工拼接体数据）标定的经验值；上服务器后必须重新标定。
DEFAULT_THRESHOLD = 0.16
DEFAULT_SCALE = 8.0
DEFAULT_BAND = 0.1


def _env_str(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name)
    return default if value is None or not value.strip() else value.strip()


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    return default if raw is None or not raw.strip() else float(raw)


def _env_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Goal2StitchedConfig:
    """拼接检测运行配置（不可变，便于在多线程环境下共享）。"""

    enabled: bool = True
    metric: str = "curvature"
    statistic: str = "max"
    band: float = DEFAULT_BAND
    threshold: float = DEFAULT_THRESHOLD
    scale: float = DEFAULT_SCALE
    gate: bool = True
    calibration_file: Path | None = None

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric 必须是 {METRICS} 之一")
        if self.statistic not in STATISTICS:
            raise ValueError(f"statistic 必须是 {STATISTICS} 之一")
        if not 0.0 <= self.band < 0.5:
            raise ValueError("band 必须在 [0, 0.5) 区间")
        if self.threshold <= 0:
            raise ValueError("threshold 必须 > 0")
        if self.scale <= 0:
            raise ValueError("scale 必须 > 0")

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        checkpoint_root: str | Path | None = None,
        calibration: Mapping[str, object] | None = None,
    ) -> "Goal2StitchedConfig":
        """构造配置；``calibration=None`` 时自动读取规范路径下的标定文件。"""
        env = os.environ if environ is None else environ
        path = checkpoint_module.calibration_path(checkpoint_root)
        if calibration is not None:
            payload: Mapping[str, object] = calibration
        else:
            payload = checkpoint_module.load_calibration(checkpoint_root) or {}
        return cls(
            enabled=_env_bool(env, "GOAL2_STITCHED", True),
            metric=_env_str(env, "GOAL2_STITCHED_METRIC", str(payload.get("metric", "curvature"))),
            statistic=_env_str(env, "GOAL2_STITCHED_STAT", str(payload.get("statistic", "max"))),
            band=_env_float(
                env, "GOAL2_STITCHED_BAND", _as_float(payload.get("band"), DEFAULT_BAND)
            ),
            threshold=_env_float(
                env,
                "GOAL2_STITCHED_THRESHOLD",
                _as_float(payload.get("threshold"), DEFAULT_THRESHOLD),
            ),
            scale=_env_float(
                env, "GOAL2_STITCHED_SCALE", _as_float(payload.get("scale"), DEFAULT_SCALE)
            ),
            gate=_env_bool(env, "GOAL2_STITCHED_GATE", True),
            calibration_file=path if payload else None,
        )

    def describe(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "metric": self.metric,
            "statistic": self.statistic,
            "band": self.band,
            "threshold": self.threshold,
            "scale": self.scale,
            "gate": self.gate,
            "calibration_file": (
                None if self.calibration_file is None else str(self.calibration_file)
            ),
        }


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
