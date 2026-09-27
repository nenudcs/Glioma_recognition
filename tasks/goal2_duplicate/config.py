"""[运行] 目标二（重复影像）配置（规范 §5.2 / §20.1）。

赛方定义：重复影像 = 「在原图基础上**轻微调整**且与原临床无差异的复制影像」，
评分口径是**图像对级 AUC-PR**，提交 ``duplicate_pairs.jsonl``（每例 Top-200，
按模型置信度降序截取）。因此配置分成三组：

1. **描述子/检索几何**（``_SLICES`` / ``_COARSE_GRID`` / ``_FINE_GRID`` / ``_SHIFT``）
   —— 一般不改；
2. **判定**（``_SIM_CENTER`` / ``_SIM_SCALE`` / ``_MIN_SIM`` / ``_COARSE_FLOOR`` /
   ``_TOP_K`` / ``_GATE_PROB``）—— ``_SIM_CENTER`` **必须用赛方金标准标定**；
3. **扫描与稀疏性**（``_SCAN`` / ``_SCAN_MAX_VOLUMES`` / ``_SCAN_WORKERS`` /
   ``_REPORT_MIN_PROB`` / ``_MAX_PAIRS``）。

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| ``GOAL2_DUPLICATE`` | ``1`` | 是否启用重复检测 |
| ``GOAL2_DUPLICATE_MODE`` | ``near`` | ``near`` 近似重复检索 / ``exact`` 只认逐像素一致 |
| ``GOAL2_DUPLICATE_SCAN`` | ``1`` | ``1`` 先扫全库建索引 / ``0`` 只与已见过的检查增量比对 |
| ``GOAL2_DUPLICATE_SCAN_MAX_VOLUMES`` | ``0`` | 全库扫描条数上限（0 = 不限） |
| ``GOAL2_DUPLICATE_SCAN_WORKERS`` | ``4`` | 全库扫描并行读盘线程数（只影响速度） |
| ``GOAL2_DUPLICATE_SIM_CENTER`` | ``0.90`` | 相似度→概率中心（**必须标定**） |
| ``GOAL2_DUPLICATE_SIM_SCALE`` | ``0.02`` | 概率映射斜率 |
| ``GOAL2_DUPLICATE_MIN_SIM`` | ``0.50`` | 精排相似度下限（低于不上报） |
| ``GOAL2_DUPLICATE_COARSE_FLOOR`` | ``0.80`` | 粗检索余弦下限（低于不精排） |
| ``GOAL2_DUPLICATE_TOP_K`` | ``20`` | 每个检查最多上报的候选对数 |
| ``GOAL2_DUPLICATE_GATE_PROB`` | ``0.5`` | 达到该概率才判定为重复并上闸 |
| ``GOAL2_DUPLICATE_GATE`` | ``1`` | 命中重复后是否对下游任务上闸 |
| ``GOAL2_DUPLICATE_REPORT_MIN_PROB`` | ``0.0`` | 低于该概率不写进提交文件 |
| ``GOAL2_DUPLICATE_MAX_PAIRS`` | ``200`` | 每例最多提交的候选对数（赛方稀疏性约束） |
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import checkpoint as checkpoint_module

MODES = ("near", "exact")

# 描述子默认几何（经本地合成数据验证：轻微调整副本相似度 0.99+，独立病例 0.68~0.87）。
DEFAULT_SLICES = 3
DEFAULT_COARSE_GRID = 12
DEFAULT_FINE_GRID = 64
DEFAULT_SHIFT = 1
DEFAULT_MIN_SIM = 0.50
DEFAULT_COARSE_FLOOR = 0.80
DEFAULT_CENTER = 0.90
DEFAULT_SCALE = 0.02
DEFAULT_TOP_K = 20
DEFAULT_MAX_PAIRS = 200


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
class Goal2DuplicateConfig:
    """重复影像检索与判定配置（不可变）。"""

    enabled: bool = True
    mode: str = "near"
    scan: bool = True
    scan_max_volumes: int = 0
    scan_workers: int = 4

    min_sim: float = DEFAULT_MIN_SIM
    coarse_floor: float = DEFAULT_COARSE_FLOOR
    center: float = DEFAULT_CENTER
    scale: float = DEFAULT_SCALE
    top_k: int = DEFAULT_TOP_K

    slices: int = DEFAULT_SLICES
    coarse_grid: int = DEFAULT_COARSE_GRID
    fine_grid: int = DEFAULT_FINE_GRID
    shift: int = DEFAULT_SHIFT

    gate: bool = True
    gate_probability: float = 0.5
    report_min_prob: float = 0.0
    max_pairs_per_study: int = DEFAULT_MAX_PAIRS
    calibration_file: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一")
        if self.scan_max_volumes < 0:
            raise ValueError("scan_max_volumes 必须 >= 0")
        if self.scan_workers < 1:
            raise ValueError("scan_workers 必须 >= 1")
        if not 0.0 <= self.min_sim <= 1.0:
            raise ValueError("min_sim 必须在 [0, 1] 区间")
        if not 0.0 <= self.coarse_floor <= 1.0:
            raise ValueError("coarse_floor 必须在 [0, 1] 区间")
        if self.scale <= 0:
            raise ValueError("scale 必须 > 0")
        if self.top_k < 1:
            raise ValueError("top_k 必须 >= 1")
        if not 0.0 < self.gate_probability <= 1.0:
            raise ValueError("gate_probability 必须在 (0, 1] 区间")
        if not 0.0 <= self.report_min_prob <= 1.0:
            raise ValueError("report_min_prob 必须在 [0, 1] 区间")
        if self.max_pairs_per_study < 1:
            raise ValueError("max_pairs_per_study 必须 >= 1")
        self.descriptor_params()          # 触发几何参数校验

    # -- 构造 ---------------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        checkpoint_root: str | Path | None = None,
        calibration: Mapping[str, object] | None = None,
    ) -> "Goal2DuplicateConfig":
        """构造配置；``calibration=None`` 时自动读取规范路径下的标定文件。

        取值优先级：显式构造参数 → ``GOAL2_DUPLICATE_*`` 环境变量 →
        ``<checkpoint_root>/goal2_duplicate/calibration.json`` → 内置默认值。
        """
        env = os.environ if environ is None else environ
        path = checkpoint_module.calibration_path(checkpoint_root)
        if calibration is not None:
            payload: Mapping[str, object] = calibration
        else:
            payload = checkpoint_module.load_calibration(checkpoint_root) or {}
        return cls(
            enabled=_env_bool(env, "GOAL2_DUPLICATE", True),
            mode=_env_str(env, "GOAL2_DUPLICATE_MODE", str(payload.get("mode", "near"))),
            scan=_env_bool(env, "GOAL2_DUPLICATE_SCAN", True),
            scan_max_volumes=_env_int(env, "GOAL2_DUPLICATE_SCAN_MAX_VOLUMES", 0),
            scan_workers=_env_int(env, "GOAL2_DUPLICATE_SCAN_WORKERS", 4),
            min_sim=_env_float(
                env, "GOAL2_DUPLICATE_MIN_SIM", _as_float(payload.get("min_sim"), DEFAULT_MIN_SIM)
            ),
            coarse_floor=_env_float(
                env,
                "GOAL2_DUPLICATE_COARSE_FLOOR",
                _as_float(payload.get("coarse_floor"), DEFAULT_COARSE_FLOOR),
            ),
            center=_env_float(
                env, "GOAL2_DUPLICATE_SIM_CENTER", _as_float(payload.get("center"), DEFAULT_CENTER)
            ),
            scale=_env_float(
                env, "GOAL2_DUPLICATE_SIM_SCALE", _as_float(payload.get("scale"), DEFAULT_SCALE)
            ),
            top_k=_env_int(
                env, "GOAL2_DUPLICATE_TOP_K", _as_int(payload.get("top_k"), DEFAULT_TOP_K)
            ),
            slices=_env_int(
                env, "GOAL2_DUPLICATE_SLICES", _as_int(payload.get("slices"), DEFAULT_SLICES)
            ),
            coarse_grid=_env_int(
                env,
                "GOAL2_DUPLICATE_COARSE_GRID",
                _as_int(payload.get("coarse_grid"), DEFAULT_COARSE_GRID),
            ),
            fine_grid=_env_int(
                env,
                "GOAL2_DUPLICATE_FINE_GRID",
                _as_int(payload.get("fine_grid"), DEFAULT_FINE_GRID),
            ),
            shift=_env_int(
                env, "GOAL2_DUPLICATE_SHIFT", _as_int(payload.get("shift"), DEFAULT_SHIFT)
            ),
            gate=_env_bool(env, "GOAL2_DUPLICATE_GATE", True),
            gate_probability=_env_float(
                env,
                "GOAL2_DUPLICATE_GATE_PROB",
                _as_float(payload.get("gate_probability"), 0.5),
            ),
            report_min_prob=_env_float(
                env, "GOAL2_DUPLICATE_REPORT_MIN_PROB", 0.0
            ),
            max_pairs_per_study=_env_int(
                env,
                "GOAL2_DUPLICATE_MAX_PAIRS",
                _as_int(payload.get("max_pairs_per_study"), DEFAULT_MAX_PAIRS),
            ),
            calibration_file=path if payload else None,
        )

    # -- 派生 ---------------------------------------------------------------
    def descriptor_params(self):
        """描述子几何参数（延迟 import，避免 config 依赖 numpy）。"""
        from .preprocess import DescriptorParams

        return DescriptorParams(
            slices=self.slices,
            coarse_grid=self.coarse_grid,
            fine_grid=self.fine_grid,
            shift=self.shift,
        )

    def index_kwargs(self) -> dict[str, object]:
        """传给 ``retrieval.NearDuplicateIndex`` 的参数。"""
        return {
            "top_k": self.top_k,
            "min_sim": self.min_sim,
            "coarse_floor": self.coarse_floor,
            "center": self.center,
            "scale": self.scale,
        }

    def gate_similarity(self) -> float:
        """闸门概率对应的相似度阈值（用于日志与文档）。"""
        from .postprocess import similarity_from_probability

        return similarity_from_probability(
            self.gate_probability, self.center, self.scale
        )

    def describe(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "scan_whole_dataset": self.scan,
            "scan_max_volumes": self.scan_max_volumes,
            "scan_workers": self.scan_workers,
            "min_similarity": self.min_sim,
            "coarse_floor": self.coarse_floor,
            "center": self.center,
            "scale": self.scale,
            "top_k": self.top_k,
            "gate": self.gate,
            "gate_probability": self.gate_probability,
            "gate_similarity": round(self.gate_similarity(), 6),
            "report_min_prob": self.report_min_prob,
            "max_pairs_per_study": self.max_pairs_per_study,
            "descriptor": {
                "slices": self.slices,
                "coarse_grid": self.coarse_grid,
                "fine_grid": self.fine_grid,
                "shift": self.shift,
            },
            "calibration_file": (
                None if self.calibration_file is None else str(self.calibration_file)
            ),
        }


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)            # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
