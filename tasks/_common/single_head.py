"""共享的"单头 StudyTask"基类。

本工程是一个**多任务共享骨干**：一次前向同时产出分割、结构化、特殊影像与嵌入。
Goal1（真实性）、Goal2-A（拼接）、Goal3（肿瘤）都只是"从该骨干取一个输出头 →
聚合成一个检查级概率"，逻辑完全同构。

规范 §5.1 要求每个 Goal 有自己的 ``task.py`` 与 ``inference.py``（接口不得共用），
因此这里把**实现**收敛到公共基类，各 Goal 只声明"取哪个头、怎么组装 Result"，
既满足"每个 Goal 独立入口"，又避免三份近乎相同的推理代码各自漂移。

**性能关键**：同一个 Study 会被多个插件依次处理。若每个插件各跑一次骨干前向，
一次前向会变成 4 次（实测直接导致 Mock Competition 超时）。
因此这里统一走 :class:`BackboneRunner`，它把前向结果缓存在 ``context.diagnostics``
上，让 Goal1/2-stitched/3/4 **共享同一次计算**，且随 context 释放、不占额外显存。

子类必须定义：``name``、``head``、``build_result()``。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.config import Settings
from tasks._common.backbone_runner import BackboneRunner
from tasks.base import StudyTask


class SingleHeadStudyTask(StudyTask[Any]):
    """从共享骨干取单个输出头的 StudyTask 基类。

    子类需要声明：
        ``name``            —— 任务名（与注册表 key 一致）
        ``head``            —— ``("special", 0|1)`` 或 ``("cls", "TumorProbability")``
        ``ckpt_rel``        —— 相对 checkpoint 根目录的权重路径
        ``build_result(p)`` —— 把 ``[0,1]`` 概率包装成对应 Result

    约定：比赛入口**不写 answer/**、不读比赛根路径、不发回调。
    """

    name: str = "single_head"
    head: tuple[str, int | str] = ("special", 0)
    ckpt_rel: str = "goal5_segmentation/core.pt"
    in_channels: int = 4
    arch: str = "mednext"
    global_size: int = 96
    global_size_mm: float = 192.0
    tta_flips: tuple[str, ...] = ("x", "y")
    tta_batch: int = 2
    common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)

    def __init__(self, settings: Settings | None = None, device: str = "cuda") -> None:
        self.settings = settings or Settings.from_env()
        self.device = device
        self._runner: BackboneRunner | None = None

    # ------------------------------------------------------------------ #
    def _resolve_ckpt(self) -> Path:
        """保留为契约测试的稳定入口（实际解析在 BackboneRunner 内）。"""
        root = Path(self.settings.ckpt_root).expanduser().resolve()
        target = (root / self.ckpt_rel).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"权重路径逃逸出 checkpoint 根目录: {self.ckpt_rel}")
        return target

    def load_model(self) -> None:
        """服务启动时加载一次（规范 §5.2），并建立共享前向。"""
        self._runner = BackboneRunner(
            ckpt_root=self.settings.ckpt_root, ckpt_rel=self.ckpt_rel,
            in_channels=self.in_channels, device=self.device,
            global_size=self.global_size, global_size_mm=self.global_size_mm,
            common_spacing=self.common_spacing, tta_flips=self.tta_flips,
            tta_batch=self.tta_batch,
        )
        self._runner.load()

    # ------------------------------------------------------------------ #
    def _head_value(self, out: dict) -> float:
        """从共享前向结果中取出本任务关心的标量概率。

        ``BackboneRunner`` 返回的是**概率**（已过 sigmoid/softmax），
        因此这里不再二次转换——重复转换会把 0.9 压成 0.71，是个很隐蔽的 bug。
        """
        kind, idx = self.head
        if kind == "special":
            arr = np.asarray(out["special"], dtype=np.float64).ravel()
            i = int(idx)
            return float(arr[i]) if arr.size > i else float("nan")
        if kind == "cls":
            spec = out.get("cls_spec") or []
            for i, (key, _n) in enumerate(spec):
                if key == idx:
                    v = np.asarray(out["cls"][i], dtype=np.float64).ravel()
                    return float(v[0])
            raise KeyError(f"权重中不存在分类字段 {idx!r}（有：{[k for k, _ in spec]}）")
        raise ValueError(f"未知的 head 类型 {kind!r}")

    def predict(self, context) -> Any:
        """单个检查的推理：共享前向（带缓存）→ 取本任务的头 → Result。"""
        if self._runner is None:
            self.load_model()
        assert self._runner is not None

        out = self._runner.forward(context.study, context)
        raw = self._head_value(out)
        prob = float(raw) if np.isfinite(raw) else 0.0
        prob = float(min(1.0, max(0.0, prob)))                     # 规范：概率必须落在 [0,1]

        context.diagnostics[self.name] = {
            "probability": prob,
            "missing_channels": list(out.get("missing_channels") or []),
        }
        return self.build_result(prob)

    # ------------------------------------------------------------------ #
    def build_result(self, probability: float) -> Any:
        """子类实现：把 ``[0,1]`` 概率包装成对应的强类型 Result。"""
        raise NotImplementedError
