"""Goal1 影像真实性识别 的损失入口（[研发] 文件）。

本 Goal 使用共享多任务损失中的**哪一路**、权重多少，由
``tasks._common.training.cli.GOAL_LOSS_OVERRIDES`` 统一声明；损失的数学
定义在 ``tasks._common.training.losses`` 中单点维护。

本文件导出该 Goal 的**稳定损失入口**：训练引擎与离线评估都调它，
两处若各写一遍迟早会分叉，而分叉不会报错，只让"训练指标"与
"评估指标"对不上。
"""
from __future__ import annotations

from typing import Any, Callable

from tasks._common.training.cli import GOAL_LOSS_OVERRIDES

GOAL = "goal1_authenticity"

#: 本 Goal 在共享多任务损失里的权重（如 ``{"special": 1.0}``）
LOSS_WEIGHTS: dict[str, float] = dict(GOAL_LOSS_OVERRIDES.get(GOAL, {}))


def loss_weights() -> dict[str, float]:
    """返回本 Goal 的损失权重副本。"""
    return dict(LOSS_WEIGHTS)


def cls_spec(raw: dict[str, Any]) -> list[tuple[str, int]]:
    """从 ``config.yaml`` 的 ``labels`` 段推出分类头规格 ``[(字段, 类别数)]``。

    二分类字段的类别数为 1（单个 logit）。
    """
    labels_cfg = raw.get("labels") or {}
    spec = labels_cfg.get("cls_spec") or []
    if not spec and labels_cfg.get("fields"):
        spec = [(f["key"], 1 if f["type"] == "binary" else len(f["classes"]))
                for f in labels_cfg["fields"]]
    return [(str(k), int(n)) for k, n in spec]


def build_loss_fn(raw: dict[str, Any], spec: list[tuple[str, int]] | None = None
                  ) -> Callable[[dict, dict], tuple[Any, dict]]:
    """返回 ``(out, batch) -> (loss_tensor, parts_dict)``。

    直接复用统一训练引擎的 ``compute_losses``：训练与离线评估因此用的是
    **同一套损失**。
    """
    from tasks._common.training.losses import LossWeights, compute_losses

    weights = LossWeights(**{**LossWeights().__dict__, **LOSS_WEIGHTS})
    cls = spec if spec is not None else cls_spec(raw)

    def _fn(out: dict, batch: dict):
        bd = compute_losses(out, batch, cls, weights)
        return bd.total, bd.parts

    return _fn
