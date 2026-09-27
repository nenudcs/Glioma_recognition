"""[运行] 目标二闸门：命中拼接/重复后不再向下执行下游任务（规范 §7 的「该病人不再进入
后续任务」）。

实现方式对公共框架零改动：

* 检测任务（``goal2_stitched``，排在任务链最前面）命中时把原因写进
  ``context.diagnostics["goal2_gate"]``；
* 下游任务用 :class:`GatedStudyTask` 包一层，看到闸门就直接返回中性结果、**不执行模型**；
* 被跳过的任务记在 ``context.diagnostics["goal2_skipped_tasks"]``，便于赛后核对；
* ``prediction.json`` 的字段结构由管线 Writer 固定，因此闸门效果体现为「下游字段是中性值」。

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| ``GOAL2_GATED_FIELDS`` | ``goal3,goal4,goal5`` | 被闸门拦下的任务（可加 ``goal1``；``goal2`` 是检测器本身，不能拦） |

注意：若要把 ``goal1`` 也纳入闸门，``tasks/real_pipeline.py`` 里必须把 ``goal2_stitched``
绑定在 ``goal1`` **之前**（闸门只能拦它后面的任务）。
"""
from __future__ import annotations

import os
from typing import Callable, Mapping, Sequence

from tasks.base import StudyTask

GATE_KEY = "goal2_gate"
SKIP_KEY = "goal2_skipped_tasks"
DIAGNOSTIC_KEY = "goal2"

DEFAULT_GATED_FIELDS = ("goal3", "goal4", "goal5")
ALLOWED_GATED_FIELDS = ("goal1", "goal3", "goal4", "goal5")


def gated_fields(environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """解析 ``GOAL2_GATED_FIELDS``；显式设置为空串表示「完全不上闸」。"""
    env = os.environ if environ is None else environ
    raw = env.get("GOAL2_GATED_FIELDS")
    if raw is None:
        return DEFAULT_GATED_FIELDS
    fields = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [item for item in fields if item not in ALLOWED_GATED_FIELDS]
    if unknown:
        raise ValueError(
            f"GOAL2_GATED_FIELDS 只能取 {sorted(ALLOWED_GATED_FIELDS)}"
            f"（goal2 是检测器自身），收到 {unknown}"
        )
    return fields


def gate_reason(context) -> str | None:
    """当前检查的闸门原因（``None`` 表示没有上闸）。"""
    gate = getattr(context, "diagnostics", {}).get(GATE_KEY)
    return None if not gate else str(gate.get("reason", "goal2"))


def mark_gate(context, reason: str, detail: Mapping[str, object] | None = None) -> None:
    """写闸门标记（只写诊断信息，不改变任何比赛字段）。"""
    payload: dict[str, object] = {"reason": reason}
    if detail:
        payload.update(detail)
    context.diagnostics[GATE_KEY] = payload
    context.warnings.append(f"goal2 gate: {reason}")


class GatedStudyTask(StudyTask):
    """下游任务闸门包装：命中闸门时返回中性结果，不调用内层模型。"""

    def __init__(
        self,
        inner: StudyTask,
        context_field: str,
        neutral: Callable[[object], object],
        *,
        gated_fields: Sequence[str] = DEFAULT_GATED_FIELDS,
    ) -> None:
        self.inner = inner
        self.context_field = context_field
        self.neutral = neutral
        self.gated_fields = tuple(gated_fields)
        self.name = inner.name

    def load_model(self) -> None:
        self.inner.load_model()

    def predict(self, context):
        if self.context_field in self.gated_fields:
            reason = gate_reason(context)
            if reason:
                skipped = context.diagnostics.setdefault(SKIP_KEY, {})
                skipped[self.context_field] = reason
                return self.neutral(context)
        return self.inner.predict(context)
