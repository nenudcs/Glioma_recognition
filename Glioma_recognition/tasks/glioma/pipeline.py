"""团队 Pipeline 工厂：把 `glioma_track4` 的真实 Goal 插件接入比赛链路。

```bash
export COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline
```

可选开关（便于按规范"每次只替换一个插件"逐项接入）::

    export GLIOMA_GOALS=goal5,goal3        # 只启用分割与肿瘤概率，其余用 Dummy 补位
    export GLIOMA_CKPT=/path/a.pth,/path/b.pth   # 显式指定权重（多折集成）
"""

from __future__ import annotations

from . import TRACK4_ROOT


def build_pipeline():
    """构建带真实插件的 ``InferencePipeline``。"""
    if TRACK4_ROOT is None:
        raise RuntimeError(
            "未找到算法实现：本仓内置副本 vendor/glioma_track4/ 应随仓库一起提供；"
            "若已被删，请设置 GLIOMA_TRACK4_ROOT 指向算法工程根，"
            "或将其放在本仓库同级目录 / /2026aicompetition/workspace/glioma_track4"
        )
    from integration.factory import build_pipeline as _build      # noqa: PLC0415

    return _build()
