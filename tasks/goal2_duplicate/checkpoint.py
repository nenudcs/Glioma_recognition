"""目标二（重复影像）标定产物路径约定。

复用 ``tasks/goal2_stitched/checkpoint.py`` 的 checkpoint 根目录解析（五级优先级），
只把 Goal 名与文件名换成重复检测的这一份：

```text
/2026aicompetition/workspace/checkpoint/
└── goal2_duplicate/calibration.json     # 相似度中心 center / scale / min_sim ...
```

``config.py`` 的取值优先级：显式参数 → ``GOAL2_DUPLICATE_*`` 环境变量 →
规范路径下的 ``calibration.json`` → 内置默认值。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from tasks.goal2_stitched import checkpoint as _shared

GOAL_NAME = "goal2_duplicate"
CALIBRATION_FILENAME = _shared.CALIBRATION_FILENAME

checkpoint_root = _shared.checkpoint_root
_from_core_config = _shared._from_core_config      # noqa: SLF001 - 共享同一套解析
DEFAULT_WORKSPACE = _shared.DEFAULT_WORKSPACE
DEFAULT_CHECKPOINT_ROOT = _shared.DEFAULT_CHECKPOINT_ROOT


def goal_dir(root: str | Path | None = None) -> Path:
    return _shared.goal_dir(GOAL_NAME, root)


def relative_calibration_path() -> Path:
    return _shared.relative_calibration_path(GOAL_NAME)


def calibration_path(root: str | Path | None = None) -> Path:
    """标定产物的规范路径：``<checkpoint_root>/goal2_duplicate/calibration.json``。"""
    return _shared.calibration_path(root, GOAL_NAME)


def load_calibration(root: str | Path | None = None) -> dict[str, Any] | None:
    return _shared.load_calibration(root, GOAL_NAME)


def write_calibration(
    payload: Mapping[str, Any],
    root: str | Path | None = None,
) -> Path:
    return _shared.write_calibration(payload, root, GOAL_NAME)


def candidate_calibrations(
    search_roots: list[str | Path] | None = None,
) -> list[Path]:
    return _shared.candidate_calibrations(search_roots)
