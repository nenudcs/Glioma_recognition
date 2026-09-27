"""目标二 checkpoint 根目录与标定产物路径约定（规范 §5.2 / §20.1 的类比物）。

Goal2 是规则式检测器，没有权重文件；需要集中管理的运行期外部产物是**标定结果**
（拼接判定阈值、重复相似度中心等）。规范要求每个 Goal 的外部产物放在同一根目录下，
且「``core/config.py`` 提供 checkpoint 根目录，各 Goal 的 ``config.py`` 只声明相对路径」：

```text
/2026aicompetition/workspace/checkpoint/
├── goal2_stitched/calibration.json      # 拼接阈值 + 口径
└── goal2_duplicate/calibration.json     # 相似度中心 + 检索参数
```

根目录解析顺序（与 ``goal1_authenticity/checkpoint.py`` 完全一致，便于将来切到
``core/config.py`` 时无需改调用方）：

1. 显式传入的参数（CLI 的 ``--checkpoint-root``）；
2. ``core.config.Settings`` 上的 ``checkpoint_root`` / ``checkpoint_dir``（若已支持）；
3. 环境变量 ``GOAL2_CHECKPOINT_ROOT`` → ``CHECKPOINT_ROOT``；
4. 环境变量 ``COMPETITION_WORKSPACE``（默认 ``/2026aicompetition/workspace``）下的 ``checkpoint/``；
5. 默认值 ``/2026aicompetition/workspace/checkpoint``。

本模块同时被 ``tasks/goal2_duplicate/checkpoint.py`` 复用（它只改 Goal 名与文件名），
避免两处各写一套路径解析。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

GOAL_NAME = "goal2_stitched"
CALIBRATION_FILENAME = "calibration.json"

DEFAULT_WORKSPACE = Path("/2026aicompetition/workspace")
DEFAULT_CHECKPOINT_ROOT = DEFAULT_WORKSPACE / "checkpoint"


def _from_core_config() -> Path | None:
    """若 ``core/config.py`` 已按规范提供 checkpoint 根目录，就优先使用它。"""
    try:
        from core.config import Settings  # type: ignore import-not-found
    except Exception:  # noqa: BLE001 - 未提供时静默回落
        return None
    names = ("checkpoint_root", "checkpoint_dir", "CHECKPOINT_ROOT")
    for name in names:
        value = getattr(Settings, name, None)
        if isinstance(value, Path):
            return value
    try:
        settings = Settings.from_env()
    except Exception:  # noqa: BLE001
        return None
    for name in names:
        value = getattr(settings, name, None)
        if isinstance(value, Path):
            return value
    return None


def checkpoint_root(explicit: str | Path | None = None) -> Path:
    """解析 checkpoint 根目录（见模块文档的 5 级优先级）。"""
    if explicit:
        return Path(explicit).expanduser()
    from_core = _from_core_config()
    if from_core is not None:
        return from_core
    for name in ("GOAL2_CHECKPOINT_ROOT", "CHECKPOINT_ROOT"):
        raw = os.environ.get(name)
        if raw and raw.strip():
            return Path(raw.strip()).expanduser()
    workspace = os.environ.get("COMPETITION_WORKSPACE")
    if workspace and workspace.strip():
        return Path(workspace.strip()).expanduser() / "checkpoint"
    return DEFAULT_CHECKPOINT_ROOT


def goal_dir(goal_name: str = GOAL_NAME, root: str | Path | None = None) -> Path:
    return checkpoint_root(root) / goal_name


def relative_calibration_path(goal_name: str = GOAL_NAME) -> Path:
    """给 ``config.py`` 用的相对路径（规范：各 Goal 只声明相对路径）。"""
    return Path(goal_name) / CALIBRATION_FILENAME


def calibration_path(
    root: str | Path | None = None,
    goal_name: str = GOAL_NAME,
) -> Path:
    """标定产物的规范路径：``<checkpoint_root>/<goal>/calibration.json``。"""
    return goal_dir(goal_name, root) / CALIBRATION_FILENAME


def load_calibration(
    root: str | Path | None = None,
    goal_name: str = GOAL_NAME,
) -> dict[str, Any] | None:
    """读取规范路径下的标定文件；文件不存在或内容损坏时返回 ``None``（回落默认值）。"""
    path = calibration_path(root, goal_name)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 损坏的标定文件不应让服务起不来
        return None
    return payload if isinstance(payload, dict) else None


def write_calibration(
    payload: Mapping[str, Any],
    root: str | Path | None = None,
    goal_name: str = GOAL_NAME,
) -> Path:
    """把标定结果写到规范路径（``evaluate.py`` 的默认落点）。"""
    path = calibration_path(root, goal_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def candidate_calibrations(search_roots: list[str | Path] | None = None) -> list[Path]:
    """发现工作区里已有的 ``calibration.json``（方便迁移到规范路径）。"""
    roots: list[Path] = []
    if search_roots:
        roots.extend(Path(item).expanduser() for item in search_roots)
    else:
        workspace = Path(
            os.environ.get("COMPETITION_WORKSPACE", DEFAULT_WORKSPACE)
        ).expanduser()
        roots.append(workspace)
        run_dir = os.environ.get("GOAL2_RUN_DIR") or os.environ.get("TASK1_RUN_DIR")
        if run_dir and run_dir.strip():
            roots.append(Path(run_dir.strip()).expanduser())

    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for pattern in ("*calibration*.json", "*/calibration.json", "*/*calibration*.json"):
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    found[str(path.resolve())] = path.resolve()
    return [found[key] for key in sorted(found)]
