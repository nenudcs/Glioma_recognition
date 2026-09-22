"""目标一 checkpoint 路径约定（规范 §5.2 / §20.1）。

规范要求权重固定放在：

```text
/2026aicompetition/workspace/checkpoint/
└── goal1_authenticity/model.pt
```

并规定「``core/config.py`` 提供 checkpoint 根目录，各 Goal 的 ``config.py`` 只声明
相对路径」。当前仓库的 ``core/config.py`` 尚未提供该根目录（规范 §26 差距清单），
因此这里按下面的顺序解析，保证现在就能跑、将来切到 core 配置时无需改调用方：

1. 显式传入的参数（CLI 的 ``--checkpoint-root``）；
2. ``core.config.Settings`` 上的 ``checkpoint_root`` / ``checkpoint_dir``（若已支持）；
3. 环境变量 ``GOAL1_CHECKPOINT_ROOT`` → ``CHECKPOINT_ROOT``；
4. 环境变量 ``COMPETITION_WORKSPACE``（默认 ``/2026aicompetition/workspace``）下的 ``checkpoint/``；
5. 默认值 ``/2026aicompetition/workspace/checkpoint``。
"""
from __future__ import annotations

import os
from pathlib import Path

GOAL_NAME = "goal1_authenticity"
MODEL_FILENAME = "model.pt"
LAST_FILENAME = "model_last.pt"
META_FILENAME = "model_meta.json"

DEFAULT_WORKSPACE = Path("/2026aicompetition/workspace")
DEFAULT_CHECKPOINT_ROOT = DEFAULT_WORKSPACE / "checkpoint"


def _from_core_config() -> Path | None:
    """若 ``core/config.py`` 已按规范提供 checkpoint 根目录，就优先使用它。"""
    try:
        from core.config import Settings  # type: ignore import-not-found
    except Exception:  # noqa: BLE001 - 未安装/未提供时静默回落
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
    for name in ("GOAL1_CHECKPOINT_ROOT", "CHECKPOINT_ROOT"):
        raw = os.environ.get(name)
        if raw and raw.strip():
            return Path(raw.strip()).expanduser()
    workspace = os.environ.get("COMPETITION_WORKSPACE")
    if workspace and workspace.strip():
        return Path(workspace.strip()).expanduser() / "checkpoint"
    return DEFAULT_CHECKPOINT_ROOT


def goal_dir(root: str | Path | None = None) -> Path:
    return checkpoint_root(root) / GOAL_NAME


def model_path(root: str | Path | None = None) -> Path:
    """规范的权重路径：``<checkpoint_root>/goal1_authenticity/model.pt``。"""
    return goal_dir(root) / MODEL_FILENAME


def last_model_path(root: str | Path | None = None) -> Path:
    return goal_dir(root) / LAST_FILENAME


def meta_path(root: str | Path | None = None) -> Path:
    return goal_dir(root) / META_FILENAME


def relative_model_path() -> Path:
    """给 ``config.py`` 用的相对路径（规范：各 Goal 只声明相对路径）。"""
    return Path(GOAL_NAME) / MODEL_FILENAME


def candidate_sources(search_roots: list[str | Path] | None = None) -> list[Path]:
    """发现可能的历史训练产物（迁移程序的默认输入）。

    覆盖常见布局：``<root>/*/best.pt``、``<root>/task1_runs/*/best.pt``，
    以及环境变量 ``TASK1_RUN_DIR`` 指向的目录。
    """
    roots: list[Path] = []
    if search_roots:
        roots.extend(Path(item).expanduser() for item in search_roots)
    else:
        workspace = Path(os.environ.get("COMPETITION_WORKSPACE", DEFAULT_WORKSPACE)).expanduser()
        roots.append(workspace)
        run_dir = os.environ.get("TASK1_RUN_DIR")
        if run_dir and run_dir.strip():
            roots.append(Path(run_dir.strip()).expanduser())

    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        patterns = ("best.pt", "*/best.pt", "task1_runs/*/best.pt", "*/*/best.pt")
        for pattern in patterns:
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    found[str(path.resolve())] = path.resolve()
    return [found[key] for key in sorted(found)]
