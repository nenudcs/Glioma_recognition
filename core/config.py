from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    workspace: Path
    answer_root: Path
    log_root: Path
    callback_url: str | None
    pipeline_factory: str | None = None
    callback_timeout_seconds: float = 10.0
    callback_attempts: int = 3
    max_workers: int = 1
    # 规范 §5.2：权重根目录 {workspace}/checkpoint。
    # 追加在**末尾并带默认值**——插在中间会破坏既有位置参数构造。
    checkpoint_root: Path | None = None

    @property
    def ckpt_root(self) -> Path:
        """checkpoint 根目录（未显式设置时按规范从 workspace 推导）。"""
        return self.checkpoint_root or (self.workspace / "checkpoint")

    @classmethod
    def from_env(cls) -> "Settings":
        workspace = Path(
            os.environ.get(
                "COMPETITION_WORKSPACE",
                "/2026aicompetition/workspace",
            )
        )
        return cls(
            workspace=workspace,
            answer_root=Path(
                os.environ.get("COMPETITION_ANSWER_ROOT", workspace / "answer")
            ),
            log_root=Path(
                os.environ.get("COMPETITION_LOG_ROOT", workspace / "logs")
            ),
            checkpoint_root=(
                Path(os.environ["COMPETITION_CHECKPOINT_ROOT"])
                if os.environ.get("COMPETITION_CHECKPOINT_ROOT")
                else None
            ),
            callback_url=os.environ.get("COMPETITION_CALLBACK_URL") or None,
            pipeline_factory=os.environ.get("COMPETITION_PIPELINE_FACTORY") or None,
            callback_timeout_seconds=float(
                os.environ.get("COMPETITION_CALLBACK_TIMEOUT", "10")
            ),
            callback_attempts=int(
                os.environ.get("COMPETITION_CALLBACK_ATTEMPTS", "3")
            ),
            max_workers=int(os.environ.get("COMPETITION_MAX_WORKERS", "1")),
        )
