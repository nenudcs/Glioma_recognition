from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CompetitionLogger:
    def __init__(self, log_root: Path) -> None:
        self.path = log_root / "inference.jsonl"
        self._lock = threading.Lock()

    def write(
        self,
        *,
        request_id: str,
        evaluation_id: str,
        phase: str,
        message: str,
        data_source: str,
        **details: Any,
    ) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "epoch": None,
            "step": None,
            "phase": phase,
            "mode": "inference",
            "data_source": data_source,
            "request_id": request_id,
            "evaluation_id": evaluation_id,
            "message": message,
            **details,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

