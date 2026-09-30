"""JSONL 训练/推理日志（严格对齐《赛事开发规范》）。

规范要求：
- JSON Lines，每行一个合法 JSON 对象；日志文件路径必须为 ``/2026aicompetition/workspace/logs``；
- 必填字段：``timestamp``（ISO 8601，示例 ``2025-04-28T10:30:00.123Z``）、
  ``epoch``（**从 1 开始**）、``step``、``phase``（train|val|test）；
- 常用可选：``loss`` / ``lr`` / ``mode``（training|inference）/ ``data_source`` / ``checkpoint``。
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone


def iso_utc_ms() -> str:
    """规范示例格式：2025-04-28T10:30:00.123Z（UTC + 毫秒 + Z）。"""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


class JSONLLogger:
    """线程安全的 JSONL 追加写。``epoch`` 传入 0 基，落盘为 1 基（规范要求）。"""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._lock = threading.Lock()

    def log(self, *, epoch=0, step=0, phase="train", loss=None, lr=None,
            mode="training", data_source="", checkpoint="", **extra) -> dict:
        rec = {
            "timestamp": iso_utc_ms(),
            "epoch": int(epoch) + 1,
            "step": int(step),
            "phase": phase,
            "loss": None if loss is None else round(float(loss), 6),
            "lr": None if lr is None else float(lr),
            "mode": mode,
            "data_source": data_source,
            "checkpoint": checkpoint,
        }
        rec.update(extra)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def run_logger(logs_dir: str, name: str) -> JSONLLogger:
    return JSONLLogger(os.path.join(logs_dir, f"{name}.jsonl"))
