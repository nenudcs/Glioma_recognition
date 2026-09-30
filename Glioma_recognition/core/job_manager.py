"""作业管理与幂等（规范 §13.1）。

规范要求三种冲突必须返回 **HTTP 409**，而不是静默接受：

1. 重复的 ``request_id``（同一请求重复投递）；
2. 已有 evaluation 正在运行（同一 evaluation 只允许一个 Writer）；
3. 该 evaluation 已有正式输出（重复评测会覆盖既有结果）。

当前 ``app/server.py`` 里内联了一份简化实现（重复请求跳过但仍返回 accepted）。
本模块提供符合目标语义的独立实现，供服务层按需切换——之所以**不直接替换**，
是因为 app 层的行为变化必须经 Leader 评审（规范 §17.1）。

线程安全：所有状态变更都在锁内完成，因为 uvicorn 的 worker 与后台执行器会并发
调用（``/call`` 快速返回、后台任务另行更新状态）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


class ConflictError(RuntimeError):
    """请求与既有作业冲突（服务层应转为 HTTP 409）。"""


@dataclass
class Job:
    request_id: str
    evaluation_id: str
    dataset_path: str
    status: str = "accepted"          # accepted | running | done | failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    output_dir: str | None = None
    error: str | None = None


class JobManager:
    """进程内作业表（含幂等与 evaluation 互斥）。"""

    def __init__(self, answer_root: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._by_eval: dict[str, str] = {}
        self._answer_root = Path(answer_root) if answer_root else None

    # ------------------------------------------------------------------ #
    def accept(self, request_id: str, evaluation_id: str, dataset_path: str) -> Job:
        """登记作业；冲突时抛 :class:`ConflictError`（服务层转为 409）。"""
        with self._lock:
            if request_id in self._jobs:
                raise ConflictError(f"duplicate request_id: {request_id}")

            running = self._by_eval.get(evaluation_id)
            if running is not None and self._jobs[running].status in ("accepted", "running"):
                raise ConflictError(
                    f"evaluation {evaluation_id} is already running "
                    f"(request={running}, status={self._jobs[running].status})"
                )

            if self._answer_root is not None:
                out = self._answer_root / evaluation_id
                if out.exists():
                    raise ConflictError(
                        f"evaluation {evaluation_id} already has published output: {out}")

            job = Job(request_id=request_id, evaluation_id=evaluation_id,
                      dataset_path=dataset_path)
            self._jobs[request_id] = job
            self._by_eval[evaluation_id] = request_id
            return job

    def mark(self, request_id: str, status: str, *, output_dir: str | None = None,
             error: str | None = None) -> None:
        """更新作业状态（后台执行器调用）。"""
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                return
            job.status = status
            job.updated_at = time.time()
            if output_dir:
                job.output_dir = output_dir
            if error:
                job.error = error

    def get(self, request_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(request_id)

    def snapshot(self) -> list[dict]:
        """作业快照（供 /status 之类的自检接口使用；平台不要求该接口）。"""
        with self._lock:
            return [{"request_id": j.request_id, "evaluation_id": j.evaluation_id,
                     "status": j.status, "created_at": j.created_at,
                     "updated_at": j.updated_at, "error": j.error}
                    for j in self._jobs.values()]
