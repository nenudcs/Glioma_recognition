from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from core.config import Settings
from core.exceptions import InvalidRequestError
from core.runner import EvaluationJob, EvaluationRunner


class JobManager:
    def __init__(self, runner: EvaluationRunner, max_workers: int) -> None:
        self.runner = runner
        # ponytail: in-memory state is enough for the platform's single process;
        # use persistent state only if restart-resume becomes a scored requirement.
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="evaluation",
        )
        self._lock = threading.Lock()
        self._request_states: dict[str, str] = {}
        self._active_evaluations: set[str] = set()

    def submit(self, job: EvaluationJob) -> bool:
        with self._lock:
            # 规范 §13.1/§25 要求"重复 request_id"返回 HTTP 409。
            # 原实现是 return False，但调用方并未检查返回值 → 仍回 accepted，
            # 平台会以为请求被受理（§26 已把这列为与目标架构的差距）。
            if job.request_id in self._request_states:
                raise InvalidRequestError(
                    f"request {job.request_id!r} already submitted"
                )
            if job.evaluation_id in self._active_evaluations:
                raise InvalidRequestError(
                    f"evaluation {job.evaluation_id!r} is already running"
                )
            if (self.runner.settings.answer_root / job.evaluation_id).exists():
                raise InvalidRequestError(
                    f"evaluation {job.evaluation_id!r} already has published output"
                )
            self._request_states[job.request_id] = "queued"
            self._active_evaluations.add(job.evaluation_id)
        self.executor.submit(self._execute, job)
        return True

    def _execute(self, job: EvaluationJob) -> None:
        with self._lock:
            self._request_states[job.request_id] = "running"
        try:
            self.runner.run(job)
        except Exception:
            state = "failed"
        else:
            state = "completed"
        finally:
            with self._lock:
                self._request_states[job.request_id] = state
                self._active_evaluations.discard(job.evaluation_id)


settings = Settings.from_env()
runner = EvaluationRunner(settings)
jobs = JobManager(runner, settings.max_workers)
app = FastAPI(title="Track 4 Competition Inference", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "active"}


@app.post("/call")
async def call(request: Request) -> dict[str, str]:
    try:
        payload = await request.json()
        job = _parse_job(payload)
        jobs.submit(job)
    except InvalidRequestError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid request: {exc}") from exc
    return {"status": "accepted", "request_id": job.request_id}


def _parse_job(payload: Any) -> EvaluationJob:
    if not isinstance(payload, dict):
        raise TypeError("JSON body must be an object")
    request_id = _identifier(payload["request_id"], "request_id")
    input_payload = payload["input"]
    if not isinstance(input_payload, dict):
        raise TypeError("input must be an object")
    evaluation_id = _identifier(input_payload["evaluation_id"], "evaluation_id")
    raw_path = input_payload["dataset_path"]
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise TypeError("dataset_path must be a non-empty string")
    dataset_path = Path(raw_path).expanduser()
    if not dataset_path.is_dir():
        raise ValueError(f"dataset_path is not a directory: {dataset_path}")
    return EvaluationJob(request_id, evaluation_id, dataset_path)


def _identifier(value: Any, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{label} must be a string or integer")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} cannot be empty")
    if any(character in text for character in ("/", "\\", "\x00")) or text in {".", ".."}:
        raise ValueError(f"{label} contains unsafe characters")
    return text
