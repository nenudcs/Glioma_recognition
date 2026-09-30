"""推理服务（严格按《赛事开发规范（赛道四）》）。

- ``POST /call``：body 为规范嵌套结构
  ``{"request_id","team_id","track_code","input":{"evaluation_id","dataset_path"}}``；
  **立即返回 200**（平台 5s 超时），推理在后台线程完成；
- ``GET /health``：平台启动探针（5s/次，最长 1800s）与存活检测，必须常回 200；
- 结果写 ``{answer_root}/{evaluation_id}/``，完成后**回调**平台
  ``{platform}/api/competition/inference/callback/``，body ``{request_id, evaluationId, predPath}``。

启动（前台常驻）：``python -m src.serving.app --port 8000``
"""
from __future__ import annotations

import os
import threading
import time
import uuid

import requests
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from ..utils.config import PROJECT_ROOT, data_source_tag, load_paths
from ..utils.logger import run_logger

app = FastAPI(title="track4-glioma-serving")
PATHS = load_paths()
LOGGER = run_logger(PATHS["logs_dir"], "serving")
_PIPE = None
_LOCK = threading.Lock()
_SEM = threading.Semaphore(int(os.environ.get("MAX_CONCURRENT_JOBS", "2")))
_JOBS: dict[str, dict] = {}

#: 推理权重：多折用逗号分隔（如 "ckpt/f0/best.pth,ckpt/f1/best.pth"）
CKPT_ENV = "GLIOMA_CKPT"

#: 回调地址的可用来源（按优先级）。平台会把完整回调地址显示在容器实例页面上方，
#: 选手需将其传入其中之一。**缺少回调地址会导致该次测评无法闭环、直接不计分**，
#: 因此这里做了多来源解析并会在日志中显式报错（而不是静默跳过）。
CALLBACK_ENVS = ("CALLBACK_URL", "PLATFORM_CALLBACK_URL", "COMPETITION_CALLBACK_URL",
                 "INFERENCE_CALLBACK_URL", "CALLBACK_ADDR", "CALLBACK")


def resolve_callback_url() -> str:
    """按 环境变量 → 文件 的顺序解析回调地址（每次调用都重新解析，支持后写入）。"""
    for k in CALLBACK_ENVS:
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    ws = os.environ.get("WORKSPACE") or PATHS.get("workspace") or "/2026aicompetition/workspace"
    cands = [os.path.join(ws, "callback_url.txt"), os.path.join(ws, "callback.txt"),
             os.path.join(PROJECT_ROOT, "callback_url.txt"),
             os.path.join(PROJECT_ROOT, "configs", "callback_url.txt")]
    for p in cands:
        try:
            if os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    v = f.read().strip()
                if v:
                    return v
        except OSError:
            continue
    return ""


class CallInput(BaseModel):
    evaluation_id: str = ""
    dataset_path: str = ""


class CallRequest(BaseModel):
    """规范为主路径（嵌套 input）；同时兼容扁平写法便于本地自测。"""
    request_id: str = ""
    team_id: str = ""
    track_code: str = ""
    input: CallInput = CallInput()
    evaluation_id: str = ""
    evaluationId: str = ""
    dataset_path: str = ""
    data_path: str = ""

    @property
    def eval_id(self) -> str:
        return str(self.input.evaluation_id or self.evaluation_id or self.evaluationId or "")

    @property
    def dataset(self) -> str:
        return self.input.dataset_path or self.dataset_path or self.data_path


def get_pipeline():
    global _PIPE
    with _LOCK:
        if _PIPE is None:
            from ..inference.pipeline import GliomaPipeline
            ckpts = [c for c in os.environ.get(CKPT_ENV, "").replace(";", ",").split(",") if c.strip()]
            if not ckpts:
                raise RuntimeError(
                    f"[serving] 未指定推理权重：请设 {CKPT_ENV}=路径（多折用逗号分隔）。\n"
                    "  · 训练产物默认在 checkpoints/g4_fold*/best.pth\n"
                    "  · 平台持久化目录建议 /2026aicompetition/workspace/train_model/")
            _PIPE = GliomaPipeline([c.strip() for c in ckpts])
    return _PIPE


def answer_dir(evaluation_id: str) -> str:
    root = PATHS.get("answer_root") or "/2026aicompetition/workspace/answer"
    out = os.path.join(root, str(evaluation_id))
    try:
        os.makedirs(out, exist_ok=True)
        return out
    except OSError:
        local = os.path.join(PATHS["uif_root"], "..", "answer", str(evaluation_id))
        os.makedirs(local, exist_ok=True)
        return local


@app.get("/health")
def health():
    """平台启动探针（5s/次，最长 1800s）与探活探针（30s/次）共用；必须常回 200。"""
    return {"status": "active", "timestamp": time.time(),
            "weights_loaded": bool(os.environ.get(CKPT_ENV)),
            "callback_ready": bool(resolve_callback_url())}


@app.post("/call")
def call(req: CallRequest, background: BackgroundTasks):
    """规范推理入口：**立即返回 200**（平台 5s 超时），推理在后台线程完成。"""
    request_id = req.request_id or str(uuid.uuid4())
    evaluation_id = req.eval_id
    dataset_path = req.dataset
    if not evaluation_id or not dataset_path:
        LOGGER.log(phase="test", mode="inference", data_source=data_source_tag(dataset_path, phase="test"),
                   request_id=request_id, error="missing evaluation_id/dataset_path")
        return {"code": 400, "msg": "缺少 input.evaluation_id / input.dataset_path"}, 400
    if not os.path.isdir(dataset_path):
        LOGGER.log(phase="test", mode="inference", data_source=data_source_tag(dataset_path, phase="test"),
                   request_id=request_id, error=f"dataset_path 不存在: {dataset_path}")
        return {"code": 400, "msg": f"dataset_path 不存在: {dataset_path}"}, 400

    out_dir = answer_dir(evaluation_id)
    _JOBS[request_id] = {"status": "running", "evaluation_id": evaluation_id, "out": out_dir}
    LOGGER.log(phase="test", mode="inference", data_source=data_source_tag(dataset_path, phase="test"),
               request_id=request_id, evaluation_id=evaluation_id, dataset_path=dataset_path)
    print(f"[serving] /call 受理 request_id={request_id} evaluation_id={evaluation_id} "
          f"dataset={dataset_path}", flush=True)
    background.add_task(run_job, request_id, evaluation_id, dataset_path, out_dir)
    return {"code": 200, "msg": "accepted", "request_id": request_id}


@app.get("/status/{request_id}")
def status(request_id: str):
    """自测用：查看某次请求的进度（平台不要求此接口）。"""
    return _JOBS.get(request_id, {"status": "unknown"})


def run_job(request_id: str, evaluation_id: str, dataset_path: str, out_dir: str) -> None:
    try:
        with _SEM:
            pipe = get_pipeline()
            rep = pipe.run_batch(dataset_path, out_dir)
        _JOBS[request_id] = {"status": "done", "out": out_dir, **rep}
        LOGGER.log(phase="test", mode="inference", data_source=data_source_tag(dataset_path, phase="test"),
                   request_id=request_id, evaluation_id=evaluation_id, ok=rep["validate"]["ok"])
        _JOBS[request_id]["callback_url"] = resolve_callback_url() or None
        callback(request_id, evaluation_id, out_dir)
    except Exception as e:  # noqa: BLE001
        _JOBS[request_id] = {"status": "failed", "error": str(e)}
        LOGGER.log(phase="test", mode="inference", data_source=data_source_tag(dataset_path, phase="test"),
                   request_id=request_id, error=str(e))


def callback(request_id: str, evaluation_id: str, out_dir: str) -> None:
    """规范回调：{request_id, evaluationId, predPath}（必须回传 /call 收到的两个 ID）。"""
    url = resolve_callback_url()
    if not url:
        # **不能静默**：缺少回调地址 = 该次测评无法闭环 = 不计分
        LOGGER.log(phase="test", mode="inference", request_id=request_id,
                   callback="FAILED_no_callback_url",
                   error="未解析到回调地址：请在启动命令中传入 CALLBACK_URL=... "
                         "（地址见容器实例页面上方），或写入 {ws}/callback_url.txt")
        print("[serving] ✗✗ 未解析到回调地址，无法回调平台（本次测评可能不计分）。"
              "请设置 CALLBACK_URL 环境变量或写入 workspace/callback_url.txt", flush=True)
        return
    body = {"request_id": request_id, "evaluationId": evaluation_id,
            "evaluation_id": evaluation_id, "predPath": out_dir}
    try:
        body["evaluationId"] = int(evaluation_id)          # 规范示例为数值
    except (TypeError, ValueError):
        pass
    for attempt in range(3):
        try:
            requests.post(url, json=body, timeout=10)
            LOGGER.log(phase="test", mode="inference", request_id=request_id, callback="ok")
            return
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                LOGGER.log(phase="test", mode="inference", request_id=request_id,
                           callback=f"error:{e}")


if __name__ == "__main__":
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()

    # ---- 启动自检：回调地址（缺失 = 不计分，必须打在最显眼处）----
    _cb = resolve_callback_url()
    if _cb:
        print(f"[serving] ✓ 回调地址: {_cb}", flush=True)
    else:
        print("=" * 78, flush=True)
        print("[serving] ⚠️⚠️ 未解析到回调地址：推理完成后**无法回调平台，本次测评可能不计分**！",
              flush=True)
        print("[serving]   解决办法（任选其一）：", flush=True)
        print("[serving]     1) 启动命令加： CALLBACK_URL='http://<平台>/api/competition/inference/callback/'",
              flush=True)
        print(f"[serving]     2) 把地址写入文件： {PATHS.get('workspace')}/callback_url.txt", flush=True)
        print("[serving]   地址见「容器实例页面」上方的完整回调地址。", flush=True)
        print("=" * 78, flush=True)

    # 启动预检：把"权重能否加载"写进日志；失败也照常起服务（否则平台 /health 探测不过 → 容器起不来）
    try:
        get_pipeline()
        print("[serving] 启动预检通过：权重已加载，服务就绪", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[serving] ⚠️ 启动预检失败（服务仍会启动，/call 会报错）：\n{e}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port)
