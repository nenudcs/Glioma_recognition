#!/usr/bin/env python3
"""逐阶段内存追踪器：定位"跑第一例就 OOM-kill 容器"的**瞬时峰值**在哪一步。

针对"容器被杀"场景的核心设计：每个阶段**进入时立即打印** ``→ start``（flush）。
容器一旦被 OOM-kill，**日志的最后一行 = 正在执行的阶段** —— 不需要存活就能定位。
后台线程每 5ms 采样 RSS，记录**全局峰值发生在哪个阶段**。

阶段覆盖一例的完整生命周期::

  build_pipeline        加载全部权重为常驻模型（服务启动等价）
  load <acc>            Loader 读入该例全部序列（原始 NIfTI 的真实大小）
  <acc> goal1 … goal5   各 Goal 的 predict（goal5 = 滑窗+TTA+EDT 后处理+写回）
  <acc> write/validate  Writer 落盘 / Validator 校验
  <acc> duplicate-update 数据集级重复影像任务

用法::

    python3 scripts/stage_mem_trace.py 3                        # 前 3 例（默认）
    python3 scripts/stage_mem_trace.py 3 --goals goal5         # 只跑 goal5（二分定位）
    python3 scripts/stage_mem_trace.py 3 --device cpu          # 强制 CPU
    python3 scripts/stage_mem_trace.py 1 --dataset <数据根>

判读::

  · 峰值在 load           → 该例原始 NIfTI 本身过大（shape×dtype 就是 GB 级）
  · 峰值在 goal5          → 公共网格爆炸（看 build_volume 报的 shape）或滑窗/EDT
  · 峰值在 goal1/2/3/4    → global_view / 常驻模型数（G6）
  · 开头提示"CUDA 不可用"  → 全部激活落在容器 RAM，本就是 GB 级
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import psutil

    _PROC = psutil.Process(os.getpid())
except Exception:                                                 # noqa: BLE001
    print("!! 没有 psutil（pip install psutil）—— 本工具依赖它测 RSS")
    raise SystemExit(2)

DEFAULT_DATASET = "/2026aicompetition/datasets/verification/original"

_state = {"label": "startup", "peak": 0.0, "peak_label": "startup"}
_stop = threading.Event()


def rss_mb() -> float:
    return _PROC.memory_info().rss / 1e6


def _sampler() -> None:
    while not _stop.is_set():
        r = rss_mb()
        if r > _state["peak"]:
            _state["peak"] = r
            _state["peak_label"] = _state["label"]
        time.sleep(0.005)


@contextmanager
def trace(label: str):
    prev = _state["label"]
    _state["label"] = label
    start_peak = _state["peak"]
    before = rss_mb()
    print(f"→ {label}", flush=True)          # 进入即打：容器死时最后一行 = 元凶
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        after = rss_mb()
        _state["label"] = prev
        seg_peak = _state["peak"]
        mark = "  ←← 全局新高" if seg_peak > start_peak else ""
        print(f"← {label}   rss {after:7.0f} MB ({after - before:+5.0f})   "
              f"耗时 {dt:6.0f} ms   阶段峰值 {seg_peak:7.0f} MB{mark}", flush=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="逐阶段内存追踪（第一例 OOM 定位）")
    ap.add_argument("n", nargs="?", type=int, default=3, help="跑几例（默认 3）")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--device", default=None, help="覆盖 GLIOMA_DEVICE（如 cpu）")
    ap.add_argument("--goals", default=None,
                    help="只启用这些 Goal（逗号分隔，如 goal5）—— 二分定位用")
    ap.add_argument("--output", default=None, help="答案目录（默认临时目录）")
    a = ap.parse_args(argv)

    if a.goals:
        os.environ["GLIOMA_GOALS"] = a.goals
    if a.device:
        os.environ["GLIOMA_DEVICE"] = a.device
    dataset = Path(a.dataset).expanduser()
    if not dataset.is_dir():
        print(f"!! 数据根不存在: {dataset}")
        return 2

    import torch

    # ---- 环境速览（这些直接决定内存水位）----
    print("=" * 96)
    print("环境")
    print("=" * 96)
    print(f"数据根       : {dataset}")
    print(f"GLIOMA_GOALS : {os.environ.get('GLIOMA_GOALS') or '（默认全开）'}")
    print(f"GLIOMA_DEVICE: {os.environ.get('GLIOMA_DEVICE') or 'cuda'}")
    cuda = torch.cuda.is_available()
    print(f"CUDA 可用    : {cuda}")
    if not cuda:
        print("  ⚠️⚠️ CUDA 不可用 → 模型与全部激活都落在容器 RAM，本就是 GB 级；"
              "先确认评测容器真的分到了 GPU。")
    ckpt_root = os.environ.get("COMPETITION_CHECKPOINT_ROOT") \
        or str(Path(os.environ.get("COMPETITION_WORKSPACE",
                                   "/2026aicompetition/workspace")) / "checkpoint")
    print(f"权重根       : {ckpt_root}")
    ck = Path(ckpt_root)
    if ck.is_dir():
        total = sum(f.stat().st_size for f in ck.rglob("*.pt")) / 1e6
        n_pt = sum(1 for _ in ck.rglob("*.pt"))
        for f in sorted(ck.rglob("*.pt")):
            print(f"    {f.stat().st_size / 1e6:8.0f} MB  {f.relative_to(ck)}")
        print(f"  合计 {n_pt} 份 / {total:.0f} MB"
              + ("   ⚠️ 目录里 *.pt 偏多 —— 每一份都是常驻模型" if n_pt > 8 else ""))
    else:
        print("  !! 权重根不存在")
    v = psutil.virtual_memory()
    print(f"系统内存     : 总 {v.total / 1e9:.1f} GB | 可用 {v.available / 1e9:.1f} GB")
    cg = Path("/sys/fs/cgroup/memory.max")
    if cg.is_file():
        txt = cg.read_text().strip()
        if txt.isdigit():
            print(f"cgroup 上限  : {int(txt) / 1e9:.1f} GB   ← 容器真正能用的上限，超过即被杀")

    # ---- 构建真实 pipeline ----
    from core.config import Settings
    from core.registry import build_pipeline
    from core.runner import EvaluationJob, EvaluationRunner
    from data.loader import DatasetLoader

    threading.Thread(target=_sampler, daemon=True).start()

    settings = Settings.from_env()
    if a.output:
        from dataclasses import replace

        out = Path(a.output).expanduser()
        settings = replace(settings, workspace=out.parent, answer_root=out,
                           log_root=out.parent / "logs", callback_url=None)

    print()
    print("=" * 96)
    print("构建 pipeline（= 服务启动：加载全部权重为常驻模型）")
    print("=" * 96)
    with trace("build_pipeline（加载全部权重）"):
        pipeline = build_pipeline(settings.pipeline_factory)
    try:
        from tasks._common.backbone_runner import shared_weight_count
        print(f"  常驻权重份数 shared_weight_count = {shared_weight_count()}")
    except Exception:                                             # noqa: BLE001
        pass

    # ---- 埋点：每个 Goal 的 predict ----
    for binding in pipeline.study_tasks:
        orig = binding.task.predict
        field = binding.context_field

        def wrapped(context, _orig=orig, _field=field):           # noqa: ANN001, ANN202
            acc = context.study.accession_number
            with trace(f"  {acc} {_field}"):
                return _orig(context)

        binding.task.predict = wrapped                            # type: ignore[method-assign]

    # ---- 埋点：Loader / Writer / Validator / duplicate ----
    class TracedLoader(DatasetLoader):
        n = 0

        def iter_studies(self, path):                             # noqa: ANN001
            for study in super().iter_studies(path):
                type(self).n += 1
                with trace(f"load {study.accession_number}（{len(study.series)} 序列）"):
                    yield study
                if type(self).n >= a.n:
                    return

    runner = EvaluationRunner(settings, loader=TracedLoader(), pipeline=pipeline)
    _ow = runner.writer.write_study
    _ov = runner.validator.validate_study
    _ou = runner.pipeline.update_dataset_task

    def traced_write(study, *args, **kw):                          # noqa: ANN001, ANN202
        with trace(f"  {study.accession_number} write"):
            return _ow(study, *args, **kw)

    def traced_validate(directory, study, *args, **kw):            # noqa: ANN001, ANN202
        with trace(f"  {study.accession_number} validate"):
            return _ov(directory, study, *args, **kw)

    def traced_update(study, context):                            # noqa: ANN001, ANN202
        with trace(f"  {study.accession_number} duplicate-update"):
            return _ou(study, context)

    runner.writer.write_study = traced_write                      # type: ignore[assignment]
    runner.validator.validate_study = traced_validate              # type: ignore[method-assign]
    runner.pipeline.update_dataset_task = traced_update            # type: ignore[assignment]

    # ---- 跑 ----
    print()
    print("=" * 96)
    print(f"开始逐例推理（前 {a.n} 例）—— 卡住/被杀时，最后一行 '→' 就是元凶阶段")
    print("=" * 96)
    t0 = time.perf_counter()
    try:
        runner.run(EvaluationJob(f"trace-{uuid.uuid4()}", "stage-trace", dataset),
                   send_callback=False)
    except Exception as exc:                                      # noqa: BLE001
        print(f"\n!! 推理中断: {type(exc).__name__}: {exc}")
        print("   ↑ 若这是「公共网格 … 超过上限」→ 元凶是病态网格（见报错内的排查步骤）")
    dt = time.perf_counter() - t0

    _stop.set()
    print()
    print("=" * 96)
    print("结论")
    print("=" * 96)
    print(f"全局 RSS 峰值   : {_state['peak']:.0f} MB")
    print(f"峰值发生阶段    : {_state['peak_label']}")
    print(f"总耗时          : {dt:.1f} s（{a.n} 例）")
    print()
    print("按峰值阶段对症：")
    print("  load            → 该例原始 NIfTI 过大：打印各序列 shape/dtype/大小")
    print("  goal5           → 公共网格爆炸或滑窗/EDT：看 goal5 阶段打印前的报错与 shape")
    print("  goal1/2/3/4     → global_view 或常驻模型数：跑 G6（pre_submit_check.py）")
    print("  write/validate  → 掩膜写回/校验的重采样")
    print("  build_pipeline  → 权重本身就装不下（目录里 *.pt 太多）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
