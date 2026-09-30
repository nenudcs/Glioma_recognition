#!/usr/bin/env python3
"""逐 Study 内存探针：测"处理 N 个检查时 RSS 是否线性增长"。

**为什么单测发现不了**：测试套件里每批只有 2~3 例，
而一个"每 Study 泄漏几百 KB~几 MB"的问题在 777 例下就是几 GB ——
测试全绿、评测 OOM，正是最难查的一类。

用法::

    python3 scripts/study_leak_probe.py                    # 50 例（默认）
    python3 scripts/study_leak_probe.py 200                # 200 例
    python3 scripts/study_leak_probe.py 50 --pipeline dummy

判读：打印每例的 RSS 与"每例增量"；末尾用**前后半段斜率对比**给结论。
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import psutil

    _PROC = psutil.Process(os.getpid())
except Exception:                                                 # noqa: BLE001
    _PROC = None


def rss_mb() -> float:
    return _PROC.memory_info().rss / 1e6 if _PROC else float("nan")


def make_dataset(root: Path, n: int) -> None:
    """最小合成数据集：每例两个序列，体积很小（内存压力只来自框架本身）。"""
    import nibabel as nib

    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = 1.2, 1.2, 3.0
    for i in range(n):
        acc = f"ACC{i:04d}"
        for uid in ("T1CE", "FLAIR"):
            d = root / acc / uid
            d.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(np.full((6, 7, 4), float(i % 7), np.float32), aff),
                     str(d / f"{uid}.nii.gz"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="逐 Study 内存探针")
    ap.add_argument("n", nargs="?", type=int, default=50, help="检查数（默认 50）")
    ap.add_argument("--pipeline", default="dummy", choices=["dummy", "real"],
                    help="dummy=空跑框架（默认）；real=用注册的真实插件（需权重）")
    ap.add_argument("--every", type=int, default=10, help="每 N 例打印一行")
    a = ap.parse_args(argv)

    if _PROC is None:
        print("!! 没有 psutil（pip install psutil）")
        return 2

    tmp = Path(tempfile.mkdtemp())
    ds = tmp / "dataset"
    make_dataset(ds, a.n)
    print(f"合成数据集: {a.n} 例 -> {ds}")

    from dataclasses import replace

    from core.config import Settings
    from core.runner import EvaluationJob, EvaluationRunner

    settings = replace(
        Settings.from_env(),
        workspace=tmp / "ws", answer_root=tmp / "ws" / "answer",
        log_root=tmp / "ws" / "logs", callback_url=None,
    )
    if a.pipeline == "dummy":
        # 强制走 Dummy：内存压力只来自框架（Loader/Writer/Validator/Logger/Aggregator）
        settings = replace(settings, pipeline_factory=None)
    else:
        print(f"插件工厂: {settings.pipeline_factory}  权重根: {settings.ckpt_root}")

    runner = EvaluationRunner(settings)
    print(f"起始 RSS = {rss_mb():.0f} MB")
    print("=" * 78)
    print(f"{'处理例数':>8}  {'RSS(MB)':>10}  {'本段增量':>10}  {'累计/例':>10}")

    samples: list[tuple[int, float]] = []
    # 用一层包装在每例结束后采样
    orig_write = runner.writer.write_study
    counter = {"n": 0}

    def counted(study, *args, **kwargs):                          # noqa: ANN001, ANN002
        out = orig_write(study, *args, **kwargs)
        counter["n"] += 1
        if counter["n"] % a.every == 0 or counter["n"] == a.n:
            samples.append((counter["n"], rss_mb()))
        return out

    runner.writer.write_study = counted                           # type: ignore[assignment]

    base = rss_mb()
    try:
        runner.run(EvaluationJob(f"leak-{uuid.uuid4()}", "leak-probe", ds),
                   send_callback=False)
    finally:
        runner.writer.write_study = orig_write                    # type: ignore[assignment]

    gc.collect()
    end = rss_mb()

    prev_n, prev_rss = 0, base
    for cnt, r in samples:
        print(f"{cnt:>8}  {r:>10.0f}  {r - prev_rss:>+10.0f}  "
              f"{(r - base) / max(1, cnt):>10.2f}")
        prev_n, prev_rss = cnt, r

    print("=" * 78)
    growth = end - base
    print(f"起始 {base:.0f} MB → 结束 {end:.0f} MB   净增长 {growth:+.1f} MB / {a.n} 例")
    if samples:
        half = max(1, len(samples) // 2)
        (n1, r1), (n2, r2) = samples[0], samples[half - 1]
        slope1 = (r2 - r1) / max(1, n2 - n1) if n2 > n1 else 0.0
        (n3, r3), (n4, r4) = samples[half], samples[-1]
        slope2 = (r4 - r3) / max(1, n4 - n3) if n4 > n3 else 0.0
        print(f"前半段斜率 {slope1:+.3f} MB/例 | 后半段斜率 {slope2:+.3f} MB/例")
        if slope2 > 0.05 and slope2 >= slope1 * 0.6:
            per = slope2
            print(f"\n**疑似泄漏**：后半段仍在以 {per:+.3f} MB/例 增长 → "
                  f"777 例约 {per * 777 / 1000:+.1f} GB")
            print("定位：这正是「单测发现不了」的那类，逐 Study 检查 diagnostics / "
                  "日志对象 / 结果列表是否被累积持有")
        else:
            print("\n**无明显逐例泄漏**：后半段斜率已趋平")

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
