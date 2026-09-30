#!/usr/bin/env python3
"""Goal5 空掩膜探针：**不重启服务**，直接对前几个检查跑一次 goal5 并打印诊断。

回答一个问题：``goal5{core=0, flair=0}`` 到底是**哪一种**成因？

* ``missing`` 非空                → 该通道是**零占位**的（数据缺这个模态）
* ``pmax`` 明显低于 ``thr``        → 模型输出本身就低（通道 / 权重 / 预处理不匹配）
* ``pmax`` 够高，``pre>0`` 而终值 0 → **后处理**吃掉的（连通域 / 最小体素 / 桥接）

用法::

    python3 scripts/probe_goal5.py                       # 默认取 3 例
    python3 scripts/probe_goal5.py /data/verification 5

**会占用 GPU**（和你正在跑的推理共卡），但只跑几例、几十秒。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import Settings                                    # noqa: E402
from data.loader import DatasetLoader                               # noqa: E402
from tasks.goal5_segmentation.task import Goal5Task                 # noqa: E402

DEFAULT_ROOT = "/2026aicompetition/datasets/verification/original"


class _Ctx:
    """``Goal5Task.predict`` 需要的最小上下文（study / warnings / diagnostics）。"""

    def __init__(self, study) -> None:
        self.study = study
        self.warnings: list[str] = []
        self.diagnostics: dict = {}


def main(argv: list[str]) -> int:
    root = Path(argv[0]).expanduser() if argv else Path(DEFAULT_ROOT)
    limit = int(argv[1]) if len(argv) > 1 else 3

    print(f"数据根 : {root} | 存在={root.is_dir()}")
    print(f"取前 {limit} 例")
    if not root.is_dir():
        print("!! 数据根不存在"); return 2

    settings = Settings.from_env()
    print(f"权重根 : {settings.ckpt_root}")
    task = Goal5Task(settings=settings)
    task.load_model()
    print(f"权重   : {task._loaded.ckpt_path}")
    print(f"阈值   : {[round(float(t), 3) for t in task._loaded.thresholds]}")
    print(f"后处理 : min_tumor_voxels={task.cfg.min_tumor_voxels} "
          f"keep_components={task.cfg.keep_components} bridge_mm={task.cfg.bridge_mm}")
    print(f"滑窗   : patch={task.cfg.patch} overlap={task.cfg.overlap} "
          f"tta_flips={task.cfg.tta_flips}")
    print("=" * 92)

    empty = 0
    seen = 0
    for study in DatasetLoader().iter_studies(root):
        ctx = _Ctx(study)
        try:
            task.predict(ctx)
        except Exception as exc:                                    # noqa: BLE001
            print(f"[{study.accession_number}] ✗ {type(exc).__name__}: {exc}")
            continue
        d = ctx.diagnostics.get("goal5") or {}
        seen += 1
        is_empty = not d.get("core_voxels") and not d.get("flair_voxels")
        empty += is_empty
        print(f"\n[{study.accession_number}]  序列数={len(study.series)}")
        for s in study.series:
            print(f"    uid={s.series_uid[:28]:<30} desc={s.modality!r}")
        print(f"    missing = {d.get('missing_channels')}")
        print(f"    thr     = {d.get('thresholds')}   pmax = {d.get('max_probs')}")
        print(f"    core    = {d.get('core_voxels')} (阈值以上 {d.get('core_pre_voxels')})")
        print(f"    flair   = {d.get('flair_voxels')} (阈值以上 {d.get('flair_pre_voxels')})")
        if is_empty:
            probs = d.get("max_probs") or [0.0, 0.0]
            thrs = d.get("thresholds") or [0.5, 0.5]
            if probs[0] < thrs[0] and probs[1] < thrs[1]:
                print("    → 成因：**概率低于阈值**（模型输出不够高，不是后处理）")
            else:
                print("    → 成因：**后处理吃掉了**（阈值以上有体素但终值为 0）")
        for w in ctx.warnings:
            print(f"    warn: {w}")
        if seen >= limit:
            break

    print("\n" + "=" * 92)
    print(f"共 {seen} 例，其中空掩膜 {empty} 例")
    if seen == limit:
        print("（只看了前几例；要更可靠就把第二个参数调大）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
