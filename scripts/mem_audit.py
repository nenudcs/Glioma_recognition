#!/usr/bin/env python3
"""内存审计：定位"一旦评测/测试就 OOM"的放大环节。

逐个量化四件事，最后给出峰值内存推算：

A. ``resolve_ckpts`` 的**真实**解析行为 —— 目录里多份 ``*.pt`` 时会发生什么；
B. 单个 Goal 权重加载后的常驻内存（参数 + 缓冲区）；
C. **放大倍数** = 启用 Goal 数 × 每 Goal 集成成员数；
D. 训练栈（``train`` / ``evaluate`` / ``_common.training``）的导入成本 ——
   契约测试会 import 它们，容器内存小的时候这一项本身就够 OOM。

用法::

    python3 scripts/mem_audit.py
    python3 scripts/mem_audit.py --ckpt-root /path/to/checkpoint
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:                                                 # noqa: BLE001
    _PROC = None


def rss_mb() -> float:
    if _PROC is None:
        return float("nan")
    return _PROC.memory_info().rss / 1e6


def rss_note() -> str:
    """含平台上限的提示（超上限会被 OOM-kill → 容器重启）。"""
    if _PROC is None:
        return "（无 psutil，无法测量）"
    v = psutil.virtual_memory()
    cg = Path("/sys/fs/cgroup/memory.max")
    limit = ""
    if cg.is_file():                                              # cgroup v2
        try:
            txt = cg.read_text().strip()
            if txt.isdigit():
                limit = f" | cgroup 上限 {int(txt) / 1e9:.1f} GB"
        except Exception:                                         # noqa: BLE001
            pass
    return f"{rss_mb():.0f} MB | 系统可用 {v.available / 1e9:.1f} GB{limit}"


def _cuda_mb():
    """``(已分配, 峰值, 预留)`` MB；无 CUDA 时 ``None``。"""
    try:
        import torch
        if torch.cuda.is_available():
            return (torch.cuda.memory_allocated() / 1e6,
                    torch.cuda.max_memory_allocated() / 1e6,
                    torch.cuda.memory_reserved() / 1e6)
    except Exception:                                             # noqa: BLE001
        pass
    return None


def stage(name: str) -> None:
    """打印某一阶段的 RSS / 显存（OOM 排障的核心：定位峰值出在哪一步）。"""
    line = f"  {name:<36} RSS {rss_mb():8.0f} MB"
    c = _cuda_mb()
    if c:
        line += (f" | CUDA 已分配 {c[0]:7.0f}  峰值 {c[1]:7.0f}  "
                 f"预留 {c[2]:7.0f} MB")
    print(line, flush=True)


def trace_one(dataset: Path, ckpt_root: Path | None, device: str) -> None:
    """单例端到端：逐阶段看内存涨在哪里。"""
    import gc

    from data.loader import DatasetLoader
    from tasks.goal5.config import Goal5Config
    from tasks.goal5.inference import infer_segmentation, load_model
    from tasks.goal5.postprocess import clean_pair
    from tasks.goal5.preprocess import build_volume
    from tasks._common.spatial import spacing_of

    stage("0 起始（仅解释器 + numpy/torch）")
    it = DatasetLoader().iter_studies(dataset)
    study = next(it)
    stage(f"1 加载 Study {study.accession_number}（{len(study.series)} 条序列）")

    cfg = Goal5Config()
    pv = build_volume(study, cfg)
    stage(f"2 build_volume  [{tuple(pv.volume.shape)}]  "
          f"missing={list(pv.missing)}")

    loaded = load_model(cfg, ckpt_root, device=device)
    stage(f"3 load_model（权重={Path(loaded.ckpt_path).name}）")

    prob = infer_segmentation(loaded, pv.volume, cfg)
    stage(f"4 infer_segmentation  概率图 {tuple(prob.shape)}")

    core_p, flair_p = prob[cfg.core_channel], prob[cfg.flair_channel]
    core, flair = clean_pair(core_p, flair_p, cfg, spacing=spacing_of(pv.affine))
    stage(f"5 postprocess  core={int(core.sum())} flair={int(flair.sum())}")

    print()
    print("  判读：")
    print("    · 第 1 步就很高        → 单例影像过大（原始 NIfTI 分辨率/层数异常）")
    print("    · 第 2 步跳升最多      → 1mm 公共网格重采样。D×H×W×4 通道×4B，")
    print("                            如有 4 个不同 ckpt_rel 的 StudyTask 会各建一次")
    print("    · 第 3 步跳升 × N      → **常驻模型数 = 启用 Goal 数 × 集成份数**（见 A 段）")
    print("    · 第 4 步跳升最多      → 滑窗累加器 + TTA。patch/tta_batch 越大越吃显存")
    del prob, pv, core_p, flair_p, core, flair, loaded, study
    gc.collect()
    stage("6 释放后")


def head(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="内存审计")
    ap.add_argument("--ckpt-root", default=None,
                    help="checkpoint 根（默认取 Settings.from_env().ckpt_root）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trace", action="store_true",
                    help="额外跑单例端到端内存轨迹（会占用 GPU，几十秒）")
    ap.add_argument("--dataset", default="/2026aicompetition/datasets/verification/original",
                    help="--trace 用的数据根")
    a = ap.parse_args(argv)

    print(f"起始内存: {rss_note()}")

    # ------------------------------------------------------------------ #
    head("A. resolve_ckpts 的真实行为：目录里多份 *.pt 会怎样")
    # ------------------------------------------------------------------ #
    import tempfile

    from tasks.goal5.config import Goal5Config
    from tasks.goal5.inference import resolve_ckpt, resolve_ckpts

    cfg = Goal5Config()
    print(f"规范约定的相对路径 core_ckpt_rel = {cfg.core_ckpt_rel!r}")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "goal5_segmentation"
        d.mkdir(parents=True)
        for i in range(5):
            (d / f"fold{i}.pt").write_bytes(b"\x00")
        missing = resolve_ckpt(Path(tmp), cfg.core_ckpt_rel)
        got = resolve_ckpts(Path(tmp), cfg.core_ckpt_rel)
        print(f"目录里: 5 个 fold*.pt，**没有** core.pt")
        print(f"resolve_ckpt()  -> {missing.name}（规范期望的文件）")
        print(f"resolve_ckpts() -> {len(got)} 个: {[p.name for p in got]}")
        print()
        print("  ==> 找不到 core.pt 时会把该目录下**全部** *.pt 当成集成成员。")
        print("      所以目录里只要多放了 epoch 快照 / 备份 / best.pth，")
        print("      **每一个都会被加载成一个常驻模型**。")
        (d / "core.pt").write_bytes(b"\x00")
        got2 = resolve_ckpts(Path(tmp), cfg.core_ckpt_rel)
        print(f"  放上 core.pt 后 -> {len(got2)} 个: {[p.name for p in got2]}（只取指定文件 ✓）")

    # ------------------------------------------------------------------ #
    head("B. 单份权重加载后的常驻内存")
    # ------------------------------------------------------------------ #
    ckpt_root = Path(a.ckpt_root) if a.ckpt_root else None
    if ckpt_root is None:
        try:
            from core.config import Settings
            ckpt_root = Settings.from_env().ckpt_root
        except Exception as exc:                                  # noqa: BLE001
            print(f"  !! 取不到 ckpt_root: {exc}")
    print(f"ckpt_root = {ckpt_root}")
    if ckpt_root is None or not Path(ckpt_root).is_dir():
        print("  !! 目录不存在 → 跳过 B/C（在容器里跑才有意义）")
    else:
        try:
            from tasks._common.factory import build_shared_backbone
            import torch
            try:
                paths = resolve_ckpts(Path(ckpt_root), cfg.core_ckpt_rel)
            except Exception as exc:                              # noqa: BLE001
                paths = []
                print(f"  !! resolve_ckpts 失败: {exc}")
            print(f"  该 Goal 解析到 {len(paths)} 份权重")
            for p in paths:
                before = rss_mb()
                ck = torch.load(str(p), map_location="cpu", weights_only=False)
                m = build_shared_backbone(ck, None, cfg.in_channels)
                state = ck.get("model_ema") or ck.get("model") or ck
                m.load_state_dict(state, strict=False)
                n_par = sum(x.numel() for x in m.parameters())
                raw_mb = n_par * 4 / 1e6
                after = rss_mb()
                print(f"  {p.name:<22} 参数 {n_par / 1e6:6.2f} M  "
                      f"fp32 权重≈{raw_mb:6.0f} MB  "
                      f"实测进程增量 {after - before:7.0f} MB")
                del m, ck, state
                gc.collect()
        except Exception as exc:                                  # noqa: BLE001
            print(f"  !! B 段失败: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------ #
    head("C. 放大倍数：启用 Goal 数 × 每 Goal 集成成员数")
    # ------------------------------------------------------------------ #
    try:
        from tasks.real_pipeline import DEFAULT_GOALS
        goals = [g for g in DEFAULT_GOALS.split(",") if g.strip()]
    except Exception:                                             # noqa: BLE001
        goals = ["goal1", "goal2_stitched", "goal3", "goal5", "goal4", "goal2_duplicate"]
    study_goals = [g for g in goals if g != "goal2_duplicate"]
    print(f"GLIOMA_GOALS 默认 = {goals}")
    print(f"其中 **逐 Study 的 StudyTask** = {study_goals}")
    print()
    print("  每个 StudyTask 各持有一个 BackboneRunner，各自独立 load()：")
    print(f"    StudyTask 数 × 每 Goal 的 *.pt 份数 = {len(study_goals)} × N 个模型常驻")
    print(f"    例：N=1 → {len(study_goals)} 个模型；N=5 → {len(study_goals) * 5} 个模型")
    print()
    print("  ⚠️ Goal5 **不走**共享骨干（它用滑窗），是额外的一套：")
    print("     goal5 的 predict_volume 还要为整个体积分配 2 通道 fp32 累加器 +")
    print("     每个 TTA 组合的 patch 激活 —— 大体积下是 GB 级。")

    # ------------------------------------------------------------------ #
    head("D. 训练栈的导入成本（契约测试会 import 它们）")
    # ------------------------------------------------------------------ #
    targets = []
    try:
        from tasks.real_pipeline import _BUILDERS
        mods = sorted({m.rsplit(".", 1)[0] for m, _ in _BUILDERS.values()})
        for m in mods:
            targets.append(f"{m}.train")
            targets.append(f"{m}.evaluate")
    except Exception:                                             # noqa: BLE001
        pass
    targets += ["tasks._common.training.cli", "tasks._common.training.engine",
                "tasks._common.training.helpers", "monai", "torch"]
    base = rss_mb()
    for mod in targets:
        before = rss_mb()
        try:
            __import__(mod)
            d = rss_mb() - before
            print(f"  {mod:<44} +{d:8.0f} MB   （累计 {rss_mb():8.0f} MB）")
        except ModuleNotFoundError as exc:
            print(f"  {mod:<44} 未安装（{exc.name}）")
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {mod:<44} 导入失败: {type(exc).__name__}: {str(exc)[:60]}")
    print(f"\n  导入阶段总增量: {rss_mb() - base:+.0f} MB | 当前 {rss_note()}")

    # ------------------------------------------------------------------ #
    head("E. 单例端到端内存轨迹（--trace 才跑；会占 GPU）")
    # ------------------------------------------------------------------ #
    if not a.trace:
        print("  （未开启。在容器里加 --trace 跑一次，就能看到峰值出现在哪一步）")
        print("    python3 scripts/mem_audit.py --trace --dataset <数据根>")
    else:
        ds = Path(a.dataset).expanduser()
        if not ds.is_dir():
            print(f"  !! 数据根不存在: {ds}")
        else:
            try:
                trace_one(ds, ckpt_root, a.device)
            except Exception as exc:                              # noqa: BLE001
                print(f"  !! 轨迹失败: {type(exc).__name__}: {exc}")
                import traceback
                traceback.print_exc()

    head("结论")
    print(f"最终内存: {rss_note()}")
    print()
    print("按下面顺序排查（都是「常驻」而非「泄漏」，所以表现为一启动就 OOM）：")
    print("  1. ls -1 <ckpt_root>/<goal>/  —— 目录里是不是有多份 *.pt（快照/备份/多折）？")
    print("     每一份都会被加载成常驻模型；放一份 core.pt 就能让 resolve_ckpts 只取它。")
    print("  2. GLIOMA_GOALS 是否 6 个全开？不需要的 Goal 用 Dummy 补位可省下整套模型。")
    print("  3. 容器内存上限（cgroup）是否小于上述峰值？用 --device cpu 先跑通再上 GPU。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
