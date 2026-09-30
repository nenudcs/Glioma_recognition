#!/usr/bin/env python3
"""Goal5 权重真伪 + 预处理一致性审计（本机可跑，只读权重与配置，**不读影像**）。

回答两个问题：

**A. 权重是真的训练产物吗？** —— 重点是 **``seg`` 头有没有真的加载上**。
``tasks/goal5_segmentation/inference.py::load_model`` 用 ``strict=False`` 加载，
但它的缺失键过滤只检查 ``("enc", "dec", "bottleneck", "stem")`` 前缀 ——
**``seg`` 头若名字对不上会被静默随机初始化**，权重校验通过、服务正常启动，
而分割输出永远是噪声 → 低于阈值 → **空掩膜**。

**B. 预处理与训练一致吗？** —— 逐项对比 ``Goal5Config`` 与训练侧
``glioma_track4/configs/preprocess.yaml``。

用法::

    python3 scripts/audit_goal5.py
    python3 scripts/audit_goal5.py --train-config /path/to/glioma_track4/configs/preprocess.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: 训练侧配置位置：**优先本仓内置副本**（``vendor/glioma_track4/configs/``），
#: 找不到才回退到平台上的独立算法工程路径。内置副本随仓库一起走，
#: 所以本脚本在干净 clone 下也能直接跑。
_VENDOR_TRAIN_CFG = (Path(__file__).resolve().parents[1]
                     / "vendor" / "glioma_track4" / "configs" / "preprocess.yaml")
TRAIN_CFG_DEFAULT = (str(_VENDOR_TRAIN_CFG) if _VENDOR_TRAIN_CFG.is_file()
                     else "/2026aicompetition/workspace/dcs/glioma_track4/configs/preprocess.yaml")

#: 训练侧 ``preprocess.yaml`` 里要核对的项（点号路径 → 中文含义）
_PARAM_KEYS = (
    "geometry.common_spacing",
    "geometry.max_spacing_factor",
    "geometry.crop_brain",
    "geometry.brain_margin_vox",
    "geometry.resample_order_img",
    "intensity.clip_percentile",
    "intensity.foreground_only",
    "inference.patch",
    "inference.overlap",
    "inference.tta_flips",
    "inference.seg_tta_flips",
    "inference.tta_batch",
    "inference.global_size",
    "inference.min_tumor_voxels",
    "inference.keep_components",
    "inference.bridge_mm",
)


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except Exception:                                             # noqa: BLE001
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:                                      # noqa: BLE001
        print(f"  !! YAML 解析失败（{type(exc).__name__}: {exc}）")
        return {}


def _dig(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _norm(v):
    """把值归一成可比较的形式（list→tuple、去浮点尾巴）。"""
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    if isinstance(v, float):
        return round(v, 6)
    return v


def part_a_weights() -> None:
    print("=" * 84)
    print("A. 权重真伪：是否真是训练产物、``seg`` 头是否真的加载上")
    print("=" * 84)
    try:
        import torch
    except Exception as exc:                                      # noqa: BLE001
        print(f"  !! 没有 torch（{exc}）→ 跳过")
        return

    from core.config import Settings
    from tasks.goal5.config import Goal5Config
    from tasks.goal5.inference import resolve_ckpts
    from tasks.goal5.models.factory import build_goal5_models

    settings = Settings.from_env()
    cfg = Goal5Config()
    print(f"checkpoint 根: {settings.ckpt_root}")
    print(f"规范相对路径 : {cfg.core_ckpt_rel}")
    try:
        paths = resolve_ckpts(settings.ckpt_root, cfg.core_ckpt_rel)
    except Exception as exc:                                      # noqa: BLE001
        print(f"  !! 解析权重失败: {type(exc).__name__}: {exc}")
        return
    print(f"解析到 {len(paths)} 个权重文件")

    for p in paths:
        print("-" * 84)
        print(f"文件: {p}")
        print(f"  大小: {p.stat().st_size / 1e6:.1f} MB")
        try:
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  !! torch.load 失败: {type(exc).__name__}: {exc}")
            continue
        if not isinstance(ck, dict):
            print(f"  !! 顶层不是 dict，而是 {type(ck).__name__}")
            continue

        # ---- 训练元数据（有没有这些字段本身就说明是不是训练脚本产出的）----
        meta_keys = ("arch", "model_cfg", "cls_spec", "thresholds", "global_size",
                     "global_size_mm", "epoch", "best", "best_metric", "fold",
                     "loss", "val_dice", "config", "args", "git", "version")
        present = [k for k in meta_keys if k in ck]
        print(f"  元数据键: {present}")
        for k in ("arch", "thresholds", "global_size", "global_size_mm",
                  "epoch", "best", "best_metric", "fold"):
            if k in ck:
                print(f"    {k} = {ck[k]}")
        if "model_cfg" in ck:
            print(f"    model_cfg = {ck['model_cfg']}")

        state = ck.get("model_ema") or ck.get("model") or ck
        if not isinstance(state, dict):
            print("  !! 取不到 state_dict")
            continue
        tensors = {k: v for k, v in state.items() if hasattr(v, "shape")}
        print(f"  state_dict: {len(tensors)} 个张量"
              f"（其中 seg 相关 {sum('seg' in k for k in tensors)} 个）")
        if not tensors:
            print("  !! state_dict 里没有张量 → 这不是一份模型权重")
            continue

        # ---- 关键：真正做一次 load_state_dict(strict=False)，看缺了什么 ----
        try:
            holder = build_goal5_models(cfg, ck.get("cls_spec") or [], shared=True)
            missing, unexpected = holder.core_model.load_state_dict(state, strict=False)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  !! load_state_dict 抛错: {type(exc).__name__}: {exc}")
            continue

        print(f"  缺失键 {len(missing)} 个 | 多余键 {len(unexpected)} 个")
        buckets: dict[str, int] = {}
        for k in missing:
            buckets.setdefault(k.split(".")[0], 0)
            buckets[k.split(".")[0]] += 1
        print(f"  缺失键按顶层模块分组: {buckets or '（无缺失）'}")
        if unexpected[:6]:
            print(f"  多余键样例: {unexpected[:6]}")

        print()
        seg_missing = [k for k in missing if "seg" in k.lower()]
        if seg_missing:
            print("  " + "!" * 76)
            print(f"  !!! **seg 头有 {len(seg_missing)} 个键没加载上 → 很可能被静默随机初始化**")
            print(f"  !!! 样例: {seg_missing[:5]}")
            print("  !!! 后果：分割输出是噪声 → 概率低于阈值 → **空掩膜**（正是当前现象）")
            print("  !!! 而 inference.load_model 只校验 enc/dec/bottleneck/stem 前缀，**查不出这个**")
            print("  " + "!" * 76)
        else:
            print("  seg 头：已全部加载 ✅")

        # ---- 权重数值：随机初始化与训练产物可区分 ----
        for key in sorted(tensors):
            if "seg" not in key.lower() or not key.endswith("weight"):
                continue
            w = tensors[key]
            if not hasattr(w, "float") or w.dim() == 0:
                continue
            f = w.detach().float()
            print(f"  {key}: shape={tuple(f.shape)} "
                  f"mean={f.mean():+.4f} std={f.std():.4f} "
                  f"min={f.min():+.4f} max={f.max():+.4f}")
        print("  ↑ 判据：训练后的权重 std 通常 0.01~0.3 且均值偏离 0；")
        print("     若 std ≈ 1/sqrt(fan_in)（如 96 通道 → ≈0.10）且均值≈0，可能是随机初始化。")
        del ck


def part_b_preprocess(train_cfg_path: Path) -> None:
    print()
    print("=" * 84)
    print("B. 预处理一致性：Goal5Config  vs  训练侧 preprocess.yaml")
    print("=" * 84)
    from tasks.goal5.config import Goal5Config
    from tasks.goal5.preprocess import CHANNEL_FALLBACK, CHANNEL_ORDER

    cfg = Goal5Config()
    print(f"训练配置: {train_cfg_path} | 存在={train_cfg_path.is_file()}")
    train = _load_yaml(train_cfg_path) if train_cfg_path.is_file() else {}

    # 提交侧的实际取值（点号路径 → 值）
    sub = {
        "geometry.common_spacing": cfg.common_spacing,
        "geometry.max_spacing_factor": cfg.max_spacing_factor,
        "geometry.crop_brain": True,
        "geometry.brain_margin_vox": 4,
        "geometry.resample_order_img": 1,
        "intensity.clip_percentile": (0.5, 99.5),
        "intensity.foreground_only": True,
        "inference.patch": cfg.patch,
        "inference.overlap": cfg.overlap,
        "inference.tta_flips": cfg.tta_flips,
        "inference.seg_tta_flips": cfg.tta_flips,
        "inference.tta_batch": cfg.tta_batch,
        "inference.global_size": cfg.global_size,
        "inference.min_tumor_voxels": cfg.min_tumor_voxels,
        "inference.keep_components": cfg.keep_components,
        "inference.bridge_mm": cfg.bridge_mm,
    }

    bad = []
    if not train:
        # ⚠️ 读不到训练配置时**绝不能**打印"一致" —— 那会把"没检查"说成"检查通过"，
        # 于是一个真实的分布不一致会被这张表盖过去。这里明说"无法判定"。
        print("  !! 读不到训练配置 → **无法判定**（下面只是提交侧取值，不代表一致）")
        for k in _PARAM_KEYS:
            print(f"    {k:<34} 提交侧 = {_norm(sub.get(k))}")
        print(f"\n  ★ 结论：**无法判定**（请用 --train-config 指向训练侧 preprocess.yaml）")
        return None

    print(f"\n  {'参数':<36}{'训练侧':>18}{'提交侧':>18}   判定")
    for k in _PARAM_KEYS:
        t, s = _norm(_dig(train, k)), _norm(sub.get(k))
        if t is None:
            mark = "训练侧未定义"
        elif s is None:
            mark = "!! 提交侧缺失"
            bad.append(k)
        elif t == s:
            mark = "OK"
        else:
            mark = "!! **不一致**"
            bad.append(k)
        print(f"  {k:<36}{str(t):>18}{str(s):>18}   {mark}")

    # 通道 fallback（结构对比，不是标量）
    print(f"\n  通道取用链 CHANNEL_FALLBACK（提交侧）:")
    for name in CHANNEL_ORDER:
        print(f"    {name:<6} <- {list(CHANNEL_FALLBACK[name])}")
    if train:
        chans = train.get("channels") or []
        print("  通道取用链（训练侧 preprocess.yaml）:")
        for ch in chans:
            if isinstance(ch, dict):
                print(f"    {str(ch.get('name')):<6} <- {[ch.get('name')] + list(ch.get('fallback') or [])}")
        want = {str(c.get("name")): tuple([c.get("name")] + list(c.get("fallback") or []))
                for c in chans if isinstance(c, dict)}
        for name in CHANNEL_ORDER:
            if name in want and want[name] != tuple(CHANNEL_FALLBACK[name]):
                bad.append(f"channels.{name}")
                print(f"    !! {name} 的取用链与训练侧不一致")

    print()
    if bad:
        print(f"  ★ 发现 {len(bad)} 处不一致: {bad}")
        print("  ★ 这些会让推理输入超出训练分布 → 分割塌陷（空掩膜）")
    else:
        print("  ★ 预处理与训练侧一致 ✅")
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Goal5 权重真伪 + 预处理一致性审计")
    ap.add_argument("--train-config", default=TRAIN_CFG_DEFAULT,
                    help=f"训练侧 preprocess.yaml（默认 {TRAIN_CFG_DEFAULT}）")
    ap.add_argument("--skip-weights", action="store_true")
    a = ap.parse_args(argv)

    if not a.skip_weights:
        part_a_weights()
    part_b_preprocess(Path(a.train_config).expanduser())
    print()
    print("=" * 84)
    print("下一步：若 A 报「seg 头没加载上」→ 权重导出问题；")
    print("        若 B 报不一致 → 预处理问题（本仓已按训练侧对齐，请确认已同步新代码）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
