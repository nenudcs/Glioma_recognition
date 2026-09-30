#!/usr/bin/env python3
"""提交前**一键预检**：跑完所有关卡，最后给一个 ``GO`` / ``NO-GO``。

设计目标：**一条命令、无需参数、退出码即结论**。
任何一关抛异常都不会让脚本崩 —— 只会把那一关记成 FAIL 并继续跑下一关，
这样你能一次看到**全部**问题，而不是修一个跑一次。

关卡::

  G1 环境     torch / CUDA / 权重文件 / 数据目录 / 官方序列表
  G2 权重     ``load_state_dict`` 后 **``seg`` 头是否真的加载上**（最容易静默出错的一关）
  G3 预处理   Goal5Config 与训练侧 preprocess.yaml 的 16 项参数 + 通道取用链
  G4 冒烟     直接对前 N 例推理，统计**空掩膜率**与 ``pmax``
  G5 端到端   完整 runner（Writer+Validator）跑一批，再离线校验输出格式（``--full`` 才跑）

用法::

  python3 scripts/pre_submit_check.py                    # 默认 5 例冒烟，~1 分钟
  python3 scripts/pre_submit_check.py --limit 20         # 多冒烟几例更可靠
  python3 scripts/pre_submit_check.py --full             # 完整跑（慢，提交前最后一遍）

退出码：``0`` = GO，``1`` = NO-GO。
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DEFAULT = "/2026aicompetition/datasets/verification/original"
#: 训练侧配置位置：**优先本仓内置副本**（``vendor/glioma_track4/configs/``），
#: 找不到才回退到平台上的独立算法工程路径。内置副本随仓库一起走，
#: 所以本脚本在干净 clone 下也能直接跑。
_VENDOR_TRAIN_CFG = ROOT / "vendor" / "glioma_track4" / "configs" / "preprocess.yaml"
TRAIN_CFG_DEFAULT = (str(_VENDOR_TRAIN_CFG) if _VENDOR_TRAIN_CFG.is_file()
                     else "/2026aicompetition/workspace/dcs/glioma_track4/configs/preprocess.yaml")

#: 关卡结果：(名称, 状态, 摘要)。状态 ∈ {PASS, FAIL, WARN, SKIP}
_GATES: list[tuple[str, str, str]] = []

#: 阻止 GO 的关卡名。``WARN`` 与 ``SKIP`` 不阻止。
_BLOCKING: set[str] = set()


def _record(name: str, status: str, summary: str = "") -> None:
    _GATES.append((name, status, summary))
    if status == "FAIL":
        _BLOCKING.add(name)
    icon = {"PASS": "✔", "FAIL": "✘", "WARN": "!", "SKIP": "-"}.get(status, "?")
    print(f"[{icon}] {name:<18} {summary}", flush=True)


def _guard(name: str):
    """把一关包成 try/except：内部抛错 → 记 FAIL，不中断其余关卡。"""
    def deco(fn):
        def wrapper(*a, **kw):
            print()
            print("-" * 88)
            print(f"▶ {name}")
            print("-" * 88)
            try:
                return fn(*a, **kw)
            except Exception as exc:                              # noqa: BLE001
                _record(name, "FAIL", f"关卡自身抛错 {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return None
        return wrapper
    return deco


# --------------------------------------------------------------------------- #
# G1 环境
# --------------------------------------------------------------------------- #
@_guard("G1 环境")
def gate_env(dataset: Path, limit: int) -> dict:
    info: dict = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        info["device"] = torch.cuda.get_device_name(0) if info["cuda"] else "cpu"
    except Exception as exc:                                      # noqa: BLE001
        _record("G1 环境", "FAIL", f"import torch 失败: {exc}")
        return info

    from core.config import Settings
    from tasks.goal5_segmentation.config import Goal5Config
    from tasks.goal5_segmentation.inference import resolve_ckpts
    from tasks.goal5_segmentation.preprocess import CHANNEL_FALLBACK, CHANNEL_ORDER

    settings = Settings.from_env()
    cfg = Goal5Config()
    info["settings"] = settings
    info["cfg"] = cfg

    print(f"torch      : {info['torch']}")
    print(f"CUDA       : {info['cuda']}  {info['device']}")
    print(f"权重根     : {settings.ckpt_root}")
    print(f"数据根     : {dataset}  存在={dataset.is_dir()}")
    print(f"取例上限   : {limit}")

    if not info["cuda"]:
        _record("G1 环境", "WARN", "CUDA 不可用 → 会用 CPU 推理（极慢，正式评测必须用 GPU）")
    if not dataset.is_dir():
        _record("G1 环境", "FAIL", f"数据目录不存在: {dataset}")
        return info

    try:
        paths = resolve_ckpts(settings.ckpt_root, cfg.core_ckpt_rel)
        info["ckpt_paths"] = paths
        size_mb = sum(p.stat().st_size for p in paths) / 1e6
        print(f"权重文件   : {len(paths)} 个，共 {size_mb:.1f} MB")
        for p in paths:
            print(f"             {p}")
    except Exception as exc:                                      # noqa: BLE001
        _record("G1 环境", "FAIL", f"权重解析失败: {type(exc).__name__}: {exc}")
        return info

    print(f"通道取用链 : " + ", ".join(
        f"{n}<-{list(CHANNEL_FALLBACK[n])}" for n in CHANNEL_ORDER))

    # ---- 插件工厂：不设 = 服务**静默跑 Dummy 基线**（本仓最致命的一条）----
    # ``core/registry.build_pipeline(None)`` 只打印一条告警就返回空 Pipeline：
    # 服务照常起来、/health 通、回调正常，但答案是占位内容。评测不可重跑 → 直接 0 分。
    factory = os.environ.get("COMPETITION_PIPELINE_FACTORY") or settings.pipeline_factory
    if not factory:
        _record("G1 环境", "FAIL",
                "COMPETITION_PIPELINE_FACTORY 未设置 → 服务会**静默跑 Dummy 基线**"
                "（答案全是占位内容）。用 ./start.sh 启动（已默认设为 "
                "tasks.real_pipeline:build_pipeline），或手动 export")
        return info
    print(f"插件工厂   : {factory}")
    if "real_pipeline" not in factory and "glioma.pipeline" not in factory:
        print("             ⚠️ 不是已知的两个真实工厂，确认它真的注册了 Goal5")

    if "G1 环境" not in _BLOCKING:
        _record("G1 环境", "PASS",
                f"torch {info['torch']} / {'cuda' if info['cuda'] else 'cpu'} / "
                f"{len(paths)} 份权重 / 数据就位")
    return info


# --------------------------------------------------------------------------- #
# G2 权重：seg 头是否真的加载上
# --------------------------------------------------------------------------- #
@_guard("G2 权重")
def gate_weights(info: dict) -> None:
    import torch
    from tasks.goal5_segmentation.models.factory import build_goal5_models

    paths = info.get("ckpt_paths") or []
    cfg = info.get("cfg")
    if not paths or cfg is None:
        _record("G2 权重", "FAIL", "G1 未取到权重路径 → 跳过")
        return

    problems: list[str] = []
    for p in paths:
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        if not isinstance(ck, dict):
            problems.append(f"{p.name}: 顶层不是 dict（是 {type(ck).__name__}）")
            continue
        meta = [k for k in ("arch", "model_cfg", "thresholds", "epoch",
                            "best_metric", "fold") if k in ck]
        print(f"\n{p.name}  大小={p.stat().st_size / 1e6:.1f}MB")
        print(f"  元数据键 : {meta}")
        for k in ("arch", "thresholds", "global_size", "epoch", "best_metric", "fold"):
            if k in ck:
                print(f"    {k} = {ck[k]}")
        if "model_cfg" in ck:
            print(f"    model_cfg = {ck['model_cfg']}")

        state = ck.get("model_ema") or ck.get("model") or ck
        n_tensor = sum(1 for v in state.values() if hasattr(v, "shape"))
        print(f"  张量数   : {n_tensor}")
        if not n_tensor:
            problems.append(f"{p.name}: state_dict 里没有张量 → 不是模型权重")
            continue

        m = build_goal5_models(cfg, ck.get("cls_spec") or [], shared=True)
        missing, unexpected = m.core_model.load_state_dict(state, strict=False)
        buckets: dict[str, int] = {}
        for k in missing:
            buckets[k.split(".")[0]] = buckets.get(k.split(".")[0], 0) + 1
        print(f"  缺失键   : {len(missing)} 个 {buckets or '（无）'}")
        print(f"  多余键   : {len(unexpected)} 个")

        seg_missing = [k for k in missing if "seg" in k.lower()]
        backbone_missing = [k for k in missing
                            if k.startswith(("enc", "dec", "bottleneck", "stem"))]
        if seg_missing:
            print("  !!! **seg 头没加载上 → 分割头是随机初始化的**")
            print(f"  !!! 样例: {seg_missing[:5]}")
            problems.append(f"{p.name}: seg 头缺失 {len(seg_missing)} 个键（随机初始化）")
        if backbone_missing:
            problems.append(f"{p.name}: 骨干缺失 {len(backbone_missing)} 个键")

        for key in sorted(state):
            if "seg" in key.lower() and key.endswith("weight") and hasattr(state[key], "float"):
                f = state[key].detach().float()
                if f.dim() >= 1:
                    print(f"  {key}: mean={f.mean():+.4f} std={f.std():.4f}")
        del ck

    if problems:
        _record("G2 权重", "FAIL", "；".join(problems))
    else:
        _record("G2 权重", "PASS", "seg 头与骨干全部加载成功，权重是训练产物")


# --------------------------------------------------------------------------- #
# G3 预处理一致性
# --------------------------------------------------------------------------- #
@_guard("G3 预处理")
def gate_preprocess(train_cfg: Path, info: dict) -> None:
    from tasks.goal5_segmentation.config import Goal5Config
    from tasks.goal5_segmentation.preprocess import CHANNEL_FALLBACK, CHANNEL_ORDER

    cfg = info.get("cfg") or Goal5Config()
    if not train_cfg.is_file():
        _record("G3 预处理", "WARN",
                f"训练配置不存在({train_cfg}) → **无法判定**，只能打印提交侧取值")
        print(f"提交侧: max_spacing_factor={cfg.max_spacing_factor} overlap={cfg.overlap} "
              f"patch={cfg.patch} min_tumor_voxels={cfg.min_tumor_voxels} "
              f"keep_components={cfg.keep_components} bridge_mm={cfg.bridge_mm}")
        return

    try:
        import yaml
        train = yaml.safe_load(train_cfg.read_text(encoding="utf-8")) or {}
    except Exception as exc:                                      # noqa: BLE001
        _record("G3 预处理", "WARN", f"YAML 解析失败({exc}) → 无法判定")
        return

    def dig(d, dotted):
        node = d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def norm(v):
        if isinstance(v, (list, tuple)):
            return tuple(norm(x) for x in v)
        if isinstance(v, float):
            return round(v, 6)
        return v

    sub = {
        "geometry.common_spacing": cfg.common_spacing,
        "geometry.max_spacing_factor": cfg.max_spacing_factor,
        "geometry.resample_order_img": 1,
        "geometry.brain_margin_vox": 4,
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
    bad: list[str] = []
    print(f"{'参数':<36}{'训练侧':>18}{'提交侧':>18}   判定")
    for k, s in sub.items():
        t = norm(dig(train, k))
        s = norm(s)
        mark = "OK" if t == s else ("训练侧未定义" if t is None else "!! 不一致")
        if mark == "!! 不一致":
            bad.append(k)
        print(f"{k:<36}{str(t):>18}{str(s):>18}   {mark}")

    chans = train.get("channels") or []
    want = {str(c.get("name")): tuple([c.get("name")] + list(c.get("fallback") or []))
            for c in chans if isinstance(c, dict)}
    for name in CHANNEL_ORDER:
        got = tuple(CHANNEL_FALLBACK[name])
        exp = want.get(name)
        if exp is not None and exp != got:
            bad.append(f"channels.{name}")
            print(f"channels.{name:<26}{str(exp):>18}{str(got):>18}   !! 不一致")
        else:
            print(f"channels.{name:<26}{str(exp):>18}{str(got):>18}   OK")

    # ---- 另一份推理侧预处理：Goal1/2/3/4 走的共享骨干 ----
    # 推理侧有**两份**多通道预处理（goal5 一份、共享骨干一份），历史上正是"只修了一份"
    # 导致四个分类/嵌入头继续吃 OOD 输入。这里一并对齐，避免再次只修一半。
    print()
    print("  共享预处理 tasks/_common/volume.py（Goal1/2/3/4 走它）:")
    try:
        from tasks._common import volume as shared

        for name in shared.CHANNEL_ORDER:
            got = tuple(shared.CHANNEL_FALLBACK[name])
            exp = want.get(name)
            if exp is not None and exp != got:
                bad.append(f"shared.channels.{name}")
                print(f"  shared.channels.{name:<19}{str(exp):>18}{str(got):>18}   !! 不一致")
            else:
                print(f"  shared.channels.{name:<19}{str(exp):>18}{str(got):>18}   OK")

        if tuple(shared._REF_PRIORITY) != ("t1c", "flair", "t2", "t1"):
            bad.append("shared._REF_PRIORITY")
            print(f"  shared._REF_PRIORITY  = {shared._REF_PRIORITY}   !! 应为 "
                  f"('t1c', 'flair', 't2', 't1')（训练 dataset.build_case_volume）")
        else:
            print(f"  shared._REF_PRIORITY  = {shared._REF_PRIORITY}   OK")

        exp_mf = dig(train, "geometry.max_spacing_factor")
        if exp_mf is not None and float(exp_mf) != float(shared.DEFAULT_MAX_SPACING_FACTOR):
            bad.append("shared.DEFAULT_MAX_SPACING_FACTOR")
            print(f"  shared.max_spacing_factor = {shared.DEFAULT_MAX_SPACING_FACTOR}   "
                  f"!! 训练侧 {exp_mf}")
        else:
            print(f"  shared.max_spacing_factor = {shared.DEFAULT_MAX_SPACING_FACTOR}   OK")
    except Exception as exc:                                      # noqa: BLE001
        _record("G3 预处理", "FAIL",
                f"读共享预处理 tasks/_common/volume.py 失败: {type(exc).__name__}: {exc}")
        return

    if bad:
        _record("G3 预处理", "FAIL", f"{len(bad)} 处与训练不一致: {bad}")
    else:
        _record("G3 预处理", "PASS",
                f"{len(sub)} 项参数 + **两份**推理侧预处理（goal5 / 共享骨干）的"
                f"通道链·参考序·max_spacing_factor 全部与训练侧一致")


# --------------------------------------------------------------------------- #
# G4 冒烟：空掩膜率
# --------------------------------------------------------------------------- #
@_guard("G4 冒烟")
def gate_smoke(dataset: Path, limit: int, info: dict) -> None:
    from data.loader import DatasetLoader
    from tasks.goal5_segmentation.task import Goal5Task

    class _Ctx:
        def __init__(self, study):
            self.study = study
            self.warnings: list[str] = []
            self.diagnostics: dict = {}

    task = Goal5Task(settings=info["settings"])
    task.load_model()
    print(f"阈值 : {[round(float(t), 3) for t in task._loaded.thresholds]}")
    print(f"滑窗 : patch={task.cfg.patch} overlap={task.cfg.overlap} "
          f"bridge={task.cfg.bridge_mm}min={task.cfg.min_tumor_voxels}")
    print("=" * 88)

    seen = empty = 0
    pmax_core: list[float] = []
    pmax_flair: list[float] = []
    pp_killed = 0
    failures: list[str] = []
    for study in DatasetLoader().iter_studies(dataset):
        ctx = _Ctx(study)
        try:
            task.predict(ctx)
        except Exception as exc:                                  # noqa: BLE001
            failures.append(f"{study.accession_number}: {type(exc).__name__}: {exc}")
            print(f"[{study.accession_number}] ✗ {type(exc).__name__}: {exc}")
            continue
        d = ctx.diagnostics.get("goal5") or {}
        seen += 1
        mp = d.get("max_probs") or [0.0, 0.0]
        th = d.get("thresholds") or [0.5, 0.5]
        pmax_core.append(float(mp[0]))
        pmax_flair.append(float(mp[1]))
        is_empty = not d.get("core_voxels") and not d.get("flair_voxels")
        empty += is_empty
        pre = (d.get("core_pre_voxels") or 0) + (d.get("flair_pre_voxels") or 0)
        pp_killed += bool(is_empty and pre > 0)
        print(f"[{study.accession_number}] missing={d.get('missing_channels')} "
              f"pmax={mp} thr={th} core={d.get('core_voxels')}"
              f"(pre {d.get('core_pre_voxels')}) flair={d.get('flair_voxels')}"
              f"(pre {d.get('flair_pre_voxels')})"
              f"{'  ← 空' if is_empty else ''}")
        for w in ctx.warnings:
            print(f"    warn: {w}")
        if seen >= limit:
            break

    import statistics
    print("=" * 88)
    print(f"推理成功 : {seen} 例 | 异常 {len(failures)} 例")
    if seen:
        rate = empty / seen
        print(f"空掩膜   : {empty}/{seen} = {rate:.0%}")
        print(f"其中被后处理吃掉 : {pp_killed} 例")
        print(f"pmax core  : 均值 {statistics.fmean(pmax_core):.3f} "
              f"最大 {max(pmax_core):.3f}")
        print(f"pmax flair : 均值 {statistics.fmean(pmax_flair):.3f} "
              f"最大 {max(pmax_flair):.3f}")
    else:
        rate = 1.0
        print("!! 一例都没跑成功")

    if failures:
        _record("G4 冒烟", "FAIL",
                f"{len(failures)} 例抛异常（首条: {failures[0][:80]}）")
    elif seen == 0:
        _record("G4 冒烟", "FAIL", "0 例成功")
    elif rate > 0.6:
        _record("G4 冒烟", "FAIL",
                f"空掩膜率 {rate:.0%}（>60%）→ 提交上去几乎必然低分，先别交")
    elif rate > 0.2:
        _record("G4 冒烟", "WARN",
                f"空掩膜率 {rate:.0%}（20%~60%）→ 可提交但分数受损，建议先看诊断")
    else:
        _record("G4 冒烟", "PASS", f"空掩膜率 {rate:.0%}（{empty}/{seen}）")


# --------------------------------------------------------------------------- #
# G6 内存：常驻模型数（OOM 的直接成因）
# --------------------------------------------------------------------------- #
@_guard("G6 内存")
def gate_memory(info: dict) -> None:
    """算清楚**会有多少个模型常驻** —— 这是"评测一启动容器就 OOM 重启"的直接成因。

    常驻模型数 = ``不同 ckpt_rel 的个数 × 每个 ckpt_rel 解析出的份数``。

    - 份数 > 1 只有当该目录下**没有**规范约定的文件名时才会发生
      （``resolve_ckpts`` 退化成"目录下全部 ``*.pt`` 做集成"）；
    - 目录里混进训练快照（``epoch_*.pt`` / ``last.pt`` / 备份）份数就会翻上去，
      而这些快照还带 Adam 优化器状态，单份体积是推理权重的数倍。
    """
    import importlib
    import os

    from tasks.real_pipeline import DEFAULT_GOALS, _BUILDERS
    from tasks.goal5_segmentation.inference import resolve_ckpt, resolve_ckpts

    settings = info["settings"]
    root = Path(settings.ckpt_root)
    print(f"checkpoint 根 : {root}  存在={root.is_dir()}")
    if not root.is_dir():
        _record("G6 内存", "FAIL", f"checkpoint 根不存在: {root}")
        return

    goals = [g.strip() for g in
             (os.environ.get("GLIOMA_GOALS") or DEFAULT_GOALS).split(",") if g.strip()]
    total = 0
    distinct: dict[tuple, int] = {}
    rows: list[tuple[str, str, int]] = []
    for g in goals:
        mod, cls_name = _BUILDERS.get(g, (None, None))
        if mod is None:
            continue
        try:
            cls = getattr(__import__(f"{mod}", fromlist=[cls_name]), cls_name)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {g:<18} 导入失败: {type(exc).__name__}: {exc}")
            continue
        rel = getattr(cls, "ckpt_rel", None)
        if not rel:
            # 部分 Goal（如 Goal5）不把权重路径放在 Task 类上，而在自己的 Config 里
            # （``Goal5Config.core_ckpt_rel``）。不查这里会**漏报** Goal5 的常驻模型。
            try:
                # ``_BUILDERS`` 给的是 ``tasks.<goal>.task``（含 ``.task``），
                # 直接拼 ``.config`` 会变成 ``...task.config`` 而导入失败。
                pkg = mod[:-5] if mod.endswith(".task") else mod
                cfg_mod = importlib.import_module(f"{pkg}.config")
                for name in dir(cfg_mod):
                    obj = getattr(cfg_mod, name)
                    if isinstance(obj, type) and name.endswith("Config"):
                        rel = getattr(obj(), "core_ckpt_rel", None) \
                            or getattr(obj(), "ckpt_rel", None)
                        if rel:
                            break
            except Exception:                                     # noqa: BLE001
                rel = None
        if not rel:
            # DatasetTask（如 goal2_duplicate）通常不持有骨干
            print(f"  {g:<18} 无 ckpt_rel（不占常驻模型）")
            continue
        target = resolve_ckpt(root, rel)
        try:
            paths = resolve_ckpts(root, rel)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  {g:<18} rel={rel}  !! 解析失败: {str(exc)[:70]}")
            _record("G6 内存", "FAIL", f"{g}: {type(exc).__name__}（见上）")
            return
        n = len(paths)
        print(f"  {g:<18} rel={rel:<34} 解析出 {n} 份"
              f"{'  ← 目录里没有 ' + target.name if n > 1 else ''}")
        rows.append((g, rel, n))
        key = (str(root.resolve()), rel)
        distinct[key] = n                     # 同 ckpt_rel 已被进程级缓存共享 → 只算一份

    total = sum(distinct.values())
    print()
    print(f"不同 ckpt_rel 数 = {len(distinct)} | **常驻模型总数 = {total}**"
          f"（已按进程级权重复用折算）")
    if total > 8:
        _record("G6 内存", "WARN",
                f"常驻模型 {total} 个 → 建议在容器里跑 scripts/mem_audit.py --trace 量峰值")
    elif any(n > 1 for _, _, n in rows):
        _record("G6 内存", "WARN",
                f"有目录走了多折集成（总 {total} 个模型）；"
                f"确认那是有意的，而不是训练快照混在里面")
    else:
        _record("G6 内存", "PASS", f"常驻模型 {total} 个（每 Goal 单份）")


# --------------------------------------------------------------------------- #
# G7 后处理：真实体积下的 _bridge 峰值内存 / 耗时
# --------------------------------------------------------------------------- #
@_guard("G7 后处理")
def gate_postprocess(info: dict) -> None:
    """对**真实体积**跑一次 ``_bridge``，量峰值内存与耗时。

    **这条守卫的是一次真实事故**：``_bridge`` 曾用 ``21³ 稠密结构元`` 做
    ``binary_closing``，实测 ``240x240x155`` 下单次 **+701 MB / 31.7 s**
    （``clean_pair`` 调用两次 ⇒ 一例约 1.4 GB 峰值），直接把容器 OOM-kill。

    而这个问题**只在"掩膜从空变成非空"之后才出现** —— 空掩膜会在
    ``clean_mask`` 的 ``if not m.any(): return`` 提前返回，永远走不到 ``_bridge``。
    同时它也**测不出来**：仓里所有单元测试用的都是 ``4³``~``8³`` 的小体积。

    所以这里刻意用真实尺寸 + 按峰值判定，把"只吃内存、不报错"的回归挡在提交之前。
    """
    import threading
    import time

    import numpy as np

    from tasks.goal5_segmentation.config import Goal5Config
    from tasks.goal5_segmentation.postprocess import _bridge

    cfg = Goal5Config()
    shape = (240, 240, 155)                                # 真实 1mm 脑尺寸
    m = np.zeros(shape, dtype=bool)
    m[100:132, 110:146, 58:96] = True                      # 瘤体大小的实心块
    m[30, 30, 20] = m[210, 200, 130] = True                # 两处孤立假阳性斑点
    print(f"体积 {shape} = {np.prod(shape) / 1e6:.1f} M 体素 | "
          f"bridge_mm={cfg.bridge_mm} | 前景 {int(m.sum())} 体素")

    try:
        import psutil
        proc = psutil.Process()
        peak = [proc.memory_info().rss]
        base = peak[0]
        stop = threading.Event()

        def sampler() -> None:
            while not stop.is_set():
                peak[0] = max(peak[0], proc.memory_info().rss)
                time.sleep(0.005)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
    except Exception:                                          # noqa: BLE001
        proc = None
        peak, base, stop, th = [0], 0, None, None

    t0 = time.perf_counter()
    out = _bridge(m, float(cfg.bridge_mm), tuple(cfg.common_spacing))
    dt = time.perf_counter() - t0
    if stop is not None:
        stop.set()
        th.join(timeout=0.5)
        mem = (peak[0] - base) / 1e6
        print(f"单次 _bridge: 峰值增量 {mem:.0f} MB  耗时 {dt:.2f} s  "
              f"（clean_pair 会调用两次 → 一例约 {mem * 2:.0f} MB）")
    else:
        mem = float("nan")
        print(f"单次 _bridge: 耗时 {dt:.2f} s（无 psutil，未量内存）")
    print(f"输出前景 {int(out.sum())} 体素")

    # 阈值刻意留大余量：当前实现实测 ~250 MB / 0.55s，旧的稠密结构元实测 ~700 MB / 32s。
    # 卡在 550 MB / 5s 就能把两者稳稳分开，机器抖动不会误报。
    if mem == mem and mem > 550:                               # NaN 检查
        _record("G7 后处理", "FAIL",
                f"_bridge 单次峰值 {mem:.0f} MB（>550MB）→ 极可能退回了稠密结构元，"
                f"评测时会把容器 OOM-kill")
    elif dt > 5.0:
        _record("G7 后处理", "FAIL", f"_bridge 单次 {dt:.1f}s（>5s）→ 一例会拖到分钟级")
    elif (mem == mem and mem > 400) or dt > 2.5:
        _record("G7 后处理", "WARN",
                f"_bridge 峰值 {mem:.0f} MB / {dt:.2f}s 偏高（期望 ~250MB / ~0.5s）")
    else:
        suffix = f"{mem:.0f} MB / {dt:.2f}s" if mem == mem else f"{dt:.2f}s"
        _record("G7 后处理", "PASS", f"真实体积下 _bridge {suffix}")


# --------------------------------------------------------------------------- #
# G5 端到端（完整 runner + 离线校验）
# --------------------------------------------------------------------------- #
@_guard("G5 端到端")
def gate_end_to_end(dataset: Path, info: dict, out_root: Path) -> None:
    import uuid
    from dataclasses import replace

    from core.runner import EvaluationJob, EvaluationRunner
    from scripts.validate_output import validate

    settings = replace(
        info["settings"],
        workspace=out_root.parent,
        answer_root=out_root,
        log_root=out_root.parent / "logs",
        callback_url=None,
    )
    runner = EvaluationRunner(settings)
    evaluation_id = "pre-submit"
    result = runner.run(
        EvaluationJob(
            request_id=f"pre-submit-{uuid.uuid4()}",
            evaluation_id=evaluation_id,
            dataset_path=dataset,
        ),
        send_callback=False,
    )
    print(f"产出目录 : {result}")
    rc = validate(Path(result), None)
    if rc != 0:
        _record("G5 端到端", "FAIL", f"输出格式校验未通过（{result}）")
    else:
        _record("G5 端到端", "PASS", f"完整链路 + 输出格式校验通过（{result}）")


# --------------------------------------------------------------------------- #
def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="提交前一键预检（GO / NO-GO）")
    ap.add_argument("--dataset", default=DATA_DEFAULT, help=f"数据根（默认 {DATA_DEFAULT}）")
    ap.add_argument("--limit", type=int, default=5, help="冒烟例数（默认 5）")
    ap.add_argument("--train-config", default=TRAIN_CFG_DEFAULT)
    ap.add_argument("--full", action="store_true",
                    help="额外跑完整 runner 一遭（慢，但这是提交前的最后一道保险）")
    ap.add_argument("--out", default="/tmp/pre_submit_answer", help="--full 时的产出目录")
    a = ap.parse_args(argv)

    dataset = Path(a.dataset).expanduser()
    print("=" * 88)
    print("提交前预检 —— 一条命令给出 GO / NO-GO")
    print("=" * 88)

    info = gate_env(dataset, a.limit) or {}
    if info:
        gate_weights(info)
        gate_preprocess(Path(a.train_config).expanduser(), info)
        gate_memory(info)
        gate_postprocess(info)
        if dataset.is_dir():
            gate_smoke(dataset, a.limit, info)
        if a.full and dataset.is_dir():
            gate_end_to_end(dataset, info, Path(a.out).expanduser())

    print()
    print("=" * 88)
    print("汇总")
    print("=" * 88)
    for name, status, summary in _GATES:
        icon = {"PASS": "✔", "FAIL": "✘", "WARN": "!", "SKIP": "-"}.get(status, "?")
        print(f"  [{icon}] {name:<18} {summary}")

    go = not _BLOCKING
    print()
    print("=" * 88)
    if go:
        warn = [n for n, s, _ in _GATES if s == "WARN"]
        print(f"结论：**GO** —— 可以提交" + (f"（有 {len(warn)} 项告警：{warn}）" if warn else ""))
    else:
        print(f"结论：**NO-GO** —— 以下关卡未通过，先别提交：{sorted(_BLOCKING)}")
        print("     每关的详细输出在上面对应小节里；把整份输出贴回来即可定位。")
    print("=" * 88)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
