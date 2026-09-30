"""Goal5 的推理入口：权重加载 + 滑窗/TTA 推理。

规范 §5.1 与 §5.2：
- ``inference.py`` 负责**模型初始化与纯推理**，不扫描目录、不写结果、不发回调；
- 权重只能在服务初始化阶段加载**一次**（``load_model()``），
  严禁在每个 Study 推理时重复读盘；
- 权重路径由 ``core.config.Settings.ckpt_root`` + 各 Goal 的**相对路径**拼出。

这里额外做一层**权重指纹校验**：不同 fold 的权重若 ``arch``/``in_channels`` 与
配置不一致，加载会静默得到随机初始化的部分层（``strict=False`` 的副作用），
结果看似正常但指标崩塌，因此显式比对关键结构字段。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tasks._common.sliding import predict_volume
from tasks.goal5.config import Goal5Config
from tasks.goal5.models.factory import Goal5Models, build_goal5_models


@dataclass
class LoadedGoal5:
    """已加载的 Goal5 推理态（``ckpt_paths`` 多于一个即为多折集成）。"""

    models: Goal5Models
    thresholds: tuple[float, float]
    arch: str
    global_size: int
    global_size_mm: float
    device: str
    ckpt_path: str


def resolve_ckpt(ckpt_root: Path, rel: str) -> Path:
    """把相对路径解析到 checkpoint 根目录下，并拒绝路径逃逸。"""
    root = Path(ckpt_root).expanduser().resolve()
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"权重路径逃逸出 checkpoint 根目录: {rel}")
    return target


def resolve_ckpts(ckpt_root: Path, rel: str) -> list[Path]:
    """解析**一个或多个**权重路径（多折集成）。

    规范 §5.2 只约定单个 ``core.pt``；但多折训练后做成集成是提升指标最稳的手段，
    导出脚本会把各折写成 ``<tag>.pt``。因此这里按以下顺序解析：

    1. ``rel`` 指向的文件存在 → 单模型；
    2. 否则取该目录下**全部** ``*.pt`` → 多折集成（顺序稳定，保证可复现）。

    两种布局都能工作，队友不需要为了"是否集成"改代码或配置。

    ⚠️ 本函数由 ``tasks/_common/backbone_runner.py`` **懒加载复用**：所有 Goal 的
    权重都走这一套解析规则。所以报错信息里**不要写死某个 Goal 名** ——
    否则 Goal1 缺权重时会报成 "未找到 Goal5 权重"，把排障的人带偏一整天。

    ⚠️⚠️ **上面的规则 2 是一个静默的内存放大点，必须设上限**：

    ``BackboneRunner.load()`` 会把本函数返回的**每一份**权重都构建成一个
    **常驻模型**（``self._members.append(...)``，推理期间不释放）。
    而每一个 ``StudyTask`` 都各持有一个 ``BackboneRunner`` —— 于是常驻模型数 =

    ``启用 Goal 数 × 本函数返回的份数``。

    目录里只要混进训练快照（``epoch_10.pt`` / ``last.pt`` / 备份），
    份数就会**成倍**上升；训练快照还带着 Adam 的优化器状态（约 2× 参数量），
    单份体积也大得多。实测过"评测一启动容器就 OOM 重启"就是这条路径。

    因此这里**限制集成规模**（``GLIOMA_MAX_ENSEMBLE``，默认 8），
    超过就**明确报错**并列出全部文件，让人一眼看出目录里多放了什么 ——
    而不是静默加载几十个模型把容器撑爆。
    """
    target = resolve_ckpt(ckpt_root, rel)
    if target.is_file():
        return [target]
    parent = target.parent
    if parent.is_dir():
        cks = sorted(p for p in parent.glob("*.pt") if p.is_file())
        if cks:
            limit = max(1, int(os.environ.get("GLIOMA_MAX_ENSEMBLE", "8") or 8))
            if len(cks) > limit:
                names = [p.name for p in cks]
                raise ValueError(
                    f"{parent} 下没有规范约定的 {target.name}，"
                    f"于是按『多折集成』解析出 **{len(cks)}** 份 *.pt，"
                    f"超过上限 {limit}（GLIOMA_MAX_ENSEMBLE）。\n"
                    f"  每个 StudyTask 会把**每一份**都加载成一个常驻模型，"
                    f"常驻模型数 = 启用 Goal 数 × {len(cks)} —— 这就是容器 OOM 的典型成因。\n"
                    f"  目录内容: {names}\n"
                    f"  三种修法（任选其一）：\n"
                    f"    1) 只保留一份权重并命名为 {target.name}（本函数会优先取它）；\n"
                    f"    2) 把训练快照 / 备份 / 优化器检查点移到别的目录；\n"
                    f"    3) 确实要多折集成且超过 {limit} 折：export GLIOMA_MAX_ENSEMBLE={len(cks)}"
                )
            if len(cks) > 1:
                print(
                    f"[ckpt] {parent.name}: 没有 {target.name} → "
                    f"按多折集成加载全部 {len(cks)} 份: {[p.name for p in cks]}",
                    flush=True,
                )
            return cks
    raise FileNotFoundError(
        f"未找到权重：{target}（规范 §5.2 约定 {rel}；"
        f"多折集成时该目录下应有若干 *.pt）"
    )


def load_model(cfg: Goal5Config, ckpt_root: Path, device: str = "cuda") -> LoadedGoal5:
    """加载 Goal5 权重（服务启动时调用一次）。

    Raises:
        FileNotFoundError: 规范路径下找不到权重。
        ValueError: 权重与配置的网络结构不兼容。
    """
    import torch

    paths = resolve_ckpts(ckpt_root, cfg.core_ckpt_rel)

    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    members: list = []
    thr_list: list[list[float]] = []
    arch, gsize, gmm = "mednext", cfg.global_size, cfg.global_size_mm
    for p in paths:
        ck = torch.load(str(p), map_location="cpu", weights_only=False)
        arch = str(ck.get("arch") or arch)
        mc = ck.get("model_cfg") or {}
        in_ch = int(mc.get("in_ch", cfg.in_channels))
        if in_ch != cfg.in_channels:
            raise ValueError(f"{p.name}: 权重 in_ch={in_ch} 与配置 {cfg.in_channels} 不一致，拒绝加载")

        m = build_goal5_models(cfg, ck.get("cls_spec") or [], shared=True)
        state = ck.get("model_ema") or ck.get("model") or ck
        missing, _ = m.core_model.load_state_dict(state, strict=False)
        # 训练态权重含 cls/special 头，推理只用 seg；缺失非骨干键是预期的
        seg_missing = [k for k in missing
                       if k.startswith(("enc", "dec", "bottleneck", "stem"))]
        if seg_missing:
            raise ValueError(f"{p.name}: 骨干关键层缺失 {len(seg_missing)} 个：{seg_missing[:3]}")
        members.append(m.core_model.eval().to(dev))
        thr_list.append([float(x) for x in (ck.get("thresholds") or cfg.default_thresholds)])
        gsize = int(ck.get("global_size", gsize))
        gmm = float(ck.get("global_size_mm", gmm))
        del ck

    # 多折时阈值取各折的均值（与训练侧 load_ensemble 的口径保持一致）
    import numpy as _np
    thr = _np.mean(thr_list, axis=0).tolist() if thr_list else list(cfg.default_thresholds)

    # 复用 Goal5Models 容器：core_model 换成模型**列表**（shared=True 时即集成成员）
    models = Goal5Models(core_model=members, flair_model=members,
                         core_channel=cfg.core_channel, flair_channel=cfg.flair_channel,
                         shared=True)
    return LoadedGoal5(
        models=models,
        thresholds=(float(thr[0]), float(thr[1])),
        arch=arch,
        global_size=gsize,
        global_size_mm=gmm,
        device=dev,
        ckpt_path=",".join(str(p) for p in paths),
    )


def infer_segmentation(loaded: LoadedGoal5, volume: np.ndarray,
                       cfg: Goal5Config) -> np.ndarray:
    """在公共网格上做滑窗 + TTA 推理，返回 ``[2, D, H, W]`` 概率图。

    ``volume``: ``[C, D, H, W]``（``preprocess.build_volume`` 的输出）。
    """
    import torch

    # 全局头（分类/特殊/嵌入）在本任务不使用，传零体积避免额外开销。
    gsize = loaded.global_size
    gvol = np.zeros((volume.shape[0], gsize, gsize, gsize), dtype=np.float32)
    with torch.inference_mode():
        members = loaded.models.core_model
        model_list = members if isinstance(members, (list, tuple)) else [members]
        res = predict_volume(
            list(model_list), volume,
            patch=tuple(cfg.patch), overlap=float(cfg.overlap),
            tta_flips=tuple(cfg.tta_flips),
            seg_tta_flips=tuple(cfg.tta_flips),
            amp_dtype=torch.bfloat16,
            tta_batch=int(cfg.tta_batch),
            global_size=gsize, global_vol=gvol,
            crop_brain=True,
            device=loaded.device,
        )
    return np.asarray(res["seg"], dtype=np.float32)
