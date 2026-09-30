"""与团队串联工程（``Glioma_recognition``）衔接的桥接核心。

设计要点（严格对齐《赛道四_自建模型组_系统架构与协作规范.md》）：

1. **不再自己扫描目录、不写 answer/、不发起回调**——输入是团队 Loader 已经读进内存的
   ``Study``/``Series``，输出是强类型 ``*Result``，最终格式由团队的 Aggregator/Writer
   统一处理（协议单点维护）；
2. **一次推理，多个 Goal 复用**：本工程是共享骨干的多任务网络（分割 + 结构化 +
   特殫影像 + 嵌入），若让 Goal1/3/4/5 各自推理一遍会白白慢 4 倍。这里用带 LRU 的
   ``InferenceEngine`` 做进程内缓存，同时满足规范"不预取、不长期持有原图"的要求；
3. **掩码必须回到各自源序列空间**：在 1mm 公共网格上推理，再按 affine 逆变换回
   ``Series.affine``/``Series.image.shape``，并保证严格 0/1；
4. **core/flair 指向同一 series_uid 的契约陷阱**：团队 Writer 对"同一序列目录下
   两个不同掩码"会直接报错，这里显式合并处理。

本模块只依赖 numpy / nibabel / torch 与本工程的 ``src``，可脱离团队仓库独立测试。
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

# 控制台编码兜底：本包（导出/桥接）会打印 ✗ / ⚠️ 这类字符，**GBK 控制台上
# `print` 会直接抛 UnicodeEncodeError**。导入 config 即生效（见其
# `_make_std_streams_unicode_safe`）；这里只是把它拉进本包的导入图。
from src.utils import config as _config_unicodesafe                # noqa: F401

# ---- 团队异常类型（独立测试时降级为同名本地异常）----------------------------- #
try:                                                              # pragma: no cover
    from core.exceptions import InvalidInputError, MissingSeriesError
except Exception:                                                 # noqa: BLE001

    class MissingSeriesError(RuntimeError):
        """无任何可用影像序列。"""

    class InvalidInputError(RuntimeError):
        """输入非法。"""


#: 本工程根目录（integration/ 的上一层）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: 认不出模态的序列在 ``available`` / ``uid_by_mod`` 里的键（全放开口径，见
#: :func:`build_available_map`）。它**不是**一个输入通道：`build_volume_from_arrays`
#: 只认 `t1c/flair/t2/t1`，这个键的作用是**保住几何与 uid**，让掩膜有地方写回去。
OTHER_MODALITY = "other"

#: 团队约定的 checkpoint 根目录（规范 §5.2）
WORKSPACE = Path(os.environ.get("COMPETITION_WORKSPACE", "/2026aicompetition/workspace"))
CHECKPOINT_ROOT = Path(os.environ.get("GLIOMA_CHECKPOINT_ROOT", WORKSPACE / "checkpoint"))


# --------------------------------------------------------------------------- #
# 1) 序列识别：把团队的 Study 映射成 {模态: (image, affine, series_uid)}
# --------------------------------------------------------------------------- #
def _guess_modality_from_series(series: Any) -> str | None:
    """按 modality → SeriesDescription → ProtocolName → series_uid 逐级猜测模态。

    团队 Loader 会把 ``Series.modality`` 填成 SeriesType.xlsx 的值、sidecar 的
    SeriesDescription/ProtocolName，或退化为目录名（即 ``series_uid``）；
    因此这里对四个来源都尝试识别，并保留识别来源便于排障。
    """
    from src.data.labels import guess_modality

    meta = getattr(series, "metadata", None) or {}
    for text in (
        getattr(series, "modality", None),
        meta.get("SeriesDescription"),
        meta.get("ProtocolName"),
        getattr(series, "series_uid", None),
    ):
        if not text:
            continue
        mod = guess_modality(str(text))
        if mod:
            return mod
    return None


def build_available_map(study: Any) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, str]]:
    """``Study`` → ``({模态: (image, affine)}, {模态: series_uid})``。

    同一模态出现多条序列时保留**第一条**（团队 Loader 已按 series_uid 确定性排序），
    并在返回值中记录实际采用的 series_uid，供掩码写回时定位。

    **全放开口径**（与 ``src/data/dataset.build_case_volume`` 一致，见 README §7.2.1）：
    **认不出模态的序列不再被丢掉**。整例都被数据信息表标成 `其他`（或只有 DWI/ADC/SWI）
    时，这里把**第一路**这样的序列挂在键 :data:`OTHER_MODALITY` 下 —— 它进不了任何
    输入通道（`build_volume_from_arrays` 只认那 4 类，会全零通道），但它的
    **几何与 uid 必须留着**：掩膜没有"对应模态的源序列"可写回时退到它上面，
    至少保证每例都产出合规文件（缺文件 = 整例 0 分）。

    只有"**连一路 3D 影像都没有**"时才抛 ``MissingSeriesError`` —— 那时没有可对齐的
    几何；团队 Runner 会把它转成该例的兜底答案。这与"没有目标模态"是两回事。
    """
    available: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    uid_by_mod: dict[str, str] = {}
    for series in study.series:
        image = np.asarray(series.image)
        if image.ndim != 3:
            continue
        # 认不出模态 → 挂 `other`（**不丢**）；同键只取第一条
        key = _guess_modality_from_series(series) or OTHER_MODALITY
        if key in available:
            continue
        available[key] = (image, np.asarray(series.affine, dtype=float))
        uid_by_mod[key] = str(series.series_uid)
    if not available:
        raise MissingSeriesError(
            f"study {getattr(study, 'accession_number', '?')!r} 没有任何可用 3D 影像"
            f"（共 {len(study.series)} 条序列）。注意：**只有 `其他` 序列 / 只有 DWI 的"
            f"病例不算这一类** —— 那种情况会全零通道照走"
        )
    return available, uid_by_mod


# --------------------------------------------------------------------------- #
# 2) 空间逆变换：公共网格二值掩码 → 目标源序列空间
# --------------------------------------------------------------------------- #
def mask_to_source_space(mask_common: np.ndarray, common_affine: np.ndarray,
                         dst_shape: tuple, dst_affine: np.ndarray) -> np.ndarray:
    """把公共网格上的二值掩码回采样到目标序列空间（最近邻，严格 0/1）。"""
    from src.inference.writer import _resample_binary_to_grid

    if tuple(mask_common.shape) == tuple(dst_shape) and \
            np.allclose(np.asarray(common_affine, float), np.asarray(dst_affine, float), atol=1e-3):
        return (np.asarray(mask_common) > 0).astype(np.uint8)
    return _resample_binary_to_grid(
        (np.asarray(mask_common) > 0).astype(np.uint8),
        np.asarray(common_affine, float), tuple(dst_shape), np.asarray(dst_affine, float))


# --------------------------------------------------------------------------- #
# 3) 权重发现
# --------------------------------------------------------------------------- #
def resolve_checkpoints(goal_dir: str = "goal5_segmentation") -> list[str]:
    """按团队约定查找本工程的权重（共享多任务权重，可复制/硬链接到各 goal 目录）。

    查找顺序：
      1. ``$GLIOMA_CKPT``（显式指定，逗号分隔，多折集成）；
      2. ``{CHECKPOINT_ROOT}/{goal_dir}/*.pt``（团队约定路径）；
      3. 本工程 ``checkpoints/g4_fold*/best.pth``（本地演练兜底）。
    """
    env = (os.environ.get("GLIOMA_CKPT") or "").strip()
    if env:
        return [p.strip() for p in env.replace(";", ",").split(",") if p.strip()]

    found: list[str] = []
    d = Path(CHECKPOINT_ROOT) / goal_dir
    if d.is_dir():
        found = sorted(str(p) for p in d.glob("*.pt")) + sorted(str(p) for p in d.glob("*.pth"))
    if not found:
        found = sorted(str(p) for p in (PROJECT_ROOT / "checkpoints").glob("g4_fold*/best.pth"))
    if not found:
        raise FileNotFoundError(
            f"未找到推理权重。请设置 GLIOMA_CKPT，或把权重放到 {d}/（团队约定路径），"
            f"或先训练产出 {PROJECT_ROOT}/checkpoints/g4_fold*/best.pth"
        )
    return found


# --------------------------------------------------------------------------- #
# 4) 推理引擎（进程内单例 + LRU 缓存，避免多 Goal 重复推理）
# --------------------------------------------------------------------------- #
class InferenceEngine:
    """共享骨干的一次推理结果：分割概率、结构化概率、特殊影像概率、重复影像嵌入。

    ``infer(study)`` 的结果按 accession 做 LRU 缓存（默认 2 个 Study），
    保证 Goal1/2A/3/4/5 在一次 ``run_study`` 中只做**一次**前向。
    """

    def __init__(self, ckpt_paths: list[str] | None = None, device: str | None = None,
                 cache_size: int = 2) -> None:
        from src.inference.pipeline import GliomaPipeline
        from src.utils.config import load_config

        self.pre = load_config("preprocess.yaml")
        self.labels = load_config("labels.yaml")["fields"]
        ckpts = ckpt_paths or resolve_checkpoints()
        self.pipeline = GliomaPipeline([str(p) for p in ckpts], device=device or "cuda")
        self.device = self.pipeline.device
        self.ckpt_paths = ckpts
        self._cache: "OrderedDict[str, dict]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))

    # ------------------------------------------------------------------ #
    def infer(self, study: Any) -> dict:
        """对当前 Study 做一次完整推理；同 accession 直接命中缓存。"""
        acc = str(study.accession_number)
        hit = self._cache.get(acc)
        if hit is not None:
            self._cache.move_to_end(acc)
            return hit

        from src.data.dataset import brain_center, build_volume_from_arrays, global_view
        from src.inference.pipeline import postprocess

        available, uid_by_mod = build_available_map(study)
        log: list[str] = []
        vol, common_aff, used = build_volume_from_arrays(available, self.pre, log)
        res = self.pipeline.predict_prob(None, vol)               # 只用 vol，不读盘

        inf = self.pre["inference"]
        spacing = tuple(float(np.linalg.norm(np.asarray(common_aff)[:3, i])) for i in range(3))
        core = res["seg"][0] > float(self.pipeline.thresholds[0])
        peri = res["seg"][1] > float(self.pipeline.thresholds[1])
        core, peri = postprocess(core, peri, int(inf["min_tumor_voxels"]), spacing,
                                 keep_n=int(inf.get("keep_components", 3)),
                                 bridge_mm=float(inf.get("bridge_mm", 10.0)))

        from src.inference.structured import derive_rules, special_heuristics

        ch_names = [c["name"] for c in self.pre["channels"]]
        rules = derive_rules(vol, ch_names, core, peri, spacing, int(inf["min_tumor_voxels"]),
                             affine=np.asarray(common_aff, float))
        probs = {f["key"]: res["cls"][i] for i, f in enumerate(self.labels)
                 if i < len(res["cls"])}
        rules["tumor_prob_model"] = float(probs["TumorProbability"][0]) \
            if probs.get("TumorProbability") is not None and len(probs["TumorProbability"]) else 0.0

        special = np.asarray(res["special"], float).ravel()
        p_fake_m = float(special[0]) if special.size else 0.5
        p_stitch_m = float(special[1]) if special.size > 1 else 0.5
        h_fake, h_stitch = special_heuristics(vol, self.pre)
        if self.pipeline.net.get("special_trained"):
            # 模型头已训练 → 以模型为主；启发式仅在**非常确定**时才补充，
            # 避免启发式的假阳性抬高正常影像的假人体/拼接概率（实测曾把
            # 正常脑影像的 IsStitchedProb 抬到 0.56）。
            p_fake = max(p_fake_m, h_fake if h_fake >= 0.9 else 0.0)
            p_stitch = max(p_stitch_m, h_stitch if h_stitch >= 0.9 else 0.0)
        else:
            p_fake, p_stitch = h_fake, h_stitch

        payload = {
            "accession": acc,
            "vol": vol, "common_affine": np.asarray(common_aff, float),
            "used": used, "uid_by_mod": uid_by_mod,
            "available": available,
            "core_common": core.astype(np.uint8), "peri_common": peri.astype(np.uint8),
            "rules": rules, "probs": probs,
            "not_human_probability": float(np.clip(p_fake, 0.0, 1.0)),
            "stitched_probability": float(np.clip(p_stitch, 0.0, 1.0)),
            "embed": np.asarray(res["embed"], np.float32),
            "warnings": list(log),
        }
        self._cache[acc] = payload
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)                        # 流式：及时释放
        return payload

    def release(self, accession: str) -> None:
        """显式释放某个 Study 的大数组（Runner 写完当前 Study 后可调用）。"""
        self._cache.pop(str(accession), None)


# --------------------------------------------------------------------------- #
# 5) 进程内单例
# --------------------------------------------------------------------------- #
_ENGINE: InferenceEngine | None = None


def get_engine() -> InferenceEngine:
    """进程内单例；模型只在服务启动阶段加载一次（规范 §5.2）。"""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InferenceEngine()
    return _ENGINE
