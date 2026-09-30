"""Goal1~Goal5 插件（对齐团队 ``StudyTask`` / ``DatasetTask`` 契约）。

每个 Task 只做三件事：
1. 从 ``PipelineContext.study`` 取数据（不碰任何比赛路径）；
2. 通过共享 ``InferenceEngine`` 拿结果（一次推理、多 Goal 复用）；
3. 返回团队的强类型 ``*Result``（不组装 ``prediction.json``、不写文件、不回调）。

Goal5 额外负责"掩码回到各自源序列空间"，并处理"core/flair 命中同一序列"的契约陷阱。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tasks.base import DatasetTask, StudyTask
from tasks.results import (
    BinaryResult,
    CategoricalResult,
    DuplicatePair,
    DuplicateResult,
    Goal1Result,
    Goal3Result,
    Goal4Result,
    Goal5Result,
    StitchedResult,
)

from .common import get_engine, mask_to_source_space


# --------------------------------------------------------------------------- #
# Goal1：假人体 / 非人体（基础考核项）
# --------------------------------------------------------------------------- #
class AuthenticityTask(StudyTask[Goal1Result]):
    name = "goal1"

    def load_model(self) -> None:
        get_engine()                                              # 启动阶段加载一次

    def predict(self, context: Any) -> Goal1Result:
        payload = get_engine().infer(context.study)
        return Goal1Result(not_human_probability=float(payload["not_human_probability"]))


# --------------------------------------------------------------------------- #
# Goal2-A：拼接影像（基础考核项）
# --------------------------------------------------------------------------- #
class StitchedTask(StudyTask[StitchedResult]):
    name = "goal2_stitched"

    def load_model(self) -> None:
        get_engine()

    def predict(self, context: Any) -> StitchedResult:
        payload = get_engine().infer(context.study)
        return StitchedResult(stitched_probability=float(payload["stitched_probability"]))


# --------------------------------------------------------------------------- #
# Goal3：胶质瘤概率（ROC-AUC 评估）
# --------------------------------------------------------------------------- #
class TumorTask(StudyTask[Goal3Result]):
    name = "goal3"

    def load_model(self) -> None:
        get_engine()

    def predict(self, context: Any) -> Goal3Result:
        payload = get_engine().infer(context.study)
        rules = payload["rules"]
        probs = payload["probs"]
        model_p = 0.0
        tp = probs.get("TumorProbability")
        if tp is not None and np.size(tp):
            model_p = float(np.asarray(tp).ravel()[0])
        # 与 assemble_prediction 相同的融合口径：模型为主，掩码证据为辅
        value = float(np.clip(0.8 * model_p + 0.2 * (1.0 if rules.get("has_tumor") else 0.0),
                              0.0, 1.0))
        return Goal3Result(tumor_probability=value)


# --------------------------------------------------------------------------- #
# Goal5：分割（任务A 核心区 / 任务B 周围总异常区）
# --------------------------------------------------------------------------- #
class SegmentationTask(StudyTask[Goal5Result]):
    name = "goal5"

    def load_model(self) -> None:
        get_engine()

    def predict(self, context: Any) -> Goal5Result:
        payload = get_engine().infer(context.study)
        available = payload["available"]
        uid_by_mod = payload["uid_by_mod"]
        used = payload["used"]
        common_aff = payload["common_affine"]

        # 实际采用的模态（例如 T1C 缺失时用 T1/T2 顶替）→ 决定掩码写回哪条源序列。
        # **全放开**：4 个通道一个都没填上时 `used` 是**空的**（原写法 `next(iter(...))`
        # 会抛 StopIteration），此时退到 `available` 里任意一路序列 —— 认不出模态的
        # 序列也会以 `other` 键留在那里（见 `common.build_available_map`），
        # 它至少提供了可写回的几何，保证每例都产出合规文件（缺文件 = 整例 0 分）。
        fallback_mod = next(iter(used.values()), None) or next(iter(available), None)
        core_mod = used.get("t1c") or used.get("t1") or fallback_mod
        flair_mod = used.get("flair") or used.get("t2") or core_mod
        core_uid = uid_by_mod[core_mod]
        flair_uid = uid_by_mod[flair_mod]

        core_img, core_aff = available[core_mod]
        flair_img, flair_aff = available[flair_mod]
        core_mask = mask_to_source_space(payload["core_common"], common_aff,
                                         core_img.shape, core_aff)
        flair_mask = mask_to_source_space(payload["peri_common"], common_aff,
                                          flair_img.shape, flair_aff)

        # 契约陷阱：团队 Writer 对"同一 series_uid 目录下两个不同掩码"直接报错。
        # 当 core 与 flair 解析到同一条序列时，取并集作为两者共同的掩码，
        # 保证输出自洽（该情形下语义退化为"总异常区"，是已知且可接受的行为）。
        if core_uid == flair_uid and not np.array_equal(core_mask, flair_mask):
            merged = ((core_mask > 0) | (flair_mask > 0)).astype(np.uint8)
            core_mask = merged
            flair_mask = merged.copy()

        return Goal5Result(
            core_mask=np.ascontiguousarray(core_mask, dtype=np.uint8),
            core_source_series_uid=core_uid,
            flair_mask=np.ascontiguousarray(flair_mask, dtype=np.uint8),
            flair_source_series_uid=flair_uid,
        )


# --------------------------------------------------------------------------- #
# Goal4：结构化诊断（14 个字段 + 结论文本）
# --------------------------------------------------------------------------- #
def _categorical(value: Any) -> CategoricalResult:
    if isinstance(value, dict):
        predicted = value.get("predicted")
        probabilities = {str(k): float(v) for k, v in (value.get("probabilities") or {}).items()}
    else:
        predicted, probabilities = value, {}
    return CategoricalResult(predicted=predicted, probabilities=probabilities)


def _binary(value: Any) -> BinaryResult:
    """二分类字段：兼容 ``{"present":..,"XxxProbability":..}`` 与 ``{"clear":..}`` 两种形态。"""
    if isinstance(value, dict):
        present = value.get("present", value.get("clear", False))
        prob = next((float(v) for k, v in value.items()
                     if k.lower().endswith("probability")), 0.0)
        return BinaryResult(present=bool(present), probability=prob)
    return BinaryResult(present=bool(value), probability=0.0)


def build_goal4_result(prediction: dict, conclusion: str) -> Goal4Result:
    """把 ``assemble_prediction`` 的规范字段映射到团队的 ``Goal4Result``。"""
    loc = prediction.get("Location")
    location = loc.get("predicted") if isinstance(loc, dict) else (loc or "Other")
    return Goal4Result(
        location=str(location or "Other"),
        morphology=_categorical(prediction.get("Morphology")),
        who_grade=_categorical(prediction.get("WHO_Grade")),
        enhancement=_binary(prediction.get("Enhancement")),
        enhancement_pattern=_categorical(prediction.get("EnhancementPattern")),
        necrosis=_binary(prediction.get("Necrosis")),
        cystic_change=_binary(prediction.get("CysticChange")),
        hemorrhage=_binary(prediction.get("Hemorrhage")),
        calcification=_binary(prediction.get("Calcification")),
        margin_clear=_binary(prediction.get("Margin")),
        lobulation=_binary(prediction.get("Lobulation")),
        signal_t2wi=_categorical(prediction.get("Signal_T2WI")),
        signal_flair=_categorical(prediction.get("Signal_FLAIR")),
        conclusion=conclusion,
    )


class DiagnosisTask(StudyTask[Goal4Result]):
    """Goal4 使用 Goal5 的掩码统计 + 共享骨干的结构化头 + 规则校正。

    顺序上排在 Goal5 之后（与团队 Dummy 的绑定顺序一致），因此可以直接复用
    ``context.goal5`` 的掩码证据。
    """
    name = "goal4"

    def load_model(self) -> None:
        get_engine()

    def predict(self, context: Any) -> Goal4Result:
        from src.inference.structured import make_conclusion
        from src.inference.writer import assemble_prediction

        payload = get_engine().infer(context.study)
        engine = get_engine()
        fields = engine.labels
        prediction = assemble_prediction(payload["accession"], fields, payload["probs"],
                                         payload["rules"])
        conclusion = make_conclusion(payload["accession"], prediction)
        return build_goal4_result(prediction, conclusion)


# --------------------------------------------------------------------------- #
# Goal2-B：重复影像（DatasetTask，流式 update + 全局 finalize）
# --------------------------------------------------------------------------- #
class DuplicateTask(DatasetTask[DuplicateResult]):
    """逐 Study 累积轻量特征，finalize 时做候选检索与概率标定。

    规范要求 ``update()`` 不长期持有原图或完整 mask —— 这里只保留
    accession、128 维嵌入与手工指纹（指纹含 16³ 缩略图，量级为 KB）。
    """
    name = "goal2_duplicate"

    def __init__(self) -> None:
        self.reset()

    def load_model(self) -> None:
        get_engine()

    def reset(self) -> None:
        self._order: list[str] = []
        self._embeds: dict[str, np.ndarray] = {}
        self._feats: dict[str, dict] = {}

    def update(self, study: Any, context: Any) -> None:
        payload = get_engine().infer(study)
        acc = str(study.accession_number)
        if acc not in self._embeds:
            self._order.append(acc)
        self._embeds[acc] = np.asarray(payload["embed"], np.float32)
        self._feats[acc] = _fingerprint_from_payload(payload)

    def finalize(self) -> DuplicateResult:
        from src.inference.duplicate import match_pairs
        from src.utils.config import load_config

        pre = load_config("preprocess.yaml")
        inf = pre["inference"]
        pairs = match_pairs(
            self._embeds, None,
            topk=int(inf.get("duplicate_topk", 200)),
            per_study_k=int(inf.get("duplicate_per_study", 50)),
            feats=self._feats,
            w_fp=float(inf.get("fp_weight", 0.5)),
        )
        # 团队 OutputValidator 明确拒绝 self-pair；仅有单个 Study 时 match_pairs 会
        # 返回 (a, a) 兜底对，这里必须剔除，交由团队 Writer 自行补位。
        out = tuple(
            DuplicatePair(left_accession=str(p["a"]), right_accession=str(p["b"]),
                          probability=float(np.clip(p["prob"], 0.0, 1.0)))
            for p in pairs if str(p["a"]) != str(p["b"])
        )
        self.reset()                                              # 及时释放状态
        return DuplicateResult(pairs=out)


def _fingerprint_from_payload(payload: dict) -> dict:
    """用已加载的公共网格体积构造指纹（避免再次读盘）。

    与训练侧 ``writer.case_fingerprint`` 保持同一结构（模态几何 + 直方图 + 16³ 缩略图），
    但数据源换成内存中的源序列数组。
    """
    fp: dict[str, Any] = {"mods": sorted(payload["available"].keys()), "hist": {}, "thumb": {}}
    for mod, (image, affine) in payload["available"].items():
        arr = np.asarray(image, np.float32)
        flat = arr.ravel()[::37]
        if flat.size:
            lo, hi = np.percentile(flat, 1), np.percentile(flat, 99)
            hist, _ = np.histogram(flat, bins=8, range=(float(lo), float(hi) + 1e-6))
            hist = hist / max(1, hist.sum())
            fp["hist"][mod] = [round(float(x), 4) for x in hist]
        step = [max(1, int(s // 16)) for s in arr.shape]
        thumb = np.asarray(arr[::step[0], ::step[1], ::step[2]][:16, :16, :16], np.float32)
        fp["thumb"][mod] = np.round(np.nan_to_num(thumb), 3).tolist()
        fp[f"shape_{mod}"] = list(arr.shape)
        fp[f"zoom_{mod}"] = [round(float(np.linalg.norm(np.asarray(affine)[:3, i])), 3)
                             for i in range(3)]
        fp[f"aff_{mod}"] = [round(float(x), 3) for x in np.asarray(affine).ravel()]
    return fp
