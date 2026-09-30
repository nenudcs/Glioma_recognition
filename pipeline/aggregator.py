from __future__ import annotations

from core.exceptions import InvalidTaskResultError
from pipeline.context import PipelineContext
from tasks.results import BinaryResult, CategoricalResult

#: 掩码 URI 模板（**官方契约已定，不再待确认**）
#:
#: 依据：上游仓库 ``nenudcs/Glioma_recognition`` 的 ``README.md``「当前规范解释」原文——
#:
#:     ``SegmentationMaskURI`` 相对于病例目录，格式为 ``./{SeriesUid}/{SeriesUid}.nii.gz``
#:
#: 由这一句，原先的【待确认】全部关闭，且**三条**同时确定：
#:
#: 1. **模板**就是 ``./{uid}/{uid}.nii.gz``（即本常量，无需改动；
#:    官方材料里那个 ``./output/sub-*_core.nii.gz`` 的写法不是规范）；
#: 2. **``{uid}`` 的身份**是**磁盘上的那一层目录名** —— 即
#:    ``<AccessionNumber>/<SeriesUid>/<SeriesUid>.nii.gz`` 里的 ``<SeriesUid>``。
#:    所以 ``data.loader`` 的 ``Series.series_uid`` 与
#:    ``data.series_selector.infer_uid_from_path`` 都是**磁盘优先**、
#:    sidecar 的 ``SeriesInstanceUID`` 只兜底（见 UPSTREAM_CONTRACT_AUDIT §4.5）；
#: 3. **归属模态**：官方 §4「核心区写入其**来源 T1 增强** Series UID 目录；
#:    周围区写入其 **Flair/T2** Series UID 目录」—— 对应
#:    ``tasks/goal5_segmentation/task.py`` 的 ``CORE_SOURCE_MODALITIES`` /
#:    ``FLAIR_SOURCE_MODALITIES``。
#:
#: 因此本模板与 ``OutputWriter._write_masks`` 的物理落盘位置严格一致，
#: 且因为 ``{uid}`` 已等于输入数据的目录名，答案目录与输入病例目录**一一对应**
#: （``OutputValidator`` 还会逐例校验 URI 指向真实文件、shape 精确相同、affine 在容差内）。
MASK_URI_TEMPLATE = "./{uid}/{uid}.nii.gz"


class PredictionAggregator:
    def build(self, context: PipelineContext) -> dict[str, object]:
        if not all(
            (
                context.goal1,
                context.goal2_stitched,
                context.goal3,
                context.goal4,
                context.goal5,
            )
        ):
            raise InvalidTaskResultError(
                f"incomplete results for {context.study.accession_number}"
            )

        goal4 = context.goal4
        goal5 = context.goal5
        prediction: dict[str, object] = {
            "AccessionNumber": context.study.accession_number,
            "IsNotHumanBodyProb": context.goal1.not_human_probability,
            "IsStitchedProb": context.goal2_stitched.stitched_probability,
            "ProcessingTime_ms": context.processing_time_ms,
            "SegmentationMaskURI": {
                "core": self._mask_uri(goal5.core_source_series_uid),
                "flair": self._mask_uri(goal5.flair_source_series_uid),
            },
            "Prediction": {
                "TumorProbability": context.goal3.tumor_probability,
                "Location": goal4.location,
                "Morphology": self._category(goal4.morphology),
                "WHO_Grade": self._category(goal4.who_grade),
                "Enhancement": self._binary(
                    goal4.enhancement,
                    "EnhancementProbability",
                ),
                "EnhancementPattern": self._category(goal4.enhancement_pattern),
                "Necrosis": self._binary(goal4.necrosis, "NecrosisProbability"),
                "CysticChange": self._binary(
                    goal4.cystic_change,
                    "CysticChangeProbability",
                ),
                "Hemorrhage": self._binary(
                    goal4.hemorrhage,
                    "HemorrhageProbability",
                ),
                "Calcification": self._binary(
                    goal4.calcification,
                    "CalcificationProbability",
                ),
                "Margin": {
                    "clear": goal4.margin_clear.present,
                    "MarginClearProbability": goal4.margin_clear.probability,
                },
                "Lobulation": self._binary(
                    goal4.lobulation,
                    "LobulationProbability",
                ),
                "Signal_T2WI": self._category(goal4.signal_t2wi),
                "Signal_FLAIR": self._category(goal4.signal_flair),
            },
            "Interpretation": {"Conclusion": goal4.conclusion},
        }
        if goal4.attention_map_uri:
            prediction["Interpretation"]["AttentionMapURI"] = goal4.attention_map_uri
        return prediction

    @staticmethod
    def _mask_uri(series_uid: str) -> str:
        """掩码 URI（**单点维护**）：``./{SeriesUid}/{SeriesUid}.nii.gz``。

        规则来自上游 README「当前规范解释」，完整依据与三条推论见
        ``MASK_URI_TEMPLATE`` 上方的注释。原先那段「目录规范 vs 阳性示例不一致、
        待组委会确认」的疑问**已由官方契约关闭**，这里不再保留第二种猜测。

        ``series_uid`` 由 ``Goal5Task`` 给出，已按契约等于**磁盘上的序列目录名**
        （``data.series_selector.infer_uid_from_path`` 为磁盘优先）。
        """
        return MASK_URI_TEMPLATE.format(uid=series_uid)

    @staticmethod
    def _category(result: CategoricalResult) -> dict[str, object]:
        return {
            "predicted": result.predicted,
            "probabilities": result.probabilities,
        }

    @staticmethod
    def _binary(result: BinaryResult, probability_key: str) -> dict[str, object]:
        return {
            "present": result.present,
            probability_key: result.probability,
        }

