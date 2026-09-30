"""Goal4 辅助诊断：WHO 分级 + 位置/形态/边界 + 坏死囊变出血钙化 + 强化形态 + T2/FLAIR 信号。

规范 §9.5：插件输出**内部枚举**，比赛字符串由 ``output/competition_mapping.py`` 转换。
本文件因此只做三件事：取头 → 转换 → 组装 ``Goal4Result``，不写任何比赛字符串。

一个设计细节值得说明：``predicted`` 与 ``probabilities`` 必须**自洽**
（predicted 必须是概率最大的那一项），且 ``probabilities`` 之和必须为 1
（容差 1e-4）。这两条很容易在"模型输出与规则结论不一致"时被破坏，
因此统一由 ``postprocess.categorical`` 收口，而不是在各字段处各自处理。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.config import Settings
from output.competition_mapping import to_competition
from tasks._common.backbone_runner import BackboneRunner
from tasks.base import StudyTask
from tasks.goal4_diagnosis.config import Goal4Config
from tasks.goal4_diagnosis.labels import FIELD_ENUMS, enum_names
from tasks.goal4_diagnosis.postprocess import binary, categorical
from tasks.results import BinaryResult, CategoricalResult, Goal4Result

#: 各字段在比赛 JSON 中是否采用"专属概率字段名"。
#: 规范示例中二分类用 ``<Name>Probability``，而 Margin 用 ``MarginClearProbability``
#: （语义是"边界清晰"而非"存在边界"），这类差异必须在映射层显式记录。
BINARY_KEYS = ("Enhancement", "Necrosis", "CysticChange", "Hemorrhage",
               "Calcification", "Margin", "Lobulation")


class DiagnosisTask(StudyTask[Goal4Result]):
    """结构化诊断（14 个字段）。"""

    name = "goal4_diagnosis"

    def __init__(self, cfg: Goal4Config | None = None, settings: Settings | None = None,
                 device: str = "cuda") -> None:
        self.cfg = cfg or Goal4Config()
        self.settings = settings or Settings.from_env()
        self.device = device
        self._runner: BackboneRunner | None = None

    # ------------------------------------------------------------------ #
    def load_model(self) -> None:
        """服务启动时加载一次（规范 §5.2），并建立共享前向。

        走 ``BackboneRunner`` 是为了与其他 Goal **共享同一次骨干前向**：
        否则同一 Study 会被前向 4 次（实测导致 Mock Competition 超时）。
        """
        self._runner = BackboneRunner(
            ckpt_root=self.settings.ckpt_root, ckpt_rel=self.cfg.ckpt_rel,
            in_channels=self.cfg.in_channels, device=self.device,
            global_size=self.cfg.global_size, global_size_mm=self.cfg.global_size_mm,
            common_spacing=self.cfg.common_spacing, tta_flips=self.cfg.tta_flips,
            tta_batch=self.cfg.tta_batch,
        )
        self._runner.load()

    # ------------------------------------------------------------------ #
    def predict(self, context) -> Goal4Result:
        """单检查结构化诊断（复用共享前向）。"""
        if self._runner is None:
            self.load_model()
        assert self._runner is not None

        out = self._runner.forward(context.study, context)
        probs = self._collect(out)
        tumor_p = float(probs.get("TumorProbability", 0.5))
        result = self._assemble(probs, tumor_p)
        context.diagnostics["goal4"] = {
            "tumor_probability": tumor_p,
            "n_heads": len(out.get("cls") or []),
            "missing_channels": list(out.get("missing_channels") or []),
        }
        return result

    # ------------------------------------------------------------------ #
    def _collect(self, out: dict) -> dict[str, Any]:
        """把共享前向结果整理成 ``{字段: 概率}``。

        ``BackboneRunner`` 返回的已是概率（sigmoid/softmax 已施加），
        这里**不再二次转换**——重复 softmax 会让峰值概率被压平（如 0.85 → 0.53），
        进而使 predicted 与 probabilities 的关系失真。
        """
        probs: dict[str, Any] = {}
        rows = out.get("cls") or []
        spec = out.get("cls_spec") or []
        # 分类头数量不足时必须**显式失败**：旧实现是 ``continue`` 跳过，
        # 于是缺失字段静默落到 ``probs.get(field, 0.5)`` 的默认值上——
        # 表现为"某几个字段永远输出同一个结论"，既不报错也难以察觉。
        # 典型诱因是加载了"只训练单一路"的研发权重（如 goal1 的 model.pt）。
        if len(rows) < len(FIELD_ENUMS):
            have = [k for k, _ in spec] or "（权重未记录 cls_spec）"
            missing = [n for n, _ in FIELD_ENUMS[len(rows):]]
            raise ValueError(
                f"goal4: 权重的分类头不足（有 {len(rows)} 个，需要 {len(FIELD_ENUMS)} 个）；"
                f"缺失字段 {missing[:5]}{'…' if len(missing) > 5 else ''}；"
                f"权重声明的头 {have}。"
                f"请改用多任务训练产出的权重（含全部 14 个字段的头）。"
            )
        for i, (name, enum_cls) in enumerate(FIELD_ENUMS):
            arr = np.asarray(rows[i], dtype=np.float64).ravel()
            probs[name] = float(arr[0]) if enum_cls is None else arr.tolist()
        # 若权重里的头顺序与内部定义不一致，以权重声明的 spec 为准做一次校正
        if spec and [k for k, _ in spec] != [n for n, _ in FIELD_ENUMS]:
            by_key = {k: np.asarray(rows[i], dtype=np.float64).ravel()
                      for i, (k, _n) in enumerate(spec) if i < len(rows)}
            for name, enum_cls in FIELD_ENUMS:
                if name in by_key:
                    arr = by_key[name]
                    probs[name] = float(arr[0]) if enum_cls is None else arr.tolist()
        return probs

    def _assemble(self, probs: dict[str, Any], tumor_p: float) -> Goal4Result:
        """按 ``Goal4Result`` 的结构组装（内部枚举 → 比赛字符串在此完成）。"""
        def _cat(field: str, output_subset: list[str] | None = None) -> CategoricalResult:
            names = enum_names(field)
            row = categorical(names, probs.get(field), output_subset)
            # 内部枚举 → 官方字符串
            pred_cn = to_competition(field, row["predicted"]) if row["predicted"] else ""
            mapped = {to_competition(field, k): v for k, v in row["probabilities"].items()}
            return CategoricalResult(predicted=pred_cn, probabilities=mapped)

        def _bin(field: str) -> BinaryResult:
            p = float(probs.get(field, 0.5))
            return BinaryResult(present=p >= 0.5, probability=round(p, 4))

        negative = tumor_p < float(self.cfg.tumor_threshold)
        loc_row = probs.get("Location")
        location = "NA"
        if not negative and loc_row is not None:
            names = enum_names("Location")
            idx = int(np.argmax(loc_row))
            location = to_competition("Location", names[idx])

        grade_names = enum_names("WHO_Grade")
        if negative:
            who = CategoricalResult(predicted=None,
                                    probabilities={to_competition("WHO_Grade", n): 0.0
                                                   for n in grade_names})
        else:
            who = _cat("WHO_Grade")

        # Morphology：内部含 NA 兜底类，而规范示例仅 Regular/Irregular
        morph = _cat("Morphology", output_subset=["REGULAR", "IRREGULAR"])

        return Goal4Result(
            location=location,
            morphology=morph,
            who_grade=who,
            enhancement=_bin("Enhancement"),
            enhancement_pattern=_cat("EnhancementPattern"),
            necrosis=_bin("Necrosis"),
            cystic_change=_bin("CysticChange"),
            hemorrhage=_bin("Hemorrhage"),
            calcification=_bin("Calcification"),
            margin_clear=_bin("Margin"),
            lobulation=_bin("Lobulation"),
            signal_t2wi=_cat("Signal_T2WI"),
            signal_flair=_cat("Signal_FLAIR"),
            conclusion=self._conclusion(negative, location, morph, who),
        )

    @staticmethod
    def _conclusion(negative: bool, location: str, morph: CategoricalResult,
                    who: CategoricalResult) -> str:
        """生成自然语言结论（规范只要求是字符串，这里输出专业中文描述）。

        注意不能把英文枚举直接拼进中文（会得到"左侧颞叶Irregular占位"这类
        中英混杂的结论），因此这里做一层中文映射。
        """
        loc_cn = {
            "RightFrontal": "右侧额叶", "LeftFrontal": "左侧额叶",
            "RightTemporal": "右侧颞叶", "LeftTemporal": "左侧颞叶",
            "RightParietal": "右侧顶叶", "LeftParietal": "左侧顶叶",
            "RightOccipital": "右侧枕叶", "LeftOccipital": "左侧枕叶",
            "RightCerebellum": "右侧小脑半球", "LeftCerebellum": "左侧小脑半球",
            "RightBasalGanglia": "右侧基底节区", "LeftBasalGanglia": "左侧基底节区",
            "Brainstem": "脑干", "Other": "颅内其他部位", "NA": "颅内",
        }.get(location, location or "颅内")
        if negative:
            return f"{loc_cn}异常信号，考虑非肿瘤性病变（如脑梗死/脓肿），建议结合临床。"
        morph_cn = {"Regular": "形态规则", "Irregular": "形态不规则"}.get(
            str(morph.predicted), "")
        grade = who.predicted
        gtxt = f"（WHO {grade}级）" if grade else ""
        return f"{loc_cn}{morph_cn}占位，考虑脑胶质瘤{gtxt}。"


def build_task(settings: Settings | None = None, **kwargs: Any) -> DiagnosisTask:
    """工厂入口。"""
    return DiagnosisTask(settings=settings, **kwargs)
