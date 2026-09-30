"""Goal5 的 Task 适配器：比赛 Pipeline 的**唯一插件入口**（规范 §5.1）。

职责边界（严格对齐规范）：
- 只接收领域对象 ``PipelineContext``，返回强类型 ``Goal5Result``；
- **不**直接读取比赛根路径、**不**写 ``answer/``、**不**发回调；
- **不**导入任何训练专用模块（dataset/augmentations/losses/train/evaluate）；
- 完成推理与**逆变换**，把掩膜恢复到各自源序列空间——Writer 只负责落盘。

输出语义：``core_mask`` 对应 T1C 序列空间，``flair_mask`` 对应 FLAIR/T2 序列空间；
两者的 shape/affine 必须与来源序列完全一致，否则该例分割会被判 0 分。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.config import Settings
from data.series_selector import infer_uid_from_path, select as select_series
from tasks.base import StudyTask
from tasks.goal5.config import Goal5Config
from tasks.goal5.inference import LoadedGoal5, infer_segmentation, load_model
from tasks.goal5.postprocess import clean_pair
from tasks.goal5.preprocess import PreparedVolume, build_volume
from tasks.goal5.spatial import restore_binary_to_source, spacing_of
from tasks.results import Goal5Result

#: core / flair 的来源模态优先级（第一个命中者作为掩膜的参考空间）
CORE_SOURCE_MODALITIES = ("t1c", "t1")
FLAIR_SOURCE_MODALITIES = ("flair", "t2")


class Goal5Task(StudyTask[Goal5Result]):
    """T1 增强核心区 + FLAIR/T2 周围总异常区的二值分割。"""

    name = "goal5_segmentation"

    def __init__(self, cfg: Goal5Config | None = None, settings: Settings | None = None,
                 device: str = "cuda") -> None:
        self.cfg = cfg or Goal5Config()
        self.settings = settings or Settings.from_env()
        self.device = device
        self._loaded: LoadedGoal5 | None = None

    # ------------------------------------------------------------------ #
    def load_model(self) -> None:
        """服务启动时加载一次权重（禁止在逐 Study 推理里重复读盘）。"""
        self._loaded = load_model(self.cfg, self.settings.ckpt_root, self.device)

    # ------------------------------------------------------------------ #
    def predict(self, context) -> Goal5Result:
        """单个检查的分割推理。"""
        if self._loaded is None:
            self.load_model()
        assert self._loaded is not None

        study = context.study
        prepared: PreparedVolume = build_volume(study, self.cfg)
        if prepared.missing:
            context.warnings.append(
                f"goal5: 缺通道 {list(prepared.missing)}（已零占位）"
            )

        prob = infer_segmentation(self._loaded, prepared.volume, self.cfg)
        core_p, flair_p = prob[self.cfg.core_channel], prob[self.cfg.flair_channel]

        cfg_pp = self.cfg
        # 阈值来自权重（训练后标定），未标定时退化为配置默认值。
        class _T:                                                # noqa: D401 - 轻量视图
            default_thresholds = self._loaded.thresholds
            min_tumor_voxels = cfg_pp.min_tumor_voxels
            keep_components = cfg_pp.keep_components
            bridge_mm = cfg_pp.bridge_mm

        # ⚠️ 必须把**公共网格的实际 spacing** 传进去：``bridge_mm`` 是按 mm 给定的，
        # 而网格对层厚 > 1.5mm 的轴保留原始 spacing（不总是 1mm）。不传就会按 1mm 换算，
        # 在 3mm 层厚的网格上把 10mm 桥接半径放大成 30mm（见 postprocess.clean_pair）。
        core_bin, flair_bin = clean_pair(core_p, flair_p, _T(),
                                         spacing=spacing_of(prepared.affine))

        # ---- 诊断：把"掩膜为空"的成因拆开（概率不够 vs 后处理吃掉了）----
        # 这两种成因的修法完全不同，而只看 ``core_voxels=0`` 无法区分：
        #   · ``core_p.max() < thr`` → 模型输出本身就不够高（通道零占位 / 权重 / 预处理）
        #   · ``core_p.max() >= thr`` 但终值为 0 → **后处理**（阈值化后的连通域 / 最小体素 /
        #     形态学桥接）把它吃掉了，见 postprocess.clean_mask
        _thr = tuple(float(t) for t in self._loaded.thresholds)
        _core_pre = int(np.count_nonzero(core_p > _thr[0])) if _thr else 0
        _flair_pre = int(np.count_nonzero(flair_p > _thr[1])) if len(_thr) > 1 else 0

        # ---- 逆变换：恢复到各自源序列空间（shape/affine 必须与源图一致）----
        core_mask, core_uid, core_fb = self._restore(
            study, prepared, core_bin, CORE_SOURCE_MODALITIES, context.warnings)
        flair_mask, flair_uid, flair_fb = self._restore(
            study, prepared, flair_bin, FLAIR_SOURCE_MODALITIES, context.warnings)

        # Writer 的契约（output/writer.py）：**core 与 flair 落同一条序列时，两份掩膜必须完全相同**
        # （URI 是 `<uid>/<uid>.nii.gz`，同一路径装不下两份不同的掩膜）。
        # 正常路径不会撞——`guess_modality` 一个序列只映射一个模态，t1c/flair 必是不同 Series；
        # 只有"走了兜底"时两路才会同时退化到同一个参考序列。此时保留**可信的那一路**作为唯一掩膜，
        # 两路共用（既满足契约，又不丢正确信息；若两路都兜底则取并集，符合 core ⊆ peri 语义）。
        if core_uid == flair_uid and not np.array_equal(core_mask, flair_mask):
            if core_fb and not flair_fb:
                keep = flair_mask
            elif flair_fb and not core_fb:
                keep = core_mask
            else:
                keep = np.maximum(core_mask, flair_mask)
            context.warnings.append(
                f"goal5: core/flair 退化到同一序列 {core_uid} 且掩膜不同 → "
                f"两路统一为同一份掩膜（避免违反输出契约）"
            )
            core_mask = flair_mask = keep

        context.diagnostics["goal5"] = {
            "missing_channels": list(prepared.missing),
            "thresholds": [round(float(t), 3) for t in self._loaded.thresholds],
            # 最大概率：**低于阈值**说明模型输出就不够高（不是后处理的问题）
            "max_probs": [round(float(core_p.max()), 3), round(float(flair_p.max()), 3)],
            # 阈值以上的体素数（后处理**之前**）：>0 而终值 =0 → 是后处理吃掉的
            "core_pre_voxels": _core_pre,
            "flair_pre_voxels": _flair_pre,
            "core_voxels": int(core_mask.sum()),
            "flair_voxels": int(flair_mask.sum()),
            # 后处理参数（min_tumor_voxels / keep_components / bridge_mm）—— 空掩膜的嫌疑点
            "postprocess": {"min_tumor_voxels": cfg_pp.min_tumor_voxels,
                            "keep_components": cfg_pp.keep_components,
                            "bridge_mm": cfg_pp.bridge_mm},
            "ckpt": self._loaded.ckpt_path,
        }
        return Goal5Result(
            core_mask=core_mask,
            core_source_series_uid=core_uid,
            flair_mask=flair_mask,
            flair_source_series_uid=flair_uid,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _reference_series(study, prepared: PreparedVolume):
        """挑不出目标模态时的**退化目标**：公共网格的参考序列。

        一定要返回**真实存在于 `study.series` 里**的那一条 —— 掩膜只有落在真实序列的
        空间里，`OutputWriter` 才能查到它、`OutputValidator` 才会通过。
        """
        by_uid = {s.series_uid: s for s in study.series}
        for info in (prepared.channel_sources or {}).values():
            uid = str((info or {}).get("series_uid") or "").strip()
            if uid in by_uid:
                return by_uid[uid]
        return study.series[0] if study.series else None

    def _restore(self, study, prepared: PreparedVolume, mask: np.ndarray,
                 modalities: tuple[str, ...], warnings: list | None = None
                 ) -> tuple[np.ndarray, str, bool]:
        """把公共网格掩膜恢复到指定模态的源序列空间。

        返回 ``(掩膜, 目标序列 UID, 是否走了兜底)``。

        找不到目标模态时**退化为参考序列**（保证仍能写出、shape/affine 自洽）。

        ⚠️ 这里**绝不能返回空字符串**（旧实现的 bug）：空 UID 会让
        `OutputWriter._write_masks` 里的 `study.series_by_uid("")` 抛
        `KeyError: unknown series UID ''`，而 `core/runner.py` 的 `_run_streaming`
        对整批只包了一层 try（**没有 per-case 容错**）→ **一例失败 = 整批评测作废**。
        """
        picked = select_series(study, modalities)
        src = next(iter(picked.values()), None)
        fell_back = src is None
        if fell_back:
            src = self._reference_series(study, prepared)
        if src is None:                                   # 极端：study 一条序列都没有（构造时已拦）
            return mask.astype(np.uint8), "", True
        if fell_back and warnings is not None:
            warnings.append(
                f"goal5: 挑不出 {'/'.join(modalities)} 模态 → 掩膜退化写入参考序列 "
                f"{src.series_uid}（模态识别失败，答案仍合规但该例分割可能不准）"
            )

        restored = restore_binary_to_source(
            mask, prepared.affine, np.asarray(src.affine, dtype=np.float64),
            tuple(int(x) for x in src.image.shape),
        )
        return restored.astype(np.uint8), infer_uid_from_path(src), fell_back


def build_task(settings: Settings | None = None, **kwargs: Any) -> Goal5Task:
    """工厂入口，供 ``core/registry`` 或团队 Pipeline 构建使用。"""
    return Goal5Task(settings=settings, **kwargs)


#: 官方占位模板的类名兼容（原 ``tasks/goal5/task.py`` 导出 ``TorchGoal5Task``）。
#: 本实现的类名本就是 ``Goal5Task``，与官方占位一致。
TorchGoal5Task = Goal5Task
__all__ = ["Goal5Task", "TorchGoal5Task", "build_task"]
