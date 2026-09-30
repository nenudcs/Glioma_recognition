from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.callback import CompetitionCallback
from core.config import Settings
from core.registry import build_pipeline
from data.loader import DatasetLoader, tolerant_mode
from observability.competition_logger import CompetitionLogger
from output.validator import OutputValidator
from output.writer import OutputWriter
from pipeline.inference import InferencePipeline

#: 是否逐例打印进度（``GLIOMA_PROGRESS=0`` 可关）。
#:
#: 为什么默认**开**：一次 700+ 例的推理要跑很久，而循环里原来**每例之间不打任何日志** ——
#: 于是「进程卡死」与「某例正在跑分钟级滑窗推理」在日志上**完全一样，无法区分**
#: （实测踩过：日志停在某例之后不动，分不清要不要 Ctrl-C）。
#: 每例两行（开始 / 完成 + 耗时 + goal5 摘要）就够分辨，还能直接估出剩余时长。
_PROGRESS = os.environ.get("GLIOMA_PROGRESS", "").strip().lower() not in {
    "0", "false", "no", "off",
}


def _diag_brief(context: object) -> str:
    """一行 goal5 摘要 —— 目标是**能直接定位"掩膜为空"的成因**。

    格式：``goal5{missing=[…], thr=[…], pmax=[…], core=终值/阈值以上, flair=…}``

    三种空掩膜的成因**修法完全不同**，所以必须分开看：

    * ``missing`` 非空 → 该通道是**零占位**的（数据本身缺这个模态）；
    * ``pmax`` **低于** ``thr`` → 模型输出就不够高（通道零占位 / 权重 / 预处理问题）；
    * ``pmax`` 够高，但 ``阈值以上 > 0`` 而 ``终值 = 0`` → **后处理吃掉了**
      （见 ``tasks/goal5_segmentation/postprocess.py``：连通域保留 / 最小体素 / 形态学桥接）。
    """
    goal5 = ((getattr(context, "diagnostics", None) or {}).get("goal5")) or {}
    if not goal5:
        return ""
    return (f"goal5{{missing={goal5.get('missing_channels')}, "
            f"thr={goal5.get('thresholds')}, "
            f"pmax={goal5.get('max_probs')}, "
            f"core={goal5.get('core_voxels')}/{goal5.get('core_pre_voxels')}, "
            f"flair={goal5.get('flair_voxels')}/{goal5.get('flair_pre_voxels')}}}")


@dataclass(frozen=True)
class EvaluationJob:
    request_id: str
    evaluation_id: str
    dataset_path: Path


class EvaluationRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        loader: DatasetLoader | None = None,
        pipeline: InferencePipeline | None = None,
        validator: OutputValidator | None = None,
        callback: CompetitionCallback | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or DatasetLoader()
        self.pipeline = pipeline or build_pipeline(settings.pipeline_factory)
        self.writer = OutputWriter(settings.answer_root)
        self.validator = validator or OutputValidator()
        self.callback = callback or CompetitionCallback(
            settings.callback_url,
            settings.callback_timeout_seconds,
            settings.callback_attempts,
        )
        self.logger = CompetitionLogger(settings.log_root)
        self._run_lock = threading.Lock()

    def run(self, job: EvaluationJob, *, send_callback: bool = True) -> Path:
        # DatasetTask has one mutable incremental state, so evaluations sharing
        # this runner must not interleave even if the job executor has workers.
        with self._run_lock:
            return self._run_streaming(job, send_callback=send_callback)

    def _run_streaming(
        self,
        job: EvaluationJob,
        *,
        send_callback: bool,
    ) -> Path:
        started = time.perf_counter()
        data_source = str(job.dataset_path)
        self.logger.write(
            request_id=job.request_id,
            evaluation_id=job.evaluation_id,
            phase="test",
            message="evaluation_started",
            data_source=data_source,
        )
        staging: Path | None = None
        current_accession: str | None = None
        skipped: list[tuple[str, str]] = []
        try:
            staging = self.writer.begin(job.evaluation_id)
            accessions: set[str] = set()
            self.pipeline.reset_dataset_task()
            processed = 0
            for study in self.loader.iter_studies(job.dataset_path):
                current_accession = study.accession_number
                if current_accession in accessions:
                    raise ValueError(
                        f"dataset has duplicate accession number: {current_accession}"
                    )
                processed += 1
                study_started = time.perf_counter()
                if _PROGRESS:
                    # 「开始」与「完成」成对出现 → 卡死与"某例很慢"立刻可分辨
                    print(f"[runner] #{processed} {current_accession} 开始 …", flush=True)
                try:
                    context = self.pipeline.run_study(study)
                    self.pipeline.update_dataset_task(study, context)
                    accession_dir = self.writer.write_study(staging, context)
                    self.validator.validate_study(accession_dir, study)
                except Exception as exc:                          # noqa: BLE001
                    # ---- 逐例容错（**仅 GLIOMA_LOADER_TOLERANT=1 时生效**）----
                    # 评测**不可重跑**：1 例脏数据 / 1 次推理异常，不该让其余几百例
                    # 一起作废。默认（容错关闭）保持严格 fail-fast：原样抛出。
                    if not tolerant_mode():
                        raise
                    skipped.append(
                        (current_accession, f"{type(exc).__name__}: {exc}")
                    )
                    print(
                        f"[runner][容错] 跳过检查 {current_accession!r}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    self.logger.write(
                        request_id=job.request_id,
                        evaluation_id=job.evaluation_id,
                        phase="test",
                        message="study_skipped",
                        data_source=data_source,
                        accession_number=current_accession,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    # 清掉该例可能已写了一半的目录：否则 validate_final_layout
                    # 会因为"多出一个目录"而整批失败（它要求 staging 的子目录
                    # 与发布集合**完全相等**）。
                    shutil.rmtree(staging / current_accession, ignore_errors=True)
                    current_accession = None
                    del study
                    continue
                if _PROGRESS:
                    elapsed_ms = round((time.perf_counter() - study_started) * 1000)
                    print(f"[runner] #{processed} {current_accession} 完成 "
                          f"{elapsed_ms}ms {_diag_brief(context)}".rstrip(), flush=True)
                accessions.add(current_accession)
                del context, study
                current_accession = None

            if skipped:
                print(
                    f"[runner][容错] 跳过 {len(skipped)} 例，"
                    f"正常发布 {len(accessions)} 例："
                    f"{[item[0] for item in skipped[:10]]}"
                    f"{' …' if len(skipped) > 10 else ''}",
                    flush=True,
                )
                self.logger.write(
                    request_id=job.request_id,
                    evaluation_id=job.evaluation_id,
                    phase="test",
                    message="studies_skipped",
                    data_source=data_source,
                    skipped_count=len(skipped),
                    written_count=len(accessions),
                    skipped=[item[0] for item in skipped[:50]],
                )
            if not accessions:
                raise ValueError(
                    "容错模式下没有任何一例成功产出，无可发布的答案"
                    f"（全部 {len(skipped)} 例失败）"
                )

            duplicates = self.pipeline.finalize_dataset_task()
            duplicate_path = self.writer.write_duplicates(
                staging,
                accessions,
                duplicates,
            )
            self.validator.validate_duplicates(duplicate_path, accessions)
            self.validator.validate_final_layout(staging, accessions)
            output_dir = self.writer.publish(staging, job.evaluation_id)
            staging = None
            duration_ms = round((time.perf_counter() - started) * 1000)
            self.logger.write(
                request_id=job.request_id,
                evaluation_id=job.evaluation_id,
                phase="test",
                message="evaluation_completed",
                data_source=data_source,
                duration_ms=duration_ms,
                study_count=len(accessions),
                pred_path=str(output_dir),
            )
            if send_callback:
                self.callback.send_success(
                    job.request_id,
                    job.evaluation_id,
                    output_dir,
                )
            return output_dir
        except Exception as exc:
            self.logger.write(
                request_id=job.request_id,
                evaluation_id=job.evaluation_id,
                phase="test",
                message="evaluation_failed",
                data_source=data_source,
                error_type=type(exc).__name__,
                error=str(exc),
                accession_number=current_accession,
            )
            if staging and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            raise
