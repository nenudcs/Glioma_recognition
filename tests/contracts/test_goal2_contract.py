"""Goal2（拼接 + 重复）契约测试（规范 §19.2）。

逐条检查规范为插件规定的契约：

1. 注册入口 ``tasks.real_pipeline.build_pipeline`` 把 Goal2 绑定在正确位置，
   数据集级重复任务由 ``Goal2DuplicateRecorder`` 承担；
2. 比赛运行链路不传递导入训练/标定模块（``dataset`` / ``evaluate``）；
3. 接收最小 ``PipelineContext``、返回强类型 ``StitchedResult``、概率合法；
4. 不产生比赛目录副作用（不写 answer/prediction.json/duplicate_pairs.jsonl）；
5. 单条序列失败可降级、整例失败不终止 evaluation；
6. 轻微调整（加噪）的重复影像也能命中，并正确对下游任务上闸；
7. ``DatasetTask`` 的 reset/update/finalize 满足 Goal2 duplicate 额外 DoD
   （无自配对、无重复对、每例 Top-200、reset 不泄漏）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.structures import Series, Study  # noqa: E402
from pipeline.context import PipelineContext  # noqa: E402
from pipeline.inference import InferencePipeline, StudyTaskBinding  # noqa: E402
from tasks.goal2_duplicate.config import Goal2DuplicateConfig  # noqa: E402
from tasks.goal2_duplicate.inference import DuplicateProbe  # noqa: E402
from tasks.goal2_duplicate.task import Goal2DuplicateRecorder  # noqa: E402
from tasks.goal2_stitched.config import Goal2StitchedConfig  # noqa: E402
from tasks.goal2_stitched.gating import (  # noqa: E402
    GATE_KEY,
    SKIP_KEY,
    GatedStudyTask,
    gated_fields,
)
from tasks.goal2_stitched.task import Goal2StitchedTask  # noqa: E402
from tasks.results import StitchedResult  # noqa: E402

FORBIDDEN_MODULES = (
    "tasks.goal2_stitched.dataset",
    "tasks.goal2_stitched.evaluate",
    "tasks.goal2_duplicate.dataset",
    "tasks.goal2_duplicate.evaluate",
)


# --------------------------------------------------------------------------
# 合成体数据（有纹理，避免不同病例之间过于相似）
# --------------------------------------------------------------------------
def _box_blur(array: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return array
    result = np.asarray(array, dtype=np.float32)
    window = 2 * radius + 1
    for axis in range(result.ndim):
        length = result.shape[axis]
        pad = [(0, 0)] * result.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(result, pad, mode="edge")
        cumulative = np.cumsum(padded, axis=axis)
        zeros_shape = list(cumulative.shape)
        zeros_shape[axis] = 1
        cumulative = np.concatenate(
            [np.zeros(zeros_shape, dtype=cumulative.dtype), cumulative], axis=axis
        )
        high = [slice(None)] * result.ndim
        high[axis] = slice(window, length + window)
        low = [slice(None)] * result.ndim
        low[axis] = slice(0, length)
        result = (cumulative[tuple(high)] - cumulative[tuple(low)]) / window
    return result.astype(np.float32)


def textured_volume(seed: int, shape: tuple[int, int, int] = (24, 64, 64)) -> np.ndarray:
    """层间**平滑**、层内**有纹理**的体数据（模拟真实 MRI：相邻层高度相关）。

    层内纹理由 seed 决定 → 不同病例的中间层细节不同，可用来评估重复检测；
    层间只做平滑的亮度变化 → 正常体数据的「层间突变」分数应当很低，
    这样拼接检测的阈值才有意义。
    """
    rng = np.random.default_rng(seed)
    plane = np.zeros(shape[1:], dtype=np.float32)
    for radius, weight in ((5, 1.0), (2, 0.6), (1, 0.30)):
        plane += weight * _box_blur(rng.normal(0.0, 1.0, shape[1:]).astype(np.float32), radius)

    ys = np.linspace(-1.0, 1.0, shape[1])
    xs = np.linspace(-1.0, 1.0, shape[2])
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    radius_map = np.sqrt(grid_y ** 2 + grid_x ** 2)
    plane += 0.8 * np.exp(-((radius_map - 0.85) ** 2) / 0.002)
    envelope = np.exp(-((radius_map / 0.95) ** 4))

    normalized = (plane - plane.min()) / max(float(np.ptp(plane)), 1e-6)
    profile = 0.45 + 0.55 * np.sin(np.linspace(0.35, 2.7, shape[0]))
    volume = (
        1000.0
        * normalized[None, :, :]
        * envelope[None, :, :]
        * profile[:, None, None]
    )
    volume += rng.normal(0.0, 5.0, shape).astype(np.float32)      # 很轻的层内噪声
    return np.ascontiguousarray(np.clip(volume, 0.0, None), dtype=np.float32)


def add_noise(volume: np.ndarray, sigma: float = 6.0, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (volume + rng.normal(0.0, sigma, volume.shape)).astype(np.float32)


def spliced_volume(seed: int, cut: int = 12, factor: float = 0.45) -> np.ndarray:
    """两个不同体数据各取一段拼在一起：拼接缝在 ``cut`` 层。"""
    top = textured_volume(seed)[:cut]
    bottom = textured_volume(seed + 100)[cut:] * factor + 25.0
    return np.ascontiguousarray(
        np.concatenate([top, bottom], axis=0).astype(np.float32)
    )


def write_series(root: Path, accession: str, series_uid: str, volume: np.ndarray) -> Path:
    import nibabel as nib

    directory = root / accession / series_uid
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{series_uid}.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(path))
    return path


def _context(volume: np.ndarray | None = None, accession: str = "ACC001") -> PipelineContext:
    image = np.zeros((8, 16, 16), dtype=np.float32) if volume is None else volume
    series = Series(
        series_uid="T1CE",
        modality="T1CE (增强)",
        image=image,
        affine=np.eye(4),
        source_path=Path("/tmp/T1CE.nii.gz"),
    )
    return PipelineContext(study=Study(accession, (series,)))


class Goal2RegistryContractTest(unittest.TestCase):
    """注册入口与配置契约。"""

    def test_real_pipeline_binds_goal2_tasks(self) -> None:
        from tasks.real_pipeline import build_pipeline

        pipeline = build_pipeline()
        self.assertIsInstance(pipeline, InferencePipeline)
        fields = [binding.context_field for binding in pipeline.study_tasks]
        self.assertEqual(["goal1", "goal2_stitched", "goal3", "goal5", "goal4"], fields)
        self.assertIsInstance(
            pipeline.study_tasks[1].task,
            Goal2StitchedTask,
            "tasks/real_pipeline.py 仍是旧版（绑定的是 DummyStitchedTask）。"
            "增量包的应用顺序必须是 goal1 → goal2：goal2 包里的 tasks/real_pipeline.py "
            "同时包含 Goal1 与 Goal2 的接线，必须最后覆盖；或直接解压 goal12_upload.zip。",
        )
        self.assertIsInstance(
            pipeline.duplicate_task,
            Goal2DuplicateRecorder,
            "tasks/real_pipeline.py 仍是旧版（数据集级任务还是 DummyDuplicateTask）："
            "请用 Goal2 增量包里的版本覆盖，或解压 goal12_upload.zip。",
        )
        # 重复检测的逐例探针必须与数据集级记录器共用同一实例
        self.assertIs(pipeline.study_tasks[1].task.duplicate, pipeline.duplicate_task.probe)

    def test_runtime_chain_avoids_training_modules(self) -> None:
        """在独立解释器里加载注册入口，确认训练/标定模块没有被传递导入。"""
        script = (
            "import sys;"
            "import tasks.real_pipeline as rp;"
            "rp.build_pipeline();"
            "print([m for m in sys.modules if 'goal2' in m])"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["GOAL1_CHECKPOINT"] = "/nonexistent/model.pt"      # 避免真加载权重
        env["GOAL1_STRICT"] = "0"
        env["GOAL2_DUPLICATE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(0, result.returncode, result.stderr[-2000:])
        loaded = result.stdout.strip().splitlines()[-1]
        for forbidden in FORBIDDEN_MODULES:
            self.assertNotIn(forbidden, loaded)

    def test_config_resolves_from_environment(self) -> None:
        stitched = Goal2StitchedConfig.from_env(
            {"GOAL2_STITCHED_THRESHOLD": "0.05", "GOAL2_STITCHED_METRIC": "local"}
        )
        self.assertEqual(0.05, stitched.threshold)
        self.assertEqual("local", stitched.metric)
        self.assertTrue(stitched.gate)
        with self.assertRaises(ValueError):
            Goal2StitchedConfig(threshold=0.0)

        duplicate = Goal2DuplicateConfig.from_env(
            {
                "GOAL2_DUPLICATE_SIM_CENTER": "0.93",
                "GOAL2_DUPLICATE_GATE_PROB": "0.8",
                "GOAL2_DUPLICATE_TOP_K": "5",
            }
        )
        self.assertEqual(0.93, duplicate.center)
        self.assertEqual(0.8, duplicate.gate_probability)
        self.assertEqual(5, duplicate.top_k)
        self.assertGreater(duplicate.gate_similarity(), duplicate.center)
        with self.assertRaises(ValueError):
            Goal2DuplicateConfig(mode="unknown")

    def test_gated_fields_parsing(self) -> None:
        self.assertEqual(("goal3", "goal4", "goal5"), gated_fields({}))
        self.assertEqual((), gated_fields({"GOAL2_GATED_FIELDS": ""}))
        self.assertEqual(("goal1",), gated_fields({"GOAL2_GATED_FIELDS": "goal1"}))
        with self.assertRaises(ValueError):
            gated_fields({"GOAL2_GATED_FIELDS": "goal2"})


class StitchedBehaviourContractTest(unittest.TestCase):
    """拼接检测行为契约。"""

    def test_predict_returns_stitched_result_with_valid_probability(self) -> None:
        task = Goal2StitchedTask(Goal2StitchedConfig(threshold=0.02))
        task.load_model()
        context = _context(textured_volume(1))

        cwd_before = {path.name for path in Path.cwd().iterdir()}
        result = task.predict(context)
        cwd_after = {path.name for path in Path.cwd().iterdir()}

        self.assertIsInstance(result, StitchedResult)
        self.assertTrue(np.isfinite(result.stitched_probability))
        self.assertGreaterEqual(result.stitched_probability, 0.0)
        self.assertLessEqual(result.stitched_probability, 1.0)
        self.assertEqual(cwd_before, cwd_after)
        self.assertFalse((Path.cwd() / "answer").exists())
        detail = context.diagnostics["goal2"]
        for key in ("task_name", "model_version", "series", "threshold", "duration_ms"):
            self.assertIn(key, detail)

    def test_spliced_volume_is_flagged_and_gated(self) -> None:
        # 阈值取在「正常体数据（≈0.02~0.04）」与「人工拼缝（≈0.32）」之间
        config = Goal2StitchedConfig(threshold=0.1, gate=True)
        task = Goal2StitchedTask(config)
        normal = task.predict(_context(textured_volume(2), "ACC_NORMAL"))
        self.assertLess(normal.stitched_probability, 0.5)

        context = _context(spliced_volume(3), "ACC_SPLICED")
        result = task.predict(context)
        self.assertGreater(result.stitched_probability, 0.5)
        self.assertEqual("stitched", context.diagnostics[GATE_KEY]["reason"])

    def test_gate_skips_downstream_task(self) -> None:
        from tasks.base import StudyTask
        from tasks.results import Goal3Result

        class Counter(StudyTask[Goal3Result]):
            name = "goal3"

            def __init__(self) -> None:
                self.calls = 0

            def predict(self, context) -> Goal3Result:
                self.calls += 1
                return Goal3Result(tumor_probability=0.7)

            def neutral(self) -> Goal3Result:
                return Goal3Result(tumor_probability=0.5)

        counter = Counter()
        gated = GatedStudyTask(
            counter,
            "goal3",
            lambda context: counter.neutral(),
            gated_fields=("goal3",),
        )
        context = _context()
        self.assertEqual(0.7, gated.predict(context).tumor_probability)
        context.diagnostics[GATE_KEY] = {"reason": "duplicate"}
        self.assertEqual(0.5, gated.predict(context).tumor_probability)
        self.assertEqual(1, counter.calls)             # 闸门生效，未调用内层模型
        self.assertEqual("duplicate", context.diagnostics[SKIP_KEY]["goal3"])

    def test_single_bad_series_is_degradable(self) -> None:
        from tasks.goal2_stitched.inference import StitchedInference
        from tasks.goal2_stitched.postprocess import SeriesScore

        task = Goal2StitchedTask(Goal2StitchedConfig())

        class _Stub:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, image, series_uid="", series_type=""):
                self.calls += 1
                if self.calls == 1:
                    return SeriesScore(
                        series_uid=str(series_uid),
                        score=None,
                        worst_slice=None,
                        slices=0,
                        series_type=str(series_type),
                        error="stub failure",
                    )
                return StitchedInference(task.config).score_series(image, series_uid, series_type)

        stub = _Stub()
        task.inference.score_series = stub                      # 第一条序列失败
        series_a = Series("A", "T1CE", textured_volume(4), np.eye(4), Path("/tmp/A.nii.gz"))
        series_b = Series("B", "FLAIR", textured_volume(5), np.eye(4), Path("/tmp/B.nii.gz"))
        context = PipelineContext(study=Study("ACC002", (series_a, series_b)))

        result = task.predict(context)
        self.assertTrue(np.isfinite(result.stitched_probability))
        self.assertEqual(1, len(context.warnings))
        self.assertEqual(1, context.diagnostics["goal2"]["scored_series"])

    def test_disabled_task_reports_zero_and_no_gate(self) -> None:
        task = Goal2StitchedTask(Goal2StitchedConfig(enabled=False, threshold=0.02))
        context = _context(spliced_volume(6))
        result = task.predict(context)
        self.assertEqual(0.0, result.stitched_probability)
        self.assertNotIn(GATE_KEY, context.diagnostics)


class DuplicateBehaviourContractTest(unittest.TestCase):
    """重复检测：DatasetTask 生命周期、近似重复命中、闸门与稀疏性。"""

    def _pipeline(self, root: Path, *, counter=None, config: Goal2DuplicateConfig | None = None):
        duplicate_config = config or Goal2DuplicateConfig(top_k=5)
        probe = DuplicateProbe(duplicate_config)
        stitched = Goal2StitchedTask(
            # 本组只验证重复检测，把拼接阈值设高以避免合成数据里的层间噪声干扰
            Goal2StitchedConfig(threshold=0.5),
            duplicate_probe=probe,
        )
        tasks = [StudyTaskBinding("goal2_stitched", stitched)]
        if counter is not None:
            tasks.append(
                StudyTaskBinding(
                    "goal3",
                    GatedStudyTask(
                        counter,
                        "goal3",
                        lambda context: counter.neutral(),
                        gated_fields=("goal3",),
                    ),
                )
            )
        return InferencePipeline(
            study_tasks=tuple(tasks),
            duplicate_task=Goal2DuplicateRecorder(duplicate_config, probe=probe),
        )

    def test_near_duplicate_is_gated_and_reported(self) -> None:
        from data.loader import DatasetLoader
        from tasks.base import StudyTask
        from tasks.results import Goal3Result

        class Counter(StudyTask[Goal3Result]):
            name = "goal3"

            def __init__(self) -> None:
                self.calls = 0

            def predict(self, context) -> Goal3Result:
                self.calls += 1
                return Goal3Result(tumor_probability=0.5)

            def neutral(self) -> Goal3Result:
                return Goal3Result(tumor_probability=0.5)

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary) / "dataset"
            base = textured_volume(41)
            write_series(root, "CASE_A", "T1", base)
            write_series(root, "CASE_A_COPY", "T1", add_noise(base, 6.0, seed=9))
            write_series(root, "CASE_B", "T1", textured_volume(42))

            counter = Counter()
            contexts, duplicates = self._pipeline(root, counter=counter).run(
                DatasetLoader().load(root)
            )

            flagged = contexts["CASE_A_COPY"]
            self.assertEqual("duplicate", flagged.diagnostics[GATE_KEY]["reason"])
            self.assertIn("CASE_A", flagged.diagnostics[GATE_KEY]["matches"])
            self.assertEqual("duplicate", flagged.diagnostics[SKIP_KEY]["goal3"])
            self.assertNotIn(GATE_KEY, contexts["CASE_B"].diagnostics)
            self.assertEqual(1, counter.calls)          # 只有 CASE_B 执行了下游任务

            pairs = {
                tuple(sorted((pair.left_accession, pair.right_accession))): pair.probability
                for pair in duplicates.pairs
            }
            self.assertGreaterEqual(pairs[("CASE_A", "CASE_A_COPY")], 0.5)
            self.assertEqual(
                max(pairs.values()), pairs[("CASE_A", "CASE_A_COPY")]
            )

    def test_exact_copy_probability_is_one(self) -> None:
        from data.loader import DatasetLoader

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary) / "dataset"
            base = textured_volume(51)
            write_series(root, "ACC001", "T1", base)
            write_series(root, "ACC002", "T1", base.copy())
            write_series(root, "ACC003", "T1", textured_volume(52))

            contexts, duplicates = self._pipeline(root).run(DatasetLoader().load(root))
            best = max(duplicates.pairs, key=lambda item: item.probability)
            self.assertEqual(1.0, best.probability)
            self.assertEqual({"ACC001", "ACC002"}, {best.left_accession, best.right_accession})
            self.assertTrue(contexts["ACC002"].diagnostics[GATE_KEY]["reason"] == "duplicate")

    def test_pairs_are_normalized_and_bounded(self) -> None:
        from tasks.results import DuplicatePair

        recorder = Goal2DuplicateRecorder(
            Goal2DuplicateConfig(max_pairs_per_study=2),
            probe=None,
            max_pairs_per_study=2,
        )
        recorder.reset()
        for index in range(4):
            study = Study(
                f"ACC{index:03d}",
                (Series("T1", "T1", np.zeros((4, 4, 4), np.float32), np.eye(4), Path("/tmp/T1.nii.gz")),),
            )
            context = PipelineContext(study=study)
            context.diagnostics["goal2"] = {
                "duplicate": {
                    "matches": [
                        {"accession": f"ACC{other:03d}", "probability": 0.9 - 0.1 * other}
                        for other in range(4)
                        if other != index
                    ]
                }
            }
            recorder.update(study, context)
        result = recorder.finalize()

        seen: set[tuple[str, str]] = set()
        counts: dict[str, int] = {}
        for pair in result.pairs:
            key = tuple(sorted((pair.left_accession, pair.right_accession)))
            self.assertNotEqual(key[0], key[1])
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertGreaterEqual(pair.probability, 0.0)
            self.assertLessEqual(pair.probability, 1.0)
            counts[key[0]] = counts.get(key[0], 0) + 1
            counts[key[1]] = counts.get(key[1], 0) + 1
        self.assertTrue(all(value <= 2 for value in counts.values()))

    def test_reset_clears_probe_state(self) -> None:
        probe = DuplicateProbe(Goal2DuplicateConfig(scan=False))
        recorder = Goal2DuplicateRecorder(Goal2DuplicateConfig(scan=False), probe=probe)
        probe.index = object()                    # 模拟上一轮留下的索引
        recorder.reset()
        self.assertIsNone(probe.index)
        self.assertEqual({}, dict(recorder._pairs))


class Goal2CalibrationContractTest(unittest.TestCase):
    """标定产物走规范路径（与 Goal1 的 checkpoint 约定同构）与离线打分入口。"""

    def test_spec_paths_and_round_trip(self) -> None:
        from tasks.goal2_duplicate import checkpoint as duplicate_checkpoint
        from tasks.goal2_stitched import checkpoint as stitched_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            self.assertEqual(
                Path("goal2_stitched") / "calibration.json",
                stitched_checkpoint.relative_calibration_path(),
            )
            self.assertEqual(
                root / "goal2_stitched" / "calibration.json",
                stitched_checkpoint.calibration_path(root),
            )
            self.assertEqual(
                root / "goal2_duplicate" / "calibration.json",
                duplicate_checkpoint.calibration_path(root),
            )

            stitched_path = stitched_checkpoint.write_calibration(
                {"threshold": 0.031, "metric": "local", "statistic": "p99", "band": 0.2},
                root,
            )
            self.assertTrue(stitched_path.is_file())
            config = Goal2StitchedConfig.from_env({}, checkpoint_root=root)
            self.assertEqual(0.031, config.threshold)
            self.assertEqual("local", config.metric)
            self.assertEqual("p99", config.statistic)
            self.assertEqual(stitched_path, config.calibration_file)
            # 环境变量优先于标定文件
            override = Goal2StitchedConfig.from_env(
                {"GOAL2_STITCHED_THRESHOLD": "0.5"}, checkpoint_root=root
            )
            self.assertEqual(0.5, override.threshold)

            duplicate_path = duplicate_checkpoint.write_calibration(
                {"center": 0.93, "scale": 0.01, "min_sim": 0.6, "top_k": 8},
                root,
            )
            self.assertTrue(duplicate_path.is_file())
            duplicate = Goal2DuplicateConfig.from_env({}, checkpoint_root=root)
            self.assertEqual(0.93, duplicate.center)
            self.assertEqual(0.6, duplicate.min_sim)
            self.assertEqual(8, duplicate.top_k)
            self.assertEqual(duplicate_path, duplicate.calibration_file)

    def test_broken_calibration_file_falls_back_to_defaults(self) -> None:
        from tasks.goal2_stitched import checkpoint as stitched_checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "checkpoint"
            path = stitched_checkpoint.calibration_path(root)
            path.parent.mkdir(parents=True)
            path.write_text("{ not json", encoding="utf-8")
            config = Goal2StitchedConfig.from_env({}, checkpoint_root=root)
            self.assertEqual(Goal2StitchedConfig().threshold, config.threshold)
            self.assertIsNone(config.calibration_file)

    def test_stitched_offline_scores_dataset(self) -> None:
        from tasks.goal2_stitched.evaluate import score_dataset

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            write_series(dataset, "CASE_A", "T1", textured_volume(61))
            write_series(dataset, "CASE_B", "T1", spliced_volume(62))
            out = root / "out"

            payload = score_dataset(
                dataset,
                threshold=0.1,
                workers=1,
                out_dir=out,
            )
            self.assertEqual(2, payload["files"])
            self.assertEqual(2, payload["scored"])
            self.assertEqual(1, payload["flagged"])          # 只有拼接例被标记
            scores_file = out / "stitched_scores.jsonl"
            self.assertTrue(scores_file.is_file())
            records = [
                json.loads(line)
                for line in scores_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(2, len(records))
            for record in records:
                self.assertIn("score", record)
                self.assertGreaterEqual(record["probability"], 0.0)
                self.assertLessEqual(record["probability"], 1.0)

    def test_duplicate_offline_pairs_dataset(self) -> None:
        from tasks.goal2_duplicate.config import Goal2DuplicateConfig
        from tasks.goal2_duplicate.evaluate import score_dataset

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            base = textured_volume(71)
            write_series(dataset, "CASE_A", "T1", base)
            write_series(dataset, "CASE_A_COPY", "T1", add_noise(base, 6.0, seed=13))
            write_series(dataset, "CASE_B", "T1", textured_volume(72))
            out = root / "out"

            payload = score_dataset(
                dataset,
                config=Goal2DuplicateConfig(top_k=3),
                workers=1,
                out_dir=out,
            )
            self.assertEqual(3, payload["studies"])
            pairs_file = out / "duplicate_pairs_offline.jsonl"
            self.assertTrue(pairs_file.is_file())
            lines = [
                json.loads(line)
                for line in pairs_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(lines)
            best = max(lines, key=lambda item: item["PairProb"])
            self.assertEqual({"CASE_A", "CASE_A_COPY"},
                             {best["StudyUID"], best["StudyUID_dup"]})
            self.assertGreaterEqual(best["PairProb"], 0.5)
            for line in lines:
                self.assertEqual({"StudyUID", "StudyUID_dup", "PairProb"}, set(line))
                self.assertNotEqual(line["StudyUID"], line["StudyUID_dup"])

    def test_make_subset_builds_test_layout(self) -> None:
        from tasks.goal2_stitched.dataset import make_subset

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            training = root / "training"
            base = textured_volume(81)
            write_series(training, "annotation/duplicate/DUP001", "T1", base)
            write_series(training, "annotation/duplicate/DUP002", "T1", add_noise(base, 6.0, 14))
            write_series(training, "annotation/Composition/CMP001", "T1", spliced_volume(82))
            write_series(training, "normal01", "T1", textured_volume(83))
            (training / "annotation" / "duplicate" / "gold.csv").write_text(
                "src_img,desc_img\nDUP001,DUP002\n",
                encoding="utf-8",
            )

            subset = make_subset(training, count=3, target_root=root / "subset")
            studies = sorted(path.name for path in subset.iterdir() if path.is_dir())
            self.assertTrue(studies)
            for accession in studies:
                for series_dir in (subset / accession).iterdir():
                    self.assertTrue(
                        (series_dir / f"{series_dir.name}.nii.gz").is_file()
                        or (series_dir / f"{series_dir.name}.nii").is_file(),
                        f"{accession}/{series_dir.name} 缺少同名原文件",
                    )
            # 切出来的样本必须能被管线 Loader 正常读入
            from data.loader import DatasetLoader

            dataset = DatasetLoader().load(subset)
            self.assertEqual(len(studies), len(dataset.studies))

    def test_layout_inspection_handles_renamed_special_dir(self) -> None:
        """标注目录被改名（没有 composition 前缀）时，体检要能指出真实目录名与用法。"""
        from tasks.goal2_stitched.dataset import (
            candidate_kind_dirs,
            inspect_layout,
            labeled_volumes,
        )

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            root = Path(temporary)
            training = root / "training"
            # 正常病例：annotation/<32位hex>/<sop>.nii.gz（与赛方训练集一致）
            write_series(training, "annotation/0050d79429cf4d86907dc8c4a34cbf04", "2.25.2574", textured_volume(91))
            write_series(training, "annotation/005e070ac5e742cbb51dfa7513dac3b3", "2.25.1887", textured_volume(92))
            # 拼接正类被改名成 stitched_cases（不是 composition 前缀）
            write_series(training, "annotation/stitched_cases/CMP001", "T1", spliced_volume(93))

            layout = inspect_layout(training)
            # 严格前缀找不到，但会按同义词解析出实际目录
            self.assertIsNone(layout["special_dirs_exact"]["composition"])
            self.assertTrue(
                str(layout["special_dirs"]["composition"]).endswith("stitched_cases"),
                layout["special_dirs"],
            )
            self.assertTrue(any("composition" in hint for hint in layout["hints"]))
            candidates = candidate_kind_dirs(training, "composition")
            self.assertTrue(any("stitched_cases" in item for item in candidates), candidates)

            # 按候选目录名当正类前缀，就能拿到正类
            report = labeled_volumes(training, positive_kind="stitch")
            self.assertEqual(1, len(report["positives"]))
            self.assertEqual(2, len(report["negatives"]))

    def test_renamed_positive_dir_is_auto_detected_and_excluded_from_negatives(self) -> None:
        """正类目录改名成 compositing 时：默认调用要自动采用它，且不能把它算进负类。"""
        from tasks.goal2_stitched.dataset import labeled_volumes
        from tasks.goal2_stitched.evaluate import calibrate

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temporary:
            training = Path(temporary) / "training"
            write_series(training, "annotation/0050d79429cf4d86907dc8c4a34cbf04", "2.25.2574", textured_volume(94))
            write_series(training, "annotation/005e070ac5e742cbb51dfa7513dac3b3", "2.25.1887", textured_volume(95))
            write_series(training, "annotation/compositing/CMP001", "T1", spliced_volume(96))
            write_series(training, "annotation/fake/FAKE001", "T1", textured_volume(97))

            report = labeled_volumes(training)          # 不传 positive_kind，默认 composition
            self.assertEqual("compositing", report["positive_kind"][0])
            self.assertEqual(1, len(report["positives"]))
            # 负类 = 2 个正常病例（compositing 与 fake 都不能算负类）
            self.assertEqual(2, len(report["negatives"]))
            for path in report["negatives"]:
                self.assertNotIn("compositing", str(path))
                self.assertNotIn("fake", str(path))

            # 标定入口也要能用默认值跑通（--positive-kind 只是可选覆盖）
            payload = calibrate(training, target_fpr=0.5, workers=1)
            self.assertEqual("compositing", payload["positive_kind"])
            self.assertEqual(1, payload["positives"])
            self.assertEqual(2, payload["negatives"])

    def test_dataset_cli_rejects_empty_or_repo_data_root(self) -> None:
        """`--data-root ""`（$DATA_ROOT 为空）或指向代码仓库时必须直接报错，而不是给误报候选。"""
        import contextlib
        import io

        from tasks.goal2_duplicate.dataset import main as duplicate_main
        from tasks.goal2_stitched.dataset import main as stitched_main

        cwd = Path.cwd()
        for args in ([], [str(cwd)]):
            for runner in (stitched_main, duplicate_main):
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    code = runner(["--data-root", *args] if args else ["--data-root", ""])
                payload = json.loads(buffer.getvalue())
                self.assertEqual(2, code, payload)
                self.assertIn("hint", payload)


if __name__ == "__main__":
    unittest.main()
