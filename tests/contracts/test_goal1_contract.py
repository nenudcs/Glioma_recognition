"""Goal1 authenticity 契约测试（规范 §19.2）。

逐条检查规范为插件规定的契约：

1. 能实例化并通过 ``task.py`` 进入（注册入口返回 InferencePipeline，顺序符合 §8.1）；
2. checkpoint 只能从 ``<checkpoint_root>/goal1_authenticity/`` 解析；
3. 接收最小 ``PipelineContext``、返回强类型 ``Goal1Result``、概率合法；
4. 不产生比赛目录副作用（不写 answer/prediction.json/duplicate_pairs.jsonl）；
5. 比赛运行链路不传递导入训练专用模块（dataset/augmentations/losses/train/evaluate）；
6. 错误转换为公共异常类型（MissingSeriesError / ModelInferenceError）。

没有可用权重时，涉及真实前向的用例自动跳过（其余用例用桩替换推理器）。
"""
from __future__ import annotations

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

from core.exceptions import MissingSeriesError, ModelInferenceError  # noqa: E402
from data.structures import Series, Study  # noqa: E402
from pipeline.context import PipelineContext  # noqa: E402
from pipeline.inference import InferencePipeline  # noqa: E402
from tasks.goal1_authenticity import checkpoint as ck  # noqa: E402
from tasks.goal1_authenticity.config import Goal1Config  # noqa: E402
from tasks.goal1_authenticity.task import Goal1AuthenticityTask  # noqa: E402
from tasks.results import Goal1Result  # noqa: E402

SPEC_RELATIVE = Path("goal1_authenticity") / "model.pt"
FORBIDDEN_MODULES = (
    "tasks.goal1_authenticity.dataset",
    "tasks.goal1_authenticity.augmentations",
    "tasks.goal1_authenticity.losses",
    "tasks.goal1_authenticity.train",
    "tasks.goal1_authenticity.evaluate",
)


def _weights_available() -> bool:
    override = os.environ.get("GOAL1_CHECKPOINT")
    if override and Path(override).is_file():
        return True
    return ck.model_path().is_file()


def _context(volume: np.ndarray | None = None) -> PipelineContext:
    image = np.zeros((8, 16, 16), dtype=np.float32) if volume is None else volume
    series = Series(
        series_uid="T1CE",
        modality="T1CE (增强)",
        image=image,
        affine=np.eye(4),
        source_path=Path("/tmp/T1CE.nii.gz"),
    )
    return PipelineContext(study=Study("ACC001", (series,)))


class Goal1RegistryContractTest(unittest.TestCase):
    def test_real_pipeline_binds_goal1_first(self) -> None:
        from tasks.real_pipeline import build_pipeline

        pipeline = build_pipeline()
        self.assertIsInstance(pipeline, InferencePipeline)
        fields = [binding.context_field for binding in pipeline.study_tasks]
        self.assertEqual(["goal1", "goal2_stitched", "goal3", "goal5", "goal4"], fields)
        self.assertIsInstance(pipeline.study_tasks[0].task, Goal1AuthenticityTask)

    def test_checkpoint_path_follows_spec_layout(self) -> None:
        self.assertEqual(SPEC_RELATIVE, ck.relative_model_path())
        root = Path("/tmp/ckroot")
        self.assertEqual(root / SPEC_RELATIVE, ck.model_path(root))
        self.assertEqual(root, ck.checkpoint_root(root))
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            previous = os.environ.get("COMPETITION_WORKSPACE")
            os.environ["COMPETITION_WORKSPACE"] = str(workspace)
            try:
                self.assertEqual(workspace / "checkpoint", ck.checkpoint_root(None))
            finally:
                if previous is None:
                    os.environ.pop("COMPETITION_WORKSPACE", None)
                else:
                    os.environ["COMPETITION_WORKSPACE"] = previous

    def test_config_resolves_weights_from_environment(self) -> None:
        # GOAL1_CHECKPOINT 是临时覆盖，用完后要还原（本用例验证的是规范路径解析）
        previous = os.environ.pop("GOAL1_CHECKPOINT", None)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                config = Goal1Config(checkpoint_root=root)
                self.assertEqual(root / SPEC_RELATIVE, config.resolved_model_path)
                self.assertEqual(str(root / SPEC_RELATIVE), config.describe()["model_path"])
        finally:
            if previous is not None:
                os.environ["GOAL1_CHECKPOINT"] = previous

    def test_runtime_chain_avoids_training_modules(self) -> None:
        """在独立解释器里加载注册入口，确认训练专用模块没有被传递导入。"""
        script = (
            "import sys;"
            "import tasks.real_pipeline as rp;"
            "rp.build_pipeline();"
            "print([m for m in sys.modules if 'goal1_authenticity' in m])"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env["GOAL1_CHECKPOINT"] = "/nonexistent/model.pt"     # 避免真加载权重
        env["GOAL1_STRICT"] = "0"
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


class Goal1BehaviourContractTest(unittest.TestCase):
    def test_predict_returns_goal1_result_with_valid_probability(self) -> None:
        if not _weights_available():
            self.skipTest("规范路径下没有权重（可先跑 migrate_checkpoint）")
        task = Goal1AuthenticityTask(Goal1Config(strict=True))
        task.load_model()
        context = _context()

        cwd_before = {path.name for path in Path.cwd().iterdir()}
        result = task.predict(context)
        cwd_after = {path.name for path in Path.cwd().iterdir()}

        self.assertIsInstance(result, Goal1Result)
        self.assertTrue(np.isfinite(result.not_human_probability))
        self.assertGreaterEqual(result.not_human_probability, 0.0)
        self.assertLessEqual(result.not_human_probability, 1.0)
        # 不产生比赛目录副作用
        self.assertEqual(cwd_before, cwd_after)
        self.assertFalse((Path.cwd() / "answer").exists())
        # 诊断信息包含任务名/耗时/模型版本（规范 §20.2）
        detail = context.diagnostics["goal1"]
        for key in ("task_name", "duration_ms", "model_version", "series"):
            self.assertIn(key, detail)

    def test_predict_without_loaded_model_raises_public_exception(self) -> None:
        task = Goal1AuthenticityTask(Goal1Config(strict=False))
        # 不调用 load_model()：模拟权重缺失（predict 必须抛公共异常，而不是输出兜底值）
        with self.assertRaises(ModelInferenceError):
            task.predict(_context())

    def test_study_without_scorable_series_raises_missing_series(self) -> None:
        task = Goal1AuthenticityTask(Goal1Config())

        class _AlwaysFailing:
            def score_volume(self, _image):
                raise RuntimeError("stub failure")

        task.inference._net = object()          # 让 ready 为真，绕过真实权重
        task.inference.score_volume = _AlwaysFailing().score_volume
        context = _context()
        with self.assertRaises(MissingSeriesError):
            task.predict(context)
        self.assertTrue(context.warnings)       # 单序列失败已记录

    def test_single_bad_series_is_degradable(self) -> None:
        task = Goal1AuthenticityTask(Goal1Config())

        class _Stub:
            def __init__(self) -> None:
                self.calls = 0

            def score_volume(self, _image):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("第一序列坏了")
                return 0.75

        stub = _Stub()
        task.inference._net = object()
        task.inference.score_volume = stub.score_volume
        series_a = Series("A", "T1CE", np.zeros((8, 16, 16), np.float32), np.eye(4), Path("/tmp/A.nii.gz"))
        series_b = Series("B", "FLAIR", np.zeros((8, 16, 16), np.float32), np.eye(4), Path("/tmp/B.nii.gz"))
        context = PipelineContext(study=Study("ACC002", (series_a, series_b)))

        result = task.predict(context)
        self.assertAlmostEqual(0.75, result.not_human_probability, places=6)
        self.assertEqual(1, len(context.warnings))
        self.assertEqual(1, context.diagnostics["goal1"]["scored_series"])


if __name__ == "__main__":
    unittest.main()
