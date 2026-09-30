"""其余五个插件的契约测试（规范 §5.1 最低运行交付 / §18.3）。

``goal5`` 已有独立契约测试（``test_goal5_contract.py``）；本文件覆盖剩下五个：
``goal1_authenticity`` / ``goal2_stitched`` / ``goal3_tumor`` / ``goal4_diagnosis``
（``StudyTask``）与 ``goal2_duplicate``（``DatasetTask``）。

规范 §5.1 要求"最低运行交付必须包含配置、模型定义、预处理、后处理、
推理入口、Task 适配器和契约测试"——前六项此前只覆盖到四个 Goal，
契约测试则只有 Goal5 一份，本文件补齐这一缺口。

**只验证接口契约与约束，不依赖真实训练权重**，因此可在 CI 中稳定运行：

1. Task 类存在、基类正确、``name`` 与模块名一致；
2. 权重相对路径合法（不得逃逸 checkpoint 根目录）；
3. 比赛入口**不传递导入**训练专用模块（``dataset``/``augmentations``/
   ``losses``/``train``/``evaluate``）——规范 §5.1 明令禁止；
4. 构建 Task 不产生比赛目录副作用（不写 ``answer/``）；
5. 规范要求的入口文件齐备。
"""
from __future__ import annotations

import ast
import importlib
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: ``(模块名, Task 类名, 是否为数据集级任务)``
PLUGINS: list[tuple[str, str, bool]] = [
    ("goal1_authenticity", "Goal1Task", False),
    ("goal2_stitched", "StitchedTask", False),
    ("goal3_tumor", "TumorTask", False),
    ("goal4_diagnosis", "DiagnosisTask", False),
    ("goal2_duplicate", "DuplicateTask", True),
]

#: 比赛入口不得传递导入的训练专用模块（规范 §5.1）
BANNED_MODULES = {"dataset", "augmentations", "losses", "train", "evaluate"}

#: 规范 §5.1 要求的入口文件
REQUIRED_FILES = ("__init__.py", "task.py", "config.py", "postprocess.py",
                  "inference.py", "preprocess.py")


def _pkg(mod: str) -> pathlib.Path:
    return ROOT / "tasks" / mod


def _load_task_class(mod: str, cls_name: str):
    return getattr(importlib.import_module(f"tasks.{mod}.task"), cls_name)


def _declared_rel(mod: str) -> str | None:
    """从源码里取 ``ckpt_rel``（类属性或 config 默认值均可）。"""
    for name in ("task.py", "config.py"):
        f = _pkg(mod) / name
        if not f.is_file():
            continue
        m = re.search(r'ckpt_rel[^=\n]*=\s*"([^"]+)"', f.read_text(encoding="utf-8"))
        if m:
            return m.group(1)
    return None


# --------------------------------------------------------------------------- #
# 1. Task 类与基类
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_task_class_exists_with_right_base(mod, cls_name, is_dataset):
    from tasks.base import DatasetTask, StudyTask

    cls = _load_task_class(mod, cls_name)
    base = DatasetTask if is_dataset else StudyTask
    assert issubclass(cls, base), f"{cls_name} 应继承 {base.__name__}"


@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_task_declares_name_matching_module(mod, cls_name, is_dataset):
    assert _load_task_class(mod, cls_name).name == mod


# --------------------------------------------------------------------------- #
# 2. 权重相对路径：合法且不逃逸
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_ckpt_rel_is_relative_and_safe(mod, cls_name, is_dataset):
    rel = _declared_rel(mod)
    assert rel, f"{mod}: 未声明 ckpt_rel"
    assert not rel.startswith("/"), f"{mod}: ckpt_rel 必须是相对路径，实际 {rel!r}"
    assert ".." not in rel, f"{mod}: ckpt_rel 不得含 ..（可逃逸 checkpoint 根目录）"


# --------------------------------------------------------------------------- #
# 3. 比赛入口不得传递导入训练专用模块（规范 §5.1）
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_entrypoint_does_not_import_training_modules(mod, cls_name, is_dataset):
    offenders: list[str] = []
    for f in sorted(_pkg(mod).rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            else:
                continue
            for m in mods:
                # 只看**本 Goal 包内**的训练专用模块；公共库不算
                if f"{mod}." in m and m.rsplit(".", 1)[-1] in BANNED_MODULES:
                    offenders.append(f"{f.name}: {m}")
    assert not offenders, f"比赛入口导入了训练专用模块: {offenders}"


# --------------------------------------------------------------------------- #
# 4. 构建 Task 不产生比赛目录副作用
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_build_task_has_no_answer_side_effect(mod, cls_name, is_dataset,
                                              tmp_path, monkeypatch):
    from core.config import Settings

    monkeypatch.setenv("COMPETITION_WORKSPACE", str(tmp_path / "ws"))
    builder = getattr(importlib.import_module(f"tasks.{mod}.task"), "build_task")
    task = builder(settings=Settings.from_env())
    assert task.name == mod
    ws = tmp_path / "ws"
    assert not (ws / "answer").exists(), "Task 构建阶段不得创建 answer/"
    assert not list(ws.rglob("prediction.json")), "Task 构建阶段不得写 prediction.json"


# --------------------------------------------------------------------------- #
# 5. 规范 §5.1 要求的入口文件齐备
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_required_entrypoint_files_exist(mod, cls_name, is_dataset):
    missing = [f for f in REQUIRED_FILES if not (_pkg(mod) / f).is_file()]
    assert not missing, f"{mod} 缺少规范 §5.1 要求的文件: {missing}"


@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_task_module_imports_its_result_type(mod, cls_name, is_dataset):
    """Task 必须产出强类型 Result（规范 §8 要求各 Goal 有自己的结果类型）。"""
    src = (_pkg(mod) / "task.py").read_text(encoding="utf-8")
    assert "Result" in src, f"{mod}/task.py 未引用任何 Result 类型"


# --------------------------------------------------------------------------- #
# 6. [研发] 入口文件必须导出**可调用**的 API（而不是空壳）
# --------------------------------------------------------------------------- #
#: ``(子模块, 必需的符号)`` —— 这些是各 Goal 对外的稳定入口
DEV_API: tuple[tuple[str, str], ...] = (
    ("dataset", "build_datasets"),
    ("augmentations", "build_augment"),
    ("augmentations", "disabled"),
    ("losses", "build_loss_fn"),
    ("losses", "loss_weights"),
    ("losses", "cls_spec"),
    ("evaluate", "main"),
    ("train", "run_goal"),
)


@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_dev_entrypoints_export_callable_api(mod, cls_name, is_dataset):
    """每个 Goal 的 dataset/augmentations/losses/evaluate 必须导出可调用入口。

    早期这些文件只有 docstring + 一个常量（``GOAL = "..."``）：**文件在、
    功能不在** —— ``from tasks.<goal>.dataset import build_datasets`` 会直接
    ImportError。这类"空壳"不会让任何测试失败，却让规范 §5.1 的
    "最低运行交付"名不副实，因此必须显式断言。
    """
    missing: list[str] = []
    for sub, sym in DEV_API:
        try:
            m = importlib.import_module(f"tasks.{mod}.{sub}")
        except Exception as exc:                                  # noqa: BLE001
            missing.append(f"{sub}: 导入失败 {type(exc).__name__}: {exc}")
            continue
        if not callable(getattr(m, sym, None)):
            missing.append(f"{sub}.{sym}")
    assert not missing, f"{mod} 缺少可调用入口: {missing}"


@pytest.mark.parametrize("mod,cls_name,is_dataset", PLUGINS)
def test_dev_entrypoints_are_not_imported_by_inference(mod, cls_name, is_dataset):
    """[研发] 入口可被独立导入，但**推理侧不得传递导入它们**。

    ``task.py`` 只允许依赖推理所需的模块；一旦它 import 了 ``dataset`` /
    ``losses`` 等，比赛运行入口就会被拖上训练期的重依赖（规范 §5.1）。
    """
    src = (_pkg(mod) / "task.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = [m for m in imported
                 if f"tasks.{mod}." in m
                 and m.rsplit(".", 1)[-1] in {"dataset", "augmentations", "losses",
                                              "train", "evaluate"}]
    assert not offenders, f"{mod}/task.py 不应导入 [研发] 模块: {offenders}"
