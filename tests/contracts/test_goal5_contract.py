"""Goal5 插件契约测试（规范 §19.2 / §18.3）。

只验证**接口契约与约束**，不依赖真实训练权重，因此可在 CI 中稳定运行：
1. 能实例化并加载模型；
2. 权重只能从 ``{ckpt_root}/<relative>`` 解析，不允许路径逃逸；
3. 权重与配置结构不兼容时必须**明确失败**，而不是静默加载随机层；
4. 返回约定的强类型 ``Goal5Result``；
5. 不产生任何比赛目录副作用（不写 answer/、不发回调）；
6. 比赛入口不传递导入训练专用模块。
"""
from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Settings                                    # noqa: E402
from data.structures import Series, Study                           # noqa: E402
from tasks.goal5.config import Goal5Config             # noqa: E402
from tasks.goal5.inference import resolve_ckpt         # noqa: E402
from tasks.results import Goal5Result                               # noqa: E402


# --------------------------------------------------------------------------- #
# 1. 路径解析：只能落在 checkpoint 根目录内
# --------------------------------------------------------------------------- #
def test_resolve_ckpt_stays_inside_root(tmp_path):
    got = resolve_ckpt(tmp_path, "goal5_segmentation/core.pt")
    assert got == (tmp_path / "goal5_segmentation" / "core.pt").resolve()


def test_resolve_ckpt_rejects_path_escape(tmp_path):
    with pytest.raises(ValueError):
        resolve_ckpt(tmp_path, "../../etc/passwd")


# --------------------------------------------------------------------------- #
# 2. 权重缺失 / 结构不兼容必须明确失败
# --------------------------------------------------------------------------- #
def test_load_model_missing_weight_raises(tmp_path):
    from tasks.goal5.inference import load_model

    with pytest.raises(FileNotFoundError):
        load_model(Goal5Config(), tmp_path, device="cpu")


def test_load_model_rejects_incompatible_channels(tmp_path):
    torch = pytest.importorskip("torch")
    from tasks.goal5.inference import load_model

    ckpt = tmp_path / "goal5_segmentation"
    ckpt.mkdir(parents=True)
    # in_ch 与配置不符（3 vs 4）→ 必须拒绝，否则会静默加载随机层
    torch.save({"arch": "mednext", "model_cfg": {"in_ch": 3}, "cls_spec": []},
               ckpt / "core.pt")
    with pytest.raises(ValueError):
        load_model(Goal5Config(in_channels=4), tmp_path, device="cpu")


# --------------------------------------------------------------------------- #
# 3. 结果类型契约
# --------------------------------------------------------------------------- #
def test_goal5_result_shape_and_dtype_contract():
    m = np.zeros((4, 5, 6), dtype=np.uint8)
    r = Goal5Result(core_mask=m, core_source_series_uid="uid_core",
                    flair_mask=m.copy(), flair_source_series_uid="uid_flair")
    assert r.core_mask.dtype == np.uint8
    assert set(np.unique(r.core_mask)) <= {0, 1}
    assert r.core_source_series_uid and r.flair_source_series_uid


# --------------------------------------------------------------------------- #
# 4. 比赛入口不导入训练专用模块（规范 §17.1）
# --------------------------------------------------------------------------- #
def test_entrypoint_does_not_import_training_modules():
    banned = {"dataset", "augmentations", "losses", "train", "evaluate"}
    offenders: list[str] = []
    for f in sorted((ROOT / "tasks" / "goal5_segmentation").rglob("*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for mod in mods:
                if "goal5_segmentation" in mod and mod.rsplit(".", 1)[-1] in banned:
                    offenders.append(f"{f.name}: {mod}")
    assert not offenders, f"比赛入口导入了训练专用模块: {offenders}"


# --------------------------------------------------------------------------- #
# 5. 插件不产生比赛目录副作用
# --------------------------------------------------------------------------- #
def test_task_does_not_touch_answer_dir(tmp_path, monkeypatch):
    """实例化与 load_model 失败都不应创建任何 answer/ 目录。

    ⚠️ **必须把权重根也钉在临时目录里**（只钉 ``COMPETITION_WORKSPACE`` 不够）：
    ``Settings.from_env()`` 会读 ``COMPETITION_CHECKPOINT_ROOT``，而正式评测容器里
    这个变量**本来就是设好的**（指向真权重）。于是本测试会真的把模型加载起来跑完
    推理，``predict`` 不再抛 ``FileNotFoundError`` → 断言失败：
    ``Failed: DID NOT RAISE FileNotFoundError``。

    症状很迷惑：CI 上通过（那里没有权重根），一进容器就失败。
    根因是**测试依赖了环境**，不是被测代码有问题 —— 把它钉死即可两边都稳定。
    """
    from tasks.goal5.task import Goal5Task

    monkeypatch.setenv("COMPETITION_WORKSPACE", str(tmp_path / "ws"))
    # 钉死权重根 → 本例内**必然**找不到权重，与外部环境无关
    monkeypatch.setenv("COMPETITION_CHECKPOINT_ROOT", str(tmp_path / "no-such-ckpt"))
    task = Goal5Task(settings=Settings.from_env())
    assert task.name == "goal5_segmentation"
    # 未加载权重时 predict 会尝试加载并因缺权重失败，但不得写任何文件
    study = Study(accession_number="ACC1", series=(
        Series(series_uid="S1", modality="t1c post contrast",
               image=np.ones((4, 4, 4), np.float32), affine=np.eye(4),
               source_path=tmp_path / "S1.nii.gz", metadata={}),
    ))
    from pipeline.context import PipelineContext
    with pytest.raises(FileNotFoundError):
        task.predict(PipelineContext(study=study))
    assert not (tmp_path / "ws" / "answer").exists()


# --------------------------------------------------------------------------- #
# 6. 缺模态必须走"零占位 + warning"，而不是崩溃
# --------------------------------------------------------------------------- #
def test_preprocess_fills_missing_channels(tmp_path):
    from tasks.goal5.preprocess import CHANNEL_ORDER, build_volume

    study = Study(accession_number="ACC1", series=(
        Series(series_uid="S1", modality="t1c post contrast",
               image=np.random.RandomState(0).rand(8, 8, 8).astype(np.float32) * 100,
               affine=np.eye(4), source_path=tmp_path / "S1.nii.gz", metadata={}),
    ))
    prep = build_volume(study, Goal5Config())
    assert prep.volume.shape[0] == len(CHANNEL_ORDER)
    assert set(prep.missing) == {"flair", "t2", "t1"}


def test_study_rejects_empty_series_at_construction():
    """``Study`` 在**构造期**就拒绝空 series。

    这条由团队公共数据结构保证的契约很重要：它意味着"无影像的 Study"
    根本不可能进入 Pipeline，因此插件无需为它设计降级分支。
    （规范 §9.1 把"整个 Study 无有效影像"定义为不可降级错误，
     这里进一步说明它连构造都过不去。）
    """
    with pytest.raises(ValueError):
        Study(accession_number="ACC_EMPTY", series=())


def test_bridge_matches_closing_and_stays_local():
    """``postprocess._bridge`` 必须是闭运算语义，且**不得**依赖稠密结构元。

    背景（真实事故）：``_bridge`` 原先用 ``ndimage.binary_closing(structure=21³)``。
    空掩膜时代 ``clean_mask`` 会在 ``if not m.any(): return`` 提前返回，
    所以它**从未被执行**；等空掩膜修好、掩膜变成非空之后它才第一次真正运行 ——

    ==================  ==================  ============
    实现                 峰值内存增量        单次耗时
    ==================  ==================  ============
    21³ 稠密结构元       **+701 MB**         **31.7 s**
    当前实现（EDT+包围盒）  +112 MB             0.30 s
    ==================  ==================  ============

    （实测体积 ``240x240x155``；``clean_pair`` 会调用两次。）改成稠密结构元
    会立刻把"容器 OOM 重启"带回来，所以这里锁死两件事：

    1. 输出与闭运算参考实现一致；
    2. **距掩膜 > radius 的体素必须保持背景** —— 这条同时保证了
       "只在包围盒外扩 radius 的窗口内计算"这个内存优化是安全的。
    """
    from scipy import ndimage

    from tasks.goal5.postprocess import _bridge

    m = np.zeros((40, 40, 24), dtype=bool)
    m[10:14, 10:14, 8:12] = True                     # 两块相距 6 体素
    m[20:24, 20:24, 8:12] = True
    m[30, 30, 5] = True                              # 一处孤立斑点

    radius = 4.0
    got = _bridge(m, radius, (1.0, 1.0, 1.0))
    # 参考：半径 4 的稠密立方结构元闭运算（体积刻意为小，避免测试自身吃内存）
    ref = ndimage.binary_closing(m, structure=np.ones((9, 9, 9), dtype=bool),
                                 border_value=0)
    inter = int((got & ref).sum())
    union = int((got | ref).sum()) or 1
    assert inter / union > 0.9, f"与闭运算参考实现差异过大 IoU={inter / union:.3f}"
    assert got.dtype == np.bool_, "闭运算结果应是布尔掩膜"

    # 闭运算 ⊆ 膨胀 ⇒ 距掩膜 > radius 的体素不可能被点亮
    far = ndimage.distance_transform_edt(~m) > radius
    assert not (got & far).any(), "闭运算影响了半径之外的体素 → 包围盒裁剪不安全"

    # 空掩膜直接原样返回（不得进入距离变换）
    empty = np.zeros((8, 8, 8), dtype=bool)
    assert not _bridge(empty, 10.0, (1.0, 1.0, 1.0)).any()
    # radius <= 0 时不做任何处理
    assert np.array_equal(_bridge(m, 0.0, (1.0, 1.0, 1.0)), m)


def test_shared_and_goal5_preprocess_agree_with_training():
    """两份**分开维护**的推理侧预处理必须与彼此、与训练侧一致。

    推理侧有**两份**多通道预处理：

    - ``tasks/_common/volume.py`` —— Goal1/2/3/4 走的共享骨干（``BackboneRunner``）；
    - ``tasks/goal5_segmentation/preprocess.py`` —— Goal5 自己的那份。

    历史上正是"**只修了一份**"：Goal5 的空掩膜修好了、通道 fallback 补上了，
    而 ``tasks/_common/volume.py`` 仍是"缺通道填零 + 参考序 ``t1c→t1→flair→t2``
    + ``max_spacing_factor`` 走默认 4.0" —— 于是四个分类/嵌入头继续吃 OOD 输入。

    这里把三件事锁死（训练事实来源是 ``glioma_track4/configs/preprocess.yaml``，
    仓内训练入口 ``tasks/_common/training/helpers.build_datasets`` 直接加载它）：
    通道取用链、参考网格优先级、``max_spacing_factor``。
    """
    from tasks._common import volume as shared
    from tasks.goal5 import preprocess as g5
    from tasks.goal5.config import Goal5Config

    # 两份互相同源
    assert shared.CHANNEL_ORDER == g5.CHANNEL_ORDER
    assert shared.CHANNEL_FALLBACK == g5.CHANNEL_FALLBACK
    assert shared._REF_PRIORITY == g5._REF_PRIORITY

    # 与训练侧逐条对齐
    assert shared.CHANNEL_FALLBACK == {
        "t1c": ("t1c", "t1", "t2"),      # preprocess.yaml: fallback: [t1, t2]
        "flair": ("flair", "t2"),        # preprocess.yaml: fallback: [t2]
        "t2": ("t2",),
        "t1": ("t1",),
    }, "通道取用链与训练侧 preprocess.yaml 的 channels 不一致"
    assert tuple(shared._REF_PRIORITY) == ("t1c", "flair", "t2", "t1"), \
        "参考网格优先级与 dataset.build_case_volume 不一致"
    assert shared.DEFAULT_MAX_SPACING_FACTOR == 1.5, \
        "max_spacing_factor 与 preprocess.yaml 的 geometry 段不一致"
    assert Goal5Config().max_spacing_factor == shared.DEFAULT_MAX_SPACING_FACTOR, \
        "Goal5Config.max_spacing_factor 与共享预处理不一致"


def test_study_rejects_duplicate_series_uid(tmp_path):
    """同一 Study 内 series_uid 不得重复（规范 §6.1）。"""
    s = Series(series_uid="S1", modality="t1c", image=np.ones((4, 4, 4), np.float32),
               affine=np.eye(4), source_path=tmp_path / "a.nii.gz", metadata={})
    with pytest.raises(ValueError):
        Study(accession_number="ACC1", series=(s, s))
