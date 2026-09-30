"""模态判别**第三条腿**：从体素统计特征判 T1CE / T2 / FLAIR。

为什么必须有它（这是一次真实"分数为 0"的根因）
------------------------------------------------
提交侧的模态识别原本只有两条腿：

1. **关键词**（`data/series_selector.guess_modality`）—— 序列描述里含 `t1c`/`flair`/`t2`；
2. **查官方序列表**（`data/modality_fallback`）—— 按 ``(检查号, 序列号)`` 或 ``SeriesUid`` 回贴模态。

两条腿**都依赖"表里的键与磁盘一致"**。实测官方验证集上：序列描述认不出模态、
`SeriesType.xlsx` 也匹配不上 → `select()` 返回空 → Goal5 的输入通道全零、
掩膜退化写入参考序列 → **该例分割必然 0 分**，整批 777 例都是这样。

本模块补上**第三条腿**：只看**体素统计量**，不读任何表、不依赖文件名。
物理依据非常直接，不需要深度网络：

* **T2**：脑脊液明亮 → 亮体素占比高、中央区（脑室）明显偏亮；
* **FLAIR**：脑脊液被抑制 → 暗体素占比大、中央区不亮；
* **T1CE**：脑脊液暗 + 增强灶亮 → 暗体素多，但高强度尾部比 FLAIR 更重。

这三者恰好能被"直方图分位点 + 暗/亮体素占比 + 中央/全局亮度比"线性分开 ——
所以一个 3×15 的多项逻辑回归就够：

| 方案 | 单例成本 | 依赖 | 可解释性 |
|---|---|---|---|
| 3D 分类网络 | GPU 前向 | 需另行分发权重 | 差 |
| **本模块**（统计特征 + 逻辑回归） | **CPU 秒级** | 仅 numpy | 好（可看权重） |

特征与系数与训练侧 `glioma_track4` 的 `modality_model` **同源**（同一套 15 个特征、
同一份在官方训练集上拟合的系数），但**系数内嵌在本文件里**，因此本工程
**不 import 任何研发侧模块**、也不依赖任何随库文件 —— 少一个会丢的资产。

想用自己重训的模型覆盖：``GLIOMA_MODALITY_MODEL=/path/to/modality_model.json``
（用 `glioma_track4/scripts/31_train_modality_model.py` 训练产出，特征顺序必须一致）。

自检（可直接对真实 .nii.gz 跑）：

    python3 -m data.voxel_modality /path/a.nii.gz /path/b.nii.gz
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from data.structures import Study

__all__ = ["FEATURE_NAMES", "extract_features", "ModalityModel",
           "load_model", "recover_study", "describe"]

#: 判别的三类（官方 ``SeriesType`` 的 ``T1CE（增强）``/``T2-Flair`` 归一化后落到这三类）
CLASSES = ("T1CE", "T2", "FLAIR")

#: 特征名（顺序即向量顺序）
FEATURE_NAMES = (
    "p02", "p10", "p25", "p50", "p75", "p90", "p98",      # 分位点（按前景中位数归一）
    "iqr_over_range",                                       # 分布集中度
    "dark_frac",                                            # 暗体素占比（脑脊液抑制程度）
    "bright_frac",                                          # 亮体素占比（脑脊液/增强灶）
    "center_over_global",                                   # 中央区（脑室）与全局的亮度比
    "center_dark_frac",                                     # 中央区的暗体素占比
    "skew", "kurtosis",                                     # 形状
    "entropy",                                              # 直方图熵
)

#: 这些取值是**权威排除**（表里明写"其他"），不猜
_EXCLUDED = ("其他", "其它", "other", "无", "none", "正常", "平扫")

#: 低于该置信度宁可不用（把 FLAIR 当 T1C 比留空通道更有害）
_MIN_CONF = 0.5

#: 已就"体素判别结果"打过日志的 ``(检查号, allow_excluded)``。
#:
#: **只存字符串/布尔**，绝不缓存 Study —— 后者会拖住 ``Series.image``，几百例下来就是内存暴涨。
#: 为什么需要它：``series_selector.select()`` 对同一检查会被调**多次**（建体积 → 指纹 →
#: 各自模态的 `_restore`），每次都要重跑整条兜底链 → **同一结果实测会打 8 遍**。
#: 按 ``allow_excluded`` 分开记，是为了保留"哪一级腿救回来的"这一信息（最多 2 行/检查）。
_LOGGED: set[tuple[str, bool]] = set()

#: 内嵌系数（在官方训练集上拟合；与 ``glioma_track4/data/modality_model.json`` 同源）。
#: 顺序：``CLASSES × FEATURE_NAMES``。
_MEAN = (0.3478613793849945, 0.6432385444641113, 0.8277721405029297, 1.0,
         1.2230347394943237, 1.5546821355819702, 2.2499234676361084,
         0.2141619324684143, 0.028318988159298897, 0.080489881336689,
         1.0093514919281006, 0.02326132170855999, 1.3464651107788086,
         9.36868953704834, 2.6018576622009277)
_STD = (0.13975700736045837, 0.10582912713289261, 0.04468283802270889, 1.0,
        0.1162203699350357, 0.3859216272830963, 0.7287018895149231,
        0.052711088210344315, 0.02454579807817936, 0.06664380431175232,
        0.10163537412881851, 0.03276767581701279, 1.0704175233840942,
        6.288598537445068, 0.2543790340423584)
_WEIGHTS = (
    (0.40226459548257965, -0.6233588291377694, -0.559556518557985,
     -1.3331867957479055e-11, -0.3341226581372159, -0.7638067816232578,
     -0.27756283366343737, 0.36022832527056414, -0.7426041523259357,
     -0.5664675133463836, 0.667051801661814, 0.7313685003441195,
     0.47501496139589283, 0.45008972952065834, -0.6141554645393734),
    (0.11457840837771309, 1.0006667234620483, 0.15824484189301868,
     -1.511097545475385e-10, 0.4555264715696953, 0.7133153462658827,
     0.29366583505466975, 0.018912367796079604, 0.016872423621200686,
     0.7628390225608754, -0.3039894062467208, -1.1257206704199176,
     0.05837438878125726, -0.4251179041364631, 0.3393332139114428),
    (-0.5168430038602928, -0.37730789432427886, 0.40131167666496587,
     1.6444162250501723e-10, -0.12140381343247847, 0.05049143535737416,
     -0.01610300139123228, -0.3791406930666445, 0.7257317287047349,
     -0.19637150921449195, -0.3630623954150939, 0.3943521700757982,
     -0.5333893501771502, -0.024971825384194984, 0.2748222506279296),
)
_BIAS = (-0.27189117835032806, -0.17280441519420103, 0.44469559354452465)


# --------------------------------------------------------------------------- #
# 特征提取（与训练侧同一套定义；改动这里必须同步重训模型）
# --------------------------------------------------------------------------- #
def extract_features(volume: np.ndarray, stride: int = 2) -> np.ndarray:
    """从单个体数据提取 :data:`FEATURE_NAMES` 对应的特征向量。

    ``stride`` 抽稀（判别依据是全脑统计量，抽稀几乎不影响结论，但把单例耗时降到 1/8）。
    """
    vol = np.asarray(volume, dtype=np.float32)
    if stride > 1:
        vol = vol[::stride, ::stride, ::stride]
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return np.zeros(len(FEATURE_NAMES), np.float32)

    # 前景 = 非零体素；中位数作强度参照，消除不同序列的绝对标定差异
    fg = finite[finite > 0]
    if fg.size < 64:
        fg = finite
    ref = float(np.median(fg)) or 1.0
    norm = fg / ref

    q = np.percentile(norm, [2, 10, 25, 50, 75, 90, 98])
    dark_frac = float(np.mean(norm < 0.35))                  # 暗体素（脑脊液被抑制）
    bright_frac = float(np.mean(norm > 1.6))                 # 亮体素（脑脊液/增强灶）
    rng = float(q[6] - q[0]) or 1.0
    iqr_over_range = float((q[4] - q[2]) / rng)

    # 中央区（脑室所在）与全局的亮度比：T2 的脑室亮，FLAIR/T1CE 的暗
    center = vol[
        vol.shape[0] // 3: vol.shape[0] * 2 // 3,
        vol.shape[1] // 3: vol.shape[1] * 2 // 3,
        vol.shape[2] // 3: vol.shape[2] * 2 // 3,
    ]
    cen = center[center > 0]
    if cen.size >= 64:
        center_over_global = float(np.median(cen) / ref)
        center_dark_frac = float(np.mean(cen / ref < 0.35))
    else:
        center_over_global = 0.0
        center_dark_frac = 0.0

    mu = float(norm.mean())
    sd = float(norm.std()) or 1.0
    skew = float(((norm - mu) ** 3).mean() / sd ** 3)
    kurt = float(((norm - mu) ** 4).mean() / sd ** 4)
    hist, _ = np.histogram(norm, bins=32, range=(0.0, 3.0))
    p = hist / max(1, hist.sum())
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))

    return np.asarray([*q, iqr_over_range, dark_frac, bright_frac,
                       center_over_global, center_dark_frac,
                       skew, kurt, entropy], np.float32)


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
class ModalityModel:
    """多项逻辑回归 + 特征标准化；系数可内嵌、也可从 JSON 覆盖。"""

    def __init__(self, classes=CLASSES, mean=_MEAN, std=_STD,
                 weights=_WEIGHTS, bias=_BIAS) -> None:
        self.classes = list(classes)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)
        self.weights = np.asarray(weights, np.float32)
        self.bias = np.asarray(bias, np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 ``(n, k)`` 概率；入参允许一维（单个样本）。"""
        Xs = (np.atleast_2d(np.asarray(X, np.float32)) - self.mean) / self.std
        logits = Xs @ self.weights.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, volume: np.ndarray, stride: int = 2) -> tuple[str, dict[str, float]]:
        """对**体数据**预测 → ``(类别, {类别: 概率})``。"""
        prob = self.predict_proba(extract_features(volume, stride=stride))[0]
        return self.classes[int(prob.argmax())], {
            c: float(p) for c, p in zip(self.classes, prob)}

    @classmethod
    def load(cls, path: str | Path) -> "ModalityModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = list(data.get("feature_names") or [])
        if names and names != list(FEATURE_NAMES):
            raise ValueError(
                f"模态模型的特征顺序与本实现不一致（{len(names)} != "
                f"{len(FEATURE_NAMES)}）→ 用 scripts/31_train_modality_model.py 重训")
        return cls(classes=data.get("classes") or CLASSES,
                   mean=data["mean"], std=data["std"],
                   weights=data["weights"], bias=data["bias"])


_CACHE: dict[str, ModalityModel | None] = {}


def load_model(path: str | Path | None = None) -> ModalityModel:
    """加载模型：``GLIOMA_MODALITY_MODEL`` / 显式路径 优先，否则用**内嵌系数**。

    内嵌是为了让本工程**少一个会丢的随库文件**；只有你想用重训后的模型时才需要外部 JSON。
    """
    target = Path(path or os.environ.get("GLIOMA_MODALITY_MODEL", "")).expanduser() \
        if (path or os.environ.get("GLIOMA_MODALITY_MODEL")) else None
    key = str(target) if target else "<embedded>"
    if key in _CACHE:
        return _CACHE[key]                                       # type: ignore[return-value]
    if target is not None and target.is_file():
        try:
            model = ModalityModel.load(target)
            print(f"[voxel-modality] 已加载外部模型 {target}", flush=True)
            _CACHE[key] = model
            return model
        except Exception as exc:                                 # noqa: BLE001
            print(f"[voxel-modality] 外部模型不可用（{type(exc).__name__}: {exc}）→ 用内嵌系数",
                  flush=True)
    _CACHE[key] = ModalityModel()
    return _CACHE[key]                                           # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# 接进序列选择：给"认不出模态"的序列重贴模态
# --------------------------------------------------------------------------- #
def _is_excluded(series) -> bool:
    """该序列是不是**权威排除**（表里明写 `其他` / 正常 / 平扫）→ 不猜。"""
    text = f"{series.modality or ''} {((series.metadata or {}).get('modality') or '')}".casefold()
    return any(tok in text for tok in _EXCLUDED)


def guess_excluded_enabled() -> bool:
    """是否允许把表里**明写 ``其他``/``正常``/``平扫``**的序列也纳入猜测。

    ``GLIOMA_VOXEL_GUESS_EXCLUDED=0`` 关闭。**库默认开启**（保持既有行为），
    但 ``start.sh`` 已按实测结论**默认关闭** —— 见下。

    开它的理由：实测验证集上存在**整例序列全被标成 ``其他``** 的情况，
    此时尊重"权威排除"就等于**该例必然 0 分**（Goal5 通道全零 + 掩膜退化）。

    关它的理由（**实测后改变结论**）：

    1. ``其他`` 序列最可能是定位像 / 非脑之类的**异常序列**，而体素判别对这类 OOD 输入
       会给出**概率饱和**（``置信 1.0 / 次优 0.0``）—— ``_MIN_CONF = 0.5`` 形同虚设；
    2. ``scripts/voxel_consistency.py`` 实测的一致率**低于三分类的随机水平**。
       把定位像当 T1CE 填进通道，比留空通道**更有害**（留空至少是训练时见过的"缺失"标记）。

    判据：**一致率 <50%（≈随机）就该关**；若重训后测到 >=80%，再设回 ``1`` 打开。
    """
    return os.environ.get("GLIOMA_VOXEL_GUESS_EXCLUDED", "").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _log_excluded(study: Study, skipped: int, allow_excluded: bool) -> None:
    """报告"因权威排除而未参与判别"的序列数（每 ``(检查号, 模式)`` 只报一次）。

    ⚠️ 调用点必须在 ``if not cand: return`` **之前**：当**全部**序列都被排除时
    ``cand`` 就是空的，若把打印放在后面，这条信息会**在最需要的时候沉默** ——
    而它恰恰说明"该例是靠权威排除挡下来的"，是判断后续策略的关键。
    """
    if not skipped:
        return
    key = (study.accession_number, allow_excluded)
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    print(f"[voxel-modality] study {study.accession_number!r}: "
          f"{skipped} 条为权威排除（其他/正常/平扫），不猜"
          f"（allow_excluded={allow_excluded}）", flush=True)


def recover_study(study: Study, allow_excluded: bool = False) -> Study:
    """用体素统计给"认不出模态"的序列重贴模态；挑不出就原样返回。

    与 `data.modality_fallback.recover_study` 的关系：那个靠**表**，这个靠**体素**。
    两者互不依赖 —— 表匹配不上（或表也判不出模态）时这一步仍能生效。

    赋值方式：把 ``Series.modality`` 换成 ``T1CE`` / ``T2`` / ``FLAIR``，
    这三个词都能被 `series_selector.guess_modality` 正确识别；
    **不动 ``metadata``**（避免把非内部键写进去反而让 `_key_of` 认不出）。

    采用**全局贪心的一对一指派**：一路序列只能是一个模态，按概率降序占用，
    避免"先到先得"在撞车时白丢一路（如真值 T1CE/T2/FLAIR 被判成 T2/T2/FLAIR 时，
    第三个还能靠次优标签救回来）。

    ``allow_excluded=False``（默认）跳过表里明写 ``其他``/``正常``/``平扫`` 的序列
    （那是**权威排除**，猜错比留空更糟）。但实测官方验证集上有**整例序列全被标成
    `其他`** 的情况 —— 此时"不猜"等于**必然 0 分**。所以 `series_selector.select`
    会在第一遍无果后再调一次 ``allow_excluded=True``（把权威排除也纳入猜测），
    由 0.5 的置信度门槛兜住乱猜。
    """
    from data.series_selector import guess_modality                   # 局部导入：避免循环

    model = load_model()
    cand: list[tuple[float, str, str]] = []
    runner_up: dict[str, float] = {}
    skipped = 0
    for series in study.series:
        if guess_modality(series.modality) is not None:
            continue                                                  # 原描述已能用 → 不动
        if _is_excluded(series) and not allow_excluded:
            skipped += 1
            continue
        try:
            _label, probs = model.predict(np.asarray(series.image))
        except Exception as exc:                                      # noqa: BLE001
            print(f"[voxel-modality] {study.accession_number} 判别失败 "
                  f"{series.series_uid}: {type(exc).__name__}: {exc}", flush=True)
            continue
        ranked = sorted((float(v) for v in probs.values()), reverse=True)
        runner_up[series.series_uid] = ranked[1] if len(ranked) > 1 else 0.0
        for cls, p in probs.items():
            cand.append((float(p), series.series_uid, cls))

    if not cand:
        _log_excluded(study, skipped, allow_excluded)                 # ← 必须在这里，见其 docstring
        return study

    cand.sort(key=lambda x: -x[0])
    used_series: set[str] = set()
    used_cls: set[str] = set()
    # 三元组：(判定类别, 该类别概率, 次优类别概率) —— 次优用于识别**概率饱和**
    # （top≈1.0 且次优≈0.0 说明输入远超训练分布，此时 0.5 门槛形同虚设）
    changed: dict[str, tuple[str, float, float]] = {}
    for p, uid, cls in cand:
        if p < _MIN_CONF:
            break                                                     # 已降序 → 可直接停
        if uid in used_series or cls in used_cls:
            continue
        used_series.add(uid)
        used_cls.add(cls)
        changed[uid] = (cls, p, runner_up.get(uid, 0.0))

    if not changed:
        _log_excluded(study, skipped, allow_excluded)
        return study

    if (study.accession_number, allow_excluded) not in _LOGGED:
        _LOGGED.add((study.accession_number, allow_excluded))
        print(f"[voxel-modality] study {study.accession_number!r} 体素判别 → "
              f"{ {k: v[0] for k, v in changed.items()} }"
              f"（置信 {[round(v[1], 3) for v in changed.values()]}，"
              f"次优 {[round(v[2], 3) for v in changed.values()]}，"
              f"allow_excluded={allow_excluded}，"
              f"共 {len(study.series)} 条序列）", flush=True)
    return replace(study, series=tuple(
        replace(s, modality=changed[s.series_uid][0]) if s.series_uid in changed else s
        for s in study.series
    ))


def describe() -> str:
    """一句话说明当前用的是哪份模型（供报错文案/自检使用）。"""
    env = os.environ.get("GLIOMA_MODALITY_MODEL", "").strip()
    return (f"体素判别模型={env}（外部）" if env else "体素判别模型=内嵌系数（默认）")


# --------------------------------------------------------------------------- #
# 自检 CLI：直接对真实 NIfTI 跑判别
# --------------------------------------------------------------------------- #
def _main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        print(describe())
        print("用法: python3 -m data.voxel_modality <a.nii.gz> [b.nii.gz ...]")
        return 2
    import nibabel as nib
    model = load_model()
    print(describe())
    print(f"{'文件':<52}{'判别':>7}{'置信':>8}   概率")
    print("-" * 96)
    for p in argv:
        try:
            vol = np.asanyarray(nib.load(p).dataobj)
            vol = np.squeeze(vol)
            label, probs = model.predict(vol)
            top = max(probs.values())
            print(f"{Path(p).name:<52}{label:>7}{top:>8.3f}   "
                  f"{ {k: round(v, 3) for k, v in probs.items()} }")
        except Exception as exc:                                      # noqa: BLE001
            print(f"{Path(p).name:<52}   ✗ {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
