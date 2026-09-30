"""模态判别：从体素统计特征判 T1CE / T2 / FLAIR（数据信息表不可用时的兜底）。

**为什么需要它**

模态的权威来源是数据信息 ``<阶段>/annotation/SeriesType.xlsx``（训练/验证集都有，
评测集在正式测试时随测试数据一起下发）。但表**可能还没到手**、或表里没有这个检查号；
此时序列目录名是 DICOM UID（``2.25.*``），任何"按名字猜"的关键词都命中不了 ——
结果就是"扫出一堆检查、却一例都挑不出模态"。
官方自己的推理主链里为此专门有一个 ``sequence`` 任务（3 分类），
本模块就是它的轻量替代。**表里明写 ``其他`` 的序列不走这条路**（权威排除）。

**为什么不用 3D CNN**

| 方案 | 单例成本 | 训练成本 | 可解释性 | 依赖 |
|---|---|---|---|---|
| 3D 分类网络 | GPU 前向 | GPU 小时级 | 差（黑盒） | 需权重分发 |
| **本模块**（统计特征 + 逻辑回归） | CPU 秒级 | CPU 秒级 | 好（可看权重） | 仅 numpy/scipy |

判别的物理依据非常直接，不需要深层网络：

* **T2**：脑脊液明亮 → 低强度体素占比小、中央区（脑室）明显偏亮；
* **FLAIR**：脑脊液被抑制 → 低强度体素占比大、中央区不亮；
* **T1CE**：脑脊液暗 + 增强灶亮 → 低强度体素多，但高强度尾部比 FLAIR 更重。

这些恰好是直方图分位点 + "暗体素占比" + "中央/全局亮度比"能表达的，
因此回归模型就能把三者分开（本机 5 折实测准确率见
``scripts/31_train_modality_model.py`` 的输出）。

**与推理侧的关系**：模块只依赖 numpy/scipy，模型是几百个参数的小 JSON，
可以随工程分发，也可以在容器里用训练集现场重训（几分钟）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

#: 官方取值域（``schema.MODALITIES``）：``SeriesType`` 里的 ``T1CE（增强）``/``T2-Flair``
#: 等写法归一化后落到这三类（``T1`` 单独处理，不需要靠体素判）
OFFICIAL_CLASSES = ("T1CE", "T2", "FLAIR")

#: 特征名（顺序即向量顺序，便于看权重时定位）
FEATURE_NAMES = (
    "p02", "p10", "p25", "p50", "p75", "p90", "p98",      # 分位点（按前景中位数归一）
    "iqr_over_range",                                       # 分布集中度
    "dark_frac",                                            # 暗体素占比（脑脊液抑制程度）
    "bright_frac",                                          # 亮体素占比（增强灶/脑脊液）
    "center_over_global",                                   # 中央区（脑室）与全局的亮度比
    "center_dark_frac",                                     # 中央区的暗体素占比
    "skew", "kurtosis",                                     # 形状
    "entropy",                                              # 直方图熵
)


# --------------------------------------------------------------------------- #
# 特征提取
# --------------------------------------------------------------------------- #
def extract_features(volume: np.ndarray, stride: int = 2) -> np.ndarray:
    """从单个体数据提取 :data:`FEATURE_NAMES` 对应的特征向量。

    ``stride`` 用于抽稀（默认隔一个体素取一个）：判别依据是**全脑统计量**，
    抽稀对结论几乎无影响，但把单例耗时降到 1/8，评测时批量调用才可行。
    """
    vol = np.asarray(volume, dtype=np.float32)
    if stride > 1:
        vol = vol[::stride, ::stride, ::stride]
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return np.zeros(len(FEATURE_NAMES), np.float32)

    # 前景 = 非零体素；中位数作为强度参照，消除不同序列的绝对标定差异
    fg = finite[finite > 0]
    if fg.size < 64:
        fg = finite
    ref = float(np.median(fg)) or 1.0
    norm = fg / ref

    q = np.percentile(norm, [2, 10, 25, 50, 75, 90, 98])
    dark_frac = float(np.mean(norm < 0.35))                 # 暗体素（脑脊液被抑制）
    bright_frac = float(np.mean(norm > 1.6))                # 亮体素（脑脊液/增强灶）
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


def features_from_file(path: str | Path, stride: int = 2) -> np.ndarray:
    """读 NIfTI 并提取特征（评测兜底路径用这个）。"""
    import nibabel as nib

    img = nib.load(str(path))
    return extract_features(np.asanyarray(img.dataobj), stride=stride)


# --------------------------------------------------------------------------- #
# 模型（多项逻辑回归，scipy L-BFGS，无 sklearn 依赖）
# --------------------------------------------------------------------------- #
class ModalityModel:
    """多项逻辑回归 + 特征标准化；JSON 可序列化。"""

    def __init__(self, classes: list[str] | None = None,
                 mean: np.ndarray | None = None, std: np.ndarray | None = None,
                 weights: np.ndarray | None = None,
                 bias: np.ndarray | None = None) -> None:
        self.classes = list(classes or [])
        self.mean = None if mean is None else np.asarray(mean, np.float32)
        self.std = None if std is None else np.asarray(std, np.float32)
        self.weights = None if weights is None else np.asarray(weights, np.float32)
        self.bias = None if bias is None else np.asarray(bias, np.float32)

    # -- 训练 ------------------------------------------------------------- #
    def fit(self, X: np.ndarray, y: Any, l2: float = 1e-2,
            max_iter: int = 400) -> "ModalityModel":
        """训练。``y`` 可以是 ``'FLAIR'`` 这类字符串（官方标注表的取值），
        也可以是整数类别号 —— 统一转成字符串再排序。

        注意**不能**写成 ``np.asarray(y, int)``：官方序列标注表给的是字符串，
        那样会直接在 ``int('FLAIR')`` 上崩掉（本模块第一版就踩了这个坑）。
        """
        from scipy.optimize import minimize

        X = np.asarray(X, np.float32)
        y_lab = [str(t) for t in np.asarray(y).tolist()]
        present = sorted(set(y_lab))
        # 调用方若已指定类别顺序（如官方 T1CE/T2/FLAIR），按它来 ——
        # 训练与推理的类别索引必须一致，否则概率会挂到错误的名字上。
        self.classes = list(self.classes) or present
        for c in present:
            if c not in self.classes:
                self.classes.append(c)
        index = np.asarray([self.classes.index(t) for t in y_lab])

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-6] = 1.0
        Xs = (X - self.mean) / self.std
        n, d = Xs.shape
        k = len(self.classes)
        rows = np.arange(n)

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return theta[: k * d].reshape(k, d), theta[k * d:]

        def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
            W, b = unpack(theta)
            logits = Xs @ W.T + b
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            prob = exp / exp.sum(axis=1, keepdims=True)
            ll = -np.log(prob[rows, index] + 1e-12).mean()
            grad_logits = prob.copy()
            grad_logits[rows, index] -= 1.0
            grad_logits /= n
            gW = grad_logits.T @ Xs + l2 * W
            gb = grad_logits.sum(axis=0)
            return float(ll + 0.5 * l2 * float((W ** 2).sum())), np.concatenate(
                [gW.ravel(), gb])

        res = minimize(loss, np.zeros(k * d + k, np.float64), jac=True,
                       method="L-BFGS-B", options={"maxiter": max_iter})
        self.weights, self.bias = unpack(res.x)
        return self

    # -- 预测 ------------------------------------------------------------- #
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 ``(n, k)`` 概率。

        入参**允许一维**（单个样本的特征向量）—— ``predict_file`` 走的就是这条路。
        这里必须 ``np.atleast_2d``：下面的 ``logits.max(axis=1)`` 在 1 维输入上
        会直接抛 ``axis 1 is out of bounds for array of dimension 1``，
        而训练/CV 传的是二维批量，所以这个错在离线评测里**根本不会暴露**。
        """
        if self.weights is None or self.mean is None or self.std is None:
            raise RuntimeError("模型未训练/未加载")
        Xs = (np.atleast_2d(np.asarray(X, np.float32)) - self.mean) / self.std
        logits = Xs @ self.weights.T + self.bias
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> list[str]:
        prob = self.predict_proba(np.atleast_2d(X))
        return [self.classes[int(i)] for i in prob.argmax(axis=1)]

    # -- 存取 ------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "feature_names": list(FEATURE_NAMES),
            "mean": None if self.mean is None else self.mean.tolist(),
            "std": None if self.std is None else self.std.tolist(),
            "weights": None if self.weights is None else self.weights.tolist(),
            "bias": None if self.bias is None else self.bias.tolist(),
        }

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> "ModalityModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = data.get("feature_names") or []
        if names and list(names) != list(FEATURE_NAMES):
            raise ValueError(
                f"模态模型的 {len(names)} 个特征与当前实现的 {len(FEATURE_NAMES)} 个不一致："
                f"请用 scripts/31_train_modality_model.py 重新训练")
        return cls(classes=data.get("classes"),
                   mean=np.asarray(data["mean"], np.float32),
                   std=np.asarray(data["std"], np.float32),
                   weights=np.asarray(data["weights"], np.float32),
                   bias=np.asarray(data["bias"], np.float32))

    # -- 便捷入口 ---------------------------------------------------------- #
    def predict_file(self, path: str | Path, stride: int = 2) -> tuple[str, dict[str, float]]:
        """直接对 NIfTI 文件预测 → ``(模态, {类别: 概率})``。"""
        prob = self.predict_proba(features_from_file(path, stride=stride))[0]
        return self.classes[int(prob.argmax())], {
            c: float(p) for c, p in zip(self.classes, prob)}


#: 默认模型路径（训练脚本的默认输出；缺失时调用方应优雅降级）
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "modality_model.json"


def load_default_model(path: str | Path | None = None) -> ModalityModel | None:
    """加载默认模型；不存在或不可用时返回 ``None``（调用方据此走无模型分支）。"""
    target = Path(path) if path else DEFAULT_MODEL_PATH
    if not target.is_file():
        return None
    try:
        return ModalityModel.load(target)
    except Exception:                                             # noqa: BLE001
        return None
