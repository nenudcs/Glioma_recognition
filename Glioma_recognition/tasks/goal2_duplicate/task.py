"""Goal2-B 重复影像检测（规范 §9.3）：**数据集级** DatasetTask。

这是六个插件里唯一的 ``DatasetTask``，生命周期与 StudyTask 不同：

    reset()                    → 开始一次 evaluation，丢弃上一轮状态
    update(study, context)     → 每个检查到达时累积轻量信息
    finalize()                 → evaluation 结束时产出全局重复对

**状态管理是这里最容易出错的地方**（规范 §18.2 专门列了 DoD）：

1. ``reset()`` 必须真正清空——否则上一轮评测的嵌入会混进来，
   表现为"重复对数量异常多、且包含不存在的 accession"；
2. ``update()`` **只允许保留 accession / embedding / 轻量指纹**，
   绝不能持有 Study、Context、原图或 GPU Tensor。624 例原图常驻会直接吃光 64GB 上限；
3. 嵌入在 ``update()`` 内即时计算并**立刻转成 numpy CPU 数组**，
   算完即释放中间张量——这是唯一能保证内存不随数据集规模线性膨胀的做法。

输出为规范化 pair（无向、去重、取最高概率、确定性排序），
**不写 JSONL**：路径、Top-200 截断与排序由 ``OutputWriter.write_duplicates()`` 统一负责
（规范 §9.3 明确要求插件不直接写文件）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from core.config import Settings
from tasks._common.volume import build_volume, global_view
from tasks.base import DatasetTask
from tasks.goal2_duplicate.config import DuplicateConfig
from tasks.goal2_duplicate.postprocess import normalize_pairs
from tasks.goal2_duplicate.retrieval import calibrate, match_pairs
from tasks.results import DuplicatePair, DuplicateResult


class DuplicateTask(DatasetTask[DuplicateResult]):
    """跨检查的重复影像检测。"""

    name = "goal2_duplicate"

    def __init__(self, cfg: DuplicateConfig | None = None,
                 settings: Settings | None = None, device: str = "cuda") -> None:
        self.cfg = cfg or DuplicateConfig()
        self.settings = settings or Settings.from_env()
        self.device = device
        self._model = None
        #: 仅保留轻量状态（规范 §18.2）
        self._embeds: dict[str, np.ndarray] = {}
        self._feats: dict[str, dict] = {}
        self._calib: dict | None = None

    # ------------------------------------------------------------------ #
    def load_model(self) -> None:
        """加载共享骨干（只用它的 ``embed`` 头），服务启动时一次。"""
        import torch

        from tasks._common.factory import build_shared_backbone

        root = Path(self.settings.ckpt_root).expanduser().resolve()
        path = (root / self.cfg.ckpt_rel).resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"权重路径逃逸出 checkpoint 根目录: {self.cfg.ckpt_rel}")
        if not path.is_file():
            raise FileNotFoundError(f"未找到权重：{path}（约定 {self.cfg.ckpt_rel}）")

        ck = torch.load(str(path), map_location="cpu", weights_only=False)
        model = build_shared_backbone(ck, None, self.cfg.in_channels)
        state = ck.get("model_ema") or ck.get("model") or ck
        missing, _ = model.load_state_dict(state, strict=False)
        key_missing = [k for k in missing if k.startswith(("enc", "dec", "bottleneck", "stem"))]
        if key_missing:
            raise ValueError(f"骨干关键层缺失 {len(key_missing)} 个：{key_missing[:3]}")

        dev = self.device if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        self._model = model.eval().to(dev)
        self._device = dev
        self._ckpt = str(path)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """开始新 evaluation：**必须彻底清空**（规范 §18.2）。"""
        self._embeds.clear()
        self._feats.clear()
        self._calib = None

    # ------------------------------------------------------------------ #
    def update(self, study, context) -> None:                     # noqa: ARG002
        """累积当前检查的嵌入与轻量指纹；不保留任何大对象。"""
        import torch

        if self._model is None:
            self.load_model()
        assert self._model is not None

        acc = study.accession_number
        try:
            prepared = build_volume(study, self.cfg.common_spacing)
            g = global_view(prepared.volume, size_mm=self.cfg.global_size_mm,
                            out=self.cfg.global_size)
            x = torch.from_numpy(np.ascontiguousarray(g))[None].to(self._device, torch.float32)
            with torch.inference_mode():
                out = self._model(x)
            # 立刻转 numpy-CPU 并释放张量：内存不随数据集规模增长
            emb = out["embed"].float().cpu().numpy().ravel().astype(np.float32)
            del out, x, g, prepared
        except Exception as exc:                                  # noqa: BLE001
            # 单例失败不应中断整批：留空嵌入（该例不参与配对），并记录警告
            context.warnings.append(f"goal2_duplicate: {acc} 嵌入失败（{exc}），已跳过")
            return

        n = float(np.linalg.norm(emb))
        self._embeds[acc] = emb / n if n > 1e-6 else emb
        self._feats[acc] = self._fingerprint(study)

    # ------------------------------------------------------------------ #
    def finalize(self) -> DuplicateResult:
        """产出全局重复对（规范化后返回，不写文件）。"""
        if not self._embeds:
            return DuplicateResult(pairs=())
        pairs = match_pairs(self._embeds, self._calib, topk=int(self.cfg.topk),
                            min_prob=float(self.cfg.min_prob), feats=self._feats,
                            w_fp=float(self.cfg.fp_weight),
                            per_study_k=int(self.cfg.per_study_k))
        norm = normalize_pairs(pairs, max_per_study=int(self.cfg.topk))
        return DuplicateResult(pairs=tuple(
            DuplicatePair(left_accession=p["a"], right_accession=p["b"], probability=p["prob"])
            for p in norm
        ))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _thumbnail(img: np.ndarray, size: int = 16) -> list:
        """固定尺寸缩略图（强度值，不做归一化）。

        "重复影像"的判据是**体素几乎逐点一致**，因此缩略图保留**原始强度**
        才有效；归一化会把"不同扫描仪但形态相似"的检查拉近，反而引入假阳。
        """
        from scipy.ndimage import zoom

        img = np.asarray(img, dtype=np.float32)
        if img.ndim != 3 or min(img.shape) < 2:
            return []
        f = [size / max(1, s) for s in img.shape]
        return zoom(img, f, order=1).astype(np.float32).ravel().tolist()

    @classmethod
    def _fingerprint(cls, study) -> dict:
        """轻量指纹：模态集合 + 几何 + 强度直方图 + 缩略图（不保留原始像素）。

        ⚠️ 键名与索引方式**必须**与 ``retrieval.fingerprint_similarity`` 的约定严格一致：

        | 键 | 含义 | 该函数里的用途 |
        |---|---|---|
        | ``shape_<模态>`` | 该模态体素形状 | 遍历模态、形状一致 → 弱证据 |
        | ``aff_<模态>`` | 仿射（逐位比较） | 完全一致 → 强证据 |
        | ``zoom_<模态>`` | 体素物理尺寸 | 与 shape 同为弱证据 |
        | ``hist[<模态>]`` | 强度分位数 | Bhattacharyya 系数 |
        | ``thumb[<模态>]`` | 缩略图 | 相对差异（核心证据） |
        | ``mods`` | 模态集合 | Jaccard |

        两个**历史缺陷**（都会让指纹融合静默失效，且不报任何错）：

        1. 旧实现产出 ``geom`` 列表与 ``shape_*/aff_*/zoom_*`` 键名不匹配 →
           几何证据（权重 0.45）永不参与；
        2. 旧实现用 ``series_uid`` 作 ``hist``/``mods`` 的键 —— UID 在
           **单个检查内唯一、跨检查永不重叠**，因此两例之间的键永远对不上，
           直方图与模态集合的权重（0.20 + 0.10）恒为 0，缩略图则完全没产出
           （权重 0.35 的核心证据也没参与）。
           必须按**模态名**索引，跨检查才可对齐。
        """
        from data.series_selector import select as select_series

        fp: dict[str, Any] = {"mods": [], "hist": {}, "thumb": {}}
        picked = select_series(study, ("t1c", "flair", "t2", "t1"))
        for mod, s in picked.items():
            try:
                img = np.asarray(s.image, dtype=np.float32)
                a = np.asarray(s.affine, dtype=np.float64)
            except Exception:                                     # noqa: BLE001 - 单序列失败不影响其余
                continue
            if img.ndim != 3 or img.size == 0:
                continue
            fp["mods"].append(mod)
            fp[f"shape_{mod}"] = [int(x) for x in img.shape]
            fp[f"zoom_{mod}"] = [round(float(np.linalg.norm(a[:3, i])), 4) for i in range(3)]
            # 逐位比较要求固定精度：直接存 float64 的原始值会因末位差异永不相等
            fp[f"aff_{mod}"] = [round(float(a[i, j]), 3) for i in range(4) for j in range(4)]
            flat = img.ravel()
            fp["hist"][mod] = np.percentile(flat, [5, 25, 50, 75, 95]).round(2).tolist()
            fp["thumb"][mod] = cls._thumbnail(img)
        fp["mods"] = sorted(fp["mods"])
        return fp


def build_task(settings: Settings | None = None, **kwargs: Any) -> DuplicateTask:
    """工厂入口。"""
    return DuplicateTask(settings=settings, **kwargs)
