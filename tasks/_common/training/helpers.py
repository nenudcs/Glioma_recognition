"""训练辅助：模型构建、数据管线接入、验证函数。

分层原则（对应规范 §5.1 对 ``[研发]`` 文件的态度）：

- **模型构建**在中央仓库内完成，保证"训练出来的结构与 ``task.py`` 能加载的结构"
  同源——这一步绝不能各自实现；
- **数据管线**默认接入算法工程 ``glioma_track4`` 的成熟实现（缓存、重采样、
  强增强、配对采样都在那里），因为那是**研发侧**的事，规范允许各 Goal 在自己的
  工作区独立训练、不强制共享训练平台；
- 若中央仓库需要完全自包含，把 ``_load_track4()`` 换成内部实现即可，
  训练入口与损失定义无需改动。

这样既保证"训练结构 == 推理结构"这一**必须一致**的契约，又不把庞大的
数据工程塞进比赛运行仓库。
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _load_track4():
    """定位并导入算法工程 ``glioma_track4``（研发侧数据管线）。"""
    candidates = [
        os.environ.get("GLIOMA_TRACK4_ROOT"),
        str(Path(__file__).resolve().parents[3].parent / "glioma_track4"),
        "/2026aicompetition/workspace/glioma_track4",
        "../glioma_track4",
    ]
    for c in candidates:
        if not c:
            continue
        root = Path(c).expanduser().resolve()
        if (root / "src" / "data" / "dataset.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    raise FileNotFoundError(
        "找不到算法工程 glioma_track4（含数据管线）。"
        "请设置 GLIOMA_TRACK4_ROOT，或把两个工程放在同级目录。"
    )


def build_model_for_training(raw: dict[str, Any]):
    """构建共享骨干（结构与 ``task.py`` 加载的完全同源）。"""
    from tasks._common.mednext import MedNeXtNet

    mc = raw.get("model") or {}
    labels_cfg = raw.get("labels") or {}
    cls_spec = labels_cfg.get("cls_spec") or []
    if not cls_spec and labels_cfg.get("fields"):
        # 由 labels.yaml 的字段定义推出 (key, n_classes)
        cls_spec = [(f["key"], 1 if f["type"] == "binary" else len(f["classes"]))
                    for f in labels_cfg["fields"]]

    model = MedNeXtNet(
        in_ch=int(mc.get("in_channels", 4)),
        base=int(mc.get("base", 32)),
        depth=int(mc.get("depth", 4)),
        cls_spec=cls_spec,
        blocks_per_stage=int(mc.get("blocks_per_stage", 2)),
        k=int(mc.get("k", 3)),
        expand=int(mc.get("expand", 2)),
        aniso_z=bool(mc.get("aniso_z", False)),
        max_ch=int(mc.get("max_ch", 320)),
    )
    # 供 _save 读取结构超参。
    # 用 SimpleNamespace 而非 ``type("MC", (), {...})()``：后者把配置放在
    # **类属性**上，任何"按实例取属性"的序列化（``vars(obj)``）都会得到空
    # 字典，于是权重里的 model_cfg 丢掉 base/depth 等结构超参，推理侧只能
    # 按默认值重建网络——形状不符的层会被 ``strict=False`` 静默跳过，
    # 变成"加载成功但精度崩坏"这种最难查的失效。
    model.cfg = SimpleNamespace(**{**mc, "in_ch": int(mc.get("in_channels", 4))})
    model.arch_name = "mednext"
    return model, cls_spec


def _special_targets(case: dict) -> tuple[float, float]:
    """返回检查级 ``(fake, stitched)`` 标签。

    规则与研发侧 ``glioma_goals`` 的 ``goal1_authenticity/dataset.py`` 与
    ``goal2_stitched/dataset.py`` **完全一致**（先看 ``label.json``，
    再看目录名关键词）。两条训练路径必须给出同样的标签，否则同一个
    special 头在"单目标训练"和"统一训练"下会学到互相矛盾的目标。
    """
    import json as _json

    fake = stitched = 0.0
    lj = Path(case.get("dir") or "") / "label.json"
    if lj.is_file():
        try:
            d = _json.loads(lj.read_text(encoding="utf-8"))
            fake = float(int(d.get("fake", d.get("not_human", 0))))
            stitched = float(int(d.get("stitched", d.get("composition", 0))))
            return fake, stitched
        except Exception:                                         # noqa: BLE001
            pass
    name = str(case.get("accession") or "").lower()
    fake = 1.0 if any(k in name for k in ("fake", "nothuman", "not_human")) else 0.0
    stitched = 1.0 if any(k in name for k in ("comp", "stitch", "拼接")) else 0.0
    return fake, stitched


# ``helpers`` 只在**训练**路径被导入（推理侧不碰它），数据管线与损失本就
# 依赖 torch，因此这里在模块级引入是安全的；放在文件中部是为了保持
# "标准库在前、重型依赖在后"的既有顺序。
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
from torch.utils.data import Dataset as _TorchDataset             # noqa: E402


class _SpecialSupervised(_TorchDataset):
    """在 ``GliomaDataset`` 之上补齐**被它漏掉的两路监督**。

    ``GliomaDataset`` 只产出分割目标与 14 个结构化字段，缺两样东西：

    1. **``special_target`` / ``special_mask``**（目标一、目标二-A）
       —— 缺了它，``compute_losses`` 的 special 分支静默跳过，
       special 头从未收到梯度，上线却照样输出 [0,1] 概率。
    2. **配对样本**（目标二-B 的 ``pair`` / ``image_b``）
       —— 缺了它，embed 头同样收不到梯度，重复影像的相似度全凭随机初始化。

    注意两处细节：

    * ``special`` 是 2 通道（0=假人体/非人体，1=拼接），且两类正样本
      **不能互为负样本**，所以按通道分别给 mask；
    * ``pair`` 必须是 **0 维 tensor**。若给 ``[1]``，collate 后成 ``[B,1]``，
      与 ``sim`` 的 ``[B]`` 广播会得到 ``[B,B]`` —— 损失仍能算、不会报错，
      但梯度完全错了。正对用同检查的两种强度扰动（"重复影像"的本质就是
      同一检查的两种呈现），负对取另一检查。
    """

    def __init__(self, base, cases: list[dict], pair_prob: float = 0.5,
                 seed: int = 42) -> None:
        self.base, self.cases = base, cases
        self.pair_prob = float(pair_prob)
        #: 用独立的 ``random.Random`` 而非全局随机源：DataLoader 的多个 worker
        #: 会各自 fork，用全局源会让"正/负对"的采样在不同 worker 之间串味，
        #: 也让训练不可复现。
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> dict:
        item = dict(self.base[i])
        case = self.cases[i % len(self.cases)]

        fake, stitched = _special_targets(case)
        item["special_target"] = torch.tensor([fake, stitched], dtype=torch.float32)
        item["special_mask"] = torch.ones(2, dtype=torch.float32)

        whole = item.get("whole")
        if self.pair_prob > 0 and whole is not None and len(self.cases) > 1:
            # ``pair`` / ``image_b`` 必须**无条件**写入：``_collate`` 以第一个样本
            # 的键集合为准，若某样本少一个键，整个 batch 就会缺这一路监督，
            # 而损失分支只会"静默跳过"（日志里才看得到那条告警）。
            if self.rng.random() < 0.5:                           # 正对：同检查两种呈现
                scale = 1.0 + self.rng.uniform(-0.08, 0.08)
                shift = self.rng.uniform(-0.08, 0.08)
                item["image_b"] = whole * scale + shift
                item["pair"] = torch.tensor(1.0)                  # 0 维，见类文档
            else:                                                 # 负对：另一检查
                neg = None
                for _ in range(3):                                # 最多试 3 次以避开"选到自己"
                    j = self.rng.randrange(len(self.cases))
                    if self.cases[j]["accession"] == case["accession"]:
                        continue
                    try:
                        # 用**完整取样本**而不是 ``base._whole(case)``：
                        # 后者在 ``cache_dir`` 缺失时会抛异常（它不像
                        # ``_WholeViewMixin`` 那样有兜底），一旦被吞掉，
                        # 配对监督就永远建立不起来——正是本次踩到的坑。
                        neg = self.base[j].get("whole")
                    except Exception:                             # noqa: BLE001
                        neg = None
                    if neg is not None:
                        break
                if neg is not None:
                    item["image_b"] = neg
                    item["pair"] = torch.tensor(0.0)
                else:                                             # 拿不到负对 → 退化为正对
                    item["image_b"] = whole.clone()
                    item["pair"] = torch.tensor(1.0)
        return item


def build_datasets(raw: dict[str, Any], fold: int = 0, limit: int = 0,
                   seed: int = 42, patch: tuple = (96, 96, 96)):
    """返回 ``(train_ds, val_ds)`` —— **不包 DataLoader**。

    各 Goal 的 ``dataset.py`` 通过它取数据集，``build_dataloaders`` 在其上包
    DataLoader。把这段抽成公共实现，是为了避免"Goal 侧再写一遍"：那会让
    各 Goal 的增强与监督信号悄悄分叉，而分叉不报错、只让实验不再可比。

    Args:
        raw: 该 Goal 的 ``config.yaml`` 内容（本函数只用到 ``data`` 段）。
        fold: 折号，对应 ``folds.json`` 的 key。
        limit: 只用前 N 例（冒烟用；0 = 全量）。
        seed: 数据增强与配对采样的随机种子。
        patch: 训练 patch 尺寸。

    Raises:
        FileNotFoundError: 缺折划分文件（需先跑探针与折划分脚本）。
    """
    import json

    root = _load_track4()
    ds_mod = __import__("src.data.dataset", fromlist=["*"])
    utils = __import__("src.utils.config", fromlist=["*"])

    paths = utils.load_paths()
    pre = utils.load_config("preprocess.yaml")
    manifest = utils.resolve(paths["manifest"])
    folds_path = utils.resolve(paths["folds"])
    if not Path(folds_path).is_file():
        raise FileNotFoundError(f"缺折划分 {folds_path}（先运行探针与折划分脚本）")

    with open(manifest, encoding="utf-8") as f:
        man = json.load(f)
    with open(folds_path, encoding="utf-8") as f:
        folds = json.load(f)
    by_acc = {c["accession"]: c for c in man["cases"]}
    split = folds[str(fold)]
    train_cases = [by_acc[a] for a in split["train"] if a in by_acc]
    val_cases = [by_acc[a] for a in split["val"] if a in by_acc]
    if limit:
        train_cases, val_cases = train_cases[:limit], val_cases[:limit]

    patch = tuple(patch)
    # 包一层补 special 与配对监督：缺了它们，special 头与 embed 头会一直
    # 收不到梯度（损失分支被静默跳过）。验证集不产配对样本——验证阶段
    # 不需要 embed 损失，且随机配对会让验证指标不可复现。
    train_ds = _SpecialSupervised(
        ds_mod.GliomaDataset(train_cases, train=True, patch=patch,
                             pre_cfg=pre, seed=seed), train_cases,
        pair_prob=0.5, seed=seed)
    val_ds = _SpecialSupervised(
        ds_mod.GliomaDataset(val_cases, train=False, patch=patch,
                             pre_cfg=pre, seed=seed), val_cases,
        pair_prob=0.0, seed=seed + 1)
    print(f"[data] 数据源 {root.name}: fold={fold} train={len(train_cases)} "
          f"val={len(val_cases)} patch={patch}"
          f"（含 special(fake/stitched) + 配对监督）", flush=True)
    return train_ds, val_ds


def build_dataloaders(cfg, fold: int, raw: dict[str, Any], limit: int = 0):
    """接入算法工程的数据管线，产出 (train_loader, val_loader)。"""
    import torch

    # 数据集构造统一走 build_datasets（Goal 侧的 dataset.py 也调它），
    # 避免"统一训练"与"Goal 侧"两套增强/监督信号悄悄分叉。
    train_ds, val_ds = build_datasets(raw, fold=fold, limit=limit,
                                      seed=cfg.seed, patch=tuple(cfg.patch))

    def _collate(batch):
        out = {}
        for k in batch[0]:
            vals = [b[k] for b in batch]
            out[k] = torch.stack(vals) if hasattr(vals[0], "shape") else vals
        # 配对分支：Dataset 用 ``a``/``b`` 命名，引擎期望 image_b
        if "a" in out:
            out["image"], out["image_b"] = out.pop("a"), out.pop("b")
            out["pair"] = out.pop("label")
            out.pop("acc_a", None)
            out.pop("acc_b", None)
        return out

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=int(cfg.num_workers), collate_fn=_collate,
        drop_last=True, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=max(1, cfg.batch_size), shuffle=False,
        num_workers=max(1, cfg.num_workers // 2), collate_fn=_collate,
    )
    return train_loader, val_loader


def build_val_fn(goal: str, val_loader, raw: dict[str, Any]):
    """按 Goal 返回验证函数；无验证逻辑时返回 None（用训练损失作为选择依据）。"""
    if val_loader is None:
        return None

    def _val(model) -> dict:
        """轻量验证：在固定 patch 上算 Dice（全图评估请用 16_finalize.sh）。"""
        import numpy as np
        import torch

        model.eval()
        dices = {"core": [], "peri": []}
        with torch.inference_mode():
            for i, batch in enumerate(val_loader):
                if i >= 20:                                      # 限批，避免拖慢训练
                    break
                img = batch["image"].to(next(model.parameters()).device)
                tgt = batch["target"]
                if tgt.dim() == 4:
                    tgt = tgt.unsqueeze(1)
                out = model(img)
                p = torch.sigmoid(out["seg"]).float().cpu().numpy()
                g = tgt.cpu().numpy()
                for c, key in enumerate(("core", "peri")):
                    if c >= p.shape[1] or g.shape[1] <= c:
                        continue
                    pb, gb = p[:, c] > 0.5, g[:, c] > 0.5
                    inter = float((pb & gb).sum())
                    den = float(pb.sum() + gb.sum())
                    dices[key].append(2 * inter / den if den > 0 else 1.0)
        model.train()
        vals = {f"dice_{k}": float(np.mean(v)) for k, v in dices.items() if v}
        vals["score"] = sum(vals.values()) / max(1, len(vals))
        return vals

    return _val
