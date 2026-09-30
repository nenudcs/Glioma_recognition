"""结构化字段的**规则推导**（与模型头融合，保证即使没训结构化头也能产出合规且有意义的字段）。

规则尽量可解释（评审友好）：
- `Enhancement`        ← 核心区掩码是否非空（T1C 上取到强化灶）
- `EnhancementPattern` ← 核心区环状度/连通域数：空心环=Ring/RimEnhancing，多灶=Multifocal，实心=Nodular
- `Necrosis`           ← 核心区内部 T1C 强度相对边缘明显偏低（非强化坏死中心）
- `CysticChange/Hemorrhage/Calcification` ← T1/T2/T1C 通道极值（弱证据，置信度低）
- `Margin` / `Lobulation` / `Morphology`  ← 周围异常区形状（球度 + 边界起伏）
- `Signal_T2WI` / `Signal_FLAIR`          ← 病灶 z-score 强度（>0.5 高 / <-0.3 低 / 其余等）
- `Location`           ← 病灶质心的 RAS 位置 → 规范 15 类
- 文本 `Conclusion`    ← 由上述字段模板化生成（可解释性）
"""
from __future__ import annotations

import numpy as np


def _cc_stats(mask: np.ndarray) -> tuple[int, float]:
    from scipy import ndimage as ndi
    lab, n = ndi.label(mask > 0)
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(sizes.max() / max(1, mask.sum()))


def _sphericity(mask: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """球度：体积 / 等效球体积；越接近 1 越规则。"""
    v = float(mask.sum()) * float(np.prod(spacing))
    if v <= 0:
        return 0.0
    from scipy import ndimage as ndi
    try:
        filled = ndi.binary_fill_holes(mask > 0)
        surf = float((mask > 0).sum() - (ndi.binary_erosion(mask > 0)).sum())
        r_eq = (3 * v / (4 * np.pi)) ** (1 / 3)
        s_ideal = 4 * np.pi * r_eq ** 2
        s_est = max(1.0, surf * (float(np.prod(spacing)) ** (2 / 3)))
        return float(np.clip(s_ideal / s_est, 0.0, 1.5))
    except Exception:  # noqa: BLE001
        return 0.0


def _ringness(core: np.ndarray) -> float:
    """环状度：1 - 实心度；空心环接近 1，实心团块接近 0。"""
    from scipy import ndimage as ndi
    m = core > 0
    if m.sum() == 0:
        return 0.0
    filled = ndi.binary_fill_holes(m)
    return float(np.clip(1.0 - m.sum() / max(1.0, filled.sum()), 0.0, 1.0))


def location_from_voxel(centroid_vox: np.ndarray, affine: np.ndarray | None, shape: tuple,
                        brain_lo: np.ndarray | None = None,
                        brain_hi: np.ndarray | None = None) -> str:
    """体素质心 → 规范 15 类位置枚举。

    **必须经 affine 转到世界坐标（RAS）**才能判断左右：早期实现直接用体素索引，
    当影像不是 RAS 存储（LPS、z 翻转等）时左右会完全反。
    归一化基准优先用脑组织包围盒（更贴近解剖比例）。
    """
    import itertools

    c = np.asarray(centroid_vox, float)
    if affine is not None:
        A = np.asarray(affine, float)
        world = (A[:3, :3] @ c) + A[:3, 3]
        corners = np.array(list(itertools.product(*[(0, s - 1) for s in shape])), float)
        cw = (A[:3, :3] @ corners.T).T + A[:3, 3]
        lo, hi = (brain_lo, brain_hi) if (brain_lo is not None and brain_hi is not None) \
            else (cw.min(0), cw.max(0))
    else:                                                          # 无 affine：假定 RAS 体素坐标
        world = c
        lo = np.zeros(3)
        hi = np.array(shape, float) - 1
    n = (world - lo) / np.maximum(hi - lo, 1e-6)                    # [0,1]^3（RAS 归一化）
    nx, ny, nz = float(n[0]), float(n[1]), float(n[2])
    side = "Right" if nx >= 0.5 else "Left"
    dx = abs(nx - 0.5) * 2.0                                        # 0=中线，1=最外侧
    if nz < 0.26:                                                   # 颅后窝：脑干 / 小脑
        if dx < 0.18 and ny < 0.60:
            return "Brainstem"
        return f"{side}Cerebellum"
    if dx < 0.30 and 0.30 <= nz < 0.62 and 0.25 <= ny <= 0.80:      # 深部近中线 → 基底节区
        return f"{side}BasalGanglia"
    if ny < 0.28 and nz < 0.62:                                     # 后部中低 → 枕叶
        return f"{side}Occipital"
    if dx > 0.42 and nz < 0.62:                                     # 外侧中低 → 颞叶
        return f"{side}Temporal"
    if ny > 0.58:                                                   # 前部 → 额叶
        return f"{side}Frontal"
    if nz >= 0.62:
        return f"{side}Parietal" if ny < 0.50 else f"{side}Frontal"
    return f"{side}Temporal"


def _brain_bbox(vol: np.ndarray):
    """脑组织（非零体素）在世界坐标下的包围盒（用于位置归一化）。"""
    nz = np.argwhere(np.abs(vol).sum(0) > 1e-3)
    if len(nz) == 0:
        return None, None
    return nz.min(0).astype(float), nz.max(0).astype(float)


def derive_rules(vol: np.ndarray, ch_names: list[str], core: np.ndarray, peri: np.ndarray,
                 spacing: tuple = (1.0, 1.0, 1.0), min_tumor_vox: int = 30,
                 affine: np.ndarray | None = None) -> dict:
    """vol: [C,D,H,W]（已 z-score）；core/peri: 公共网格上的二值掩码。"""
    core = (core > 0)
    peri = (peri > 0) | core
    n_core, _ = _cc_stats(core)
    n_peri, _ = _cc_stats(peri)
    v_core = int(core.sum())
    has_tumor = (v_core + int(peri.sum())) >= min_tumor_vox

    rules: dict = {"has_tumor": bool(has_tumor), "v_core": v_core, "v_peri": int(peri.sum()),
                   "n_cc_core": n_core, "n_cc_peri": n_peri}

    # 强化 / 强化形态
    rules["Enhancement"] = bool(v_core >= max(8, min_tumor_vox // 2))
    ring = _ringness(core) if v_core else 0.0
    if not rules["Enhancement"]:
        rules["EnhancementPattern"] = "None"
    elif n_core >= 3:
        rules["EnhancementPattern"] = "Multifocal"
    elif ring > 0.45:
        rules["EnhancementPattern"] = "Ring"          # 明显空心环节
    elif ring > 0.18:
        rules["EnhancementPattern"] = "RimEnhancing"  # 环状强化
    else:
        rules["EnhancementPattern"] = "Nodular"

    # 信号强度（z-score 空间：>0.5 高 / <-0.3 低）
    def _sig(ch: str) -> str | None:
        if ch not in ch_names or peri.sum() == 0:
            return None
        ci = ch_names.index(ch)
        m = float(vol[ci][peri].mean())
        return "High" if m > 0.5 else ("Low" if m < -0.3 else "Iso")

    rules["Signal_T2WI"] = _sig("t2") or _sig("flair")
    rules["Signal_FLAIR"] = _sig("flair") or _sig("t2")

    # 坏死：核心区内部 T1C 强度显著低于核心区边缘
    if "t1c" in ch_names and v_core > 50:
        ci = ch_names.index("t1c")
        from scipy import ndimage as ndi
        inner = ndi.binary_erosion(core, iterations=2)
        rim = core & ~inner
        if inner.sum() > 10 and rim.sum() > 10:
            rules["Necrosis"] = bool(float(vol[ci][inner].mean()) < float(vol[ci][rim].mean()) - 0.25)
    # 出血/囊变/钙化：弱证据（T1/T2/T1C 极值），仅在高置信时置 True
    for field, ch, mode, thr in (("Hemorrhage", "t1", "high", 1.6),
                                 ("CysticChange", "t2", "high", 1.8),
                                 ("Calcification", "t1c", "low", -1.6)):
        if ch in ch_names and peri.sum() > 0:
            ci = ch_names.index(ch)
            vals = vol[ci][peri]
            flag = (float(np.percentile(vals, 95)) > thr) if mode == "high" \
                else (float(np.percentile(vals, 5)) < thr)
            rules.setdefault(field, bool(flag))

    # 形状类
    sph = _sphericity(peri, spacing)
    rules["Margin"] = bool(sph > 0.45)
    rules["Lobulation"] = bool(sph <= 0.45)
    rules["Morphology"] = "Regular" if sph > 0.55 else "Irregular"

    # 位置（经 affine 转世界坐标，用脑包围盒归一化）
    if has_tumor:
        idx = np.argwhere(peri if peri.sum() else core)
        ctr = idx.mean(0) if len(idx) else np.array(peri.shape, float) / 2
        lo, hi = _brain_bbox(vol)
        if affine is not None and lo is not None:
            A = np.asarray(affine, float)
            corners = np.array([[lo[0], lo[1], lo[2]], [hi[0], hi[1], hi[2]]])
            cw = (A[:3, :3] @ corners.T).T + A[:3, 3]
            lo_w, hi_w = cw.min(0), cw.max(0)
        else:
            lo_w = hi_w = None
        rules["Location"] = location_from_voxel(ctr, affine, tuple(peri.shape), lo_w, hi_w)
    else:
        rules["Location"] = "NA"
    return rules


def special_heuristics(vol: np.ndarray, cfg: dict) -> tuple[float, float]:
    """目标一/二的**可解释启发式兜底**（模型头未训练或欠拟合时使用）。

    返回 ``(p_fake_or_nonhuman, p_stitched)``，两者都在 [0,1]。

    - 假人体/非人体：组织体素占比过低（视野异常、伪影），或层间相关性整体极低
      （纯噪声 / 非解剖结构 / 条纹）；
    - 拼接影像：z 轴层间相关性在某处**骤降**（不同检查段拼接），或层间距不均匀。

    阈值刻意保守（宁可漏报也不误报真人体），避免伤及基础考核项的阴性样本。
    """
    sc = (cfg.get("special") or {}) if cfg else {}
    v_all = np.asarray(vol, float)
    if v_all.ndim != 4 or v_all.size == 0:
        return 0.0, 0.0

    tissue = float((np.abs(v_all) > 1.5).mean())                  # 组织体素占比
    v = v_all.mean(0)
    d = v.shape[0]
    corrs = []
    for k in range(d - 1):
        a, b = v[k].ravel(), v[k + 1].ravel()
        if a.std() < 1e-5 or b.std() < 1e-5:
            corrs.append(0.0)
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    corrs = np.asarray(corrs) if len(corrs) else np.asarray([1.0])
    # **关键修正**：首尾各 10% 的层面脑组织很少，层间相关性天然偏低，
    # 早期实现直接取全序列 min 会把正常影像误判为"拼接"（实测给出 0.5 的假阳性）。
    # 这里裁剪掉边缘层后再统计，排除边缘效应。
    margin = max(1, int(len(corrs) * 0.10))
    inner = corrs[margin: len(corrs) - margin] if len(corrs) > 3 * margin else corrs
    med = float(np.median(inner))
    jump = float(np.min(np.diff(inner))) if len(inner) > 1 else 0.0
    min_inner = float(inner.min()) if len(inner) else 1.0

    p_fake = 0.0
    cov_thr = float(sc.get("fake_body_min_brain_ratio", 0.05))
    if tissue < cov_thr:
        p_fake = max(p_fake, float(np.clip(1.0 - tissue / max(1e-6, cov_thr), 0, 1)) * 0.9)
    if med < 0.2:                                                 # 层间几乎不相关 → 非解剖影像
        p_fake = max(p_fake, float(np.clip((0.2 - med) / 0.2, 0, 1)) * 0.8)

    p_stitch = 0.0
    st_thr = float(sc.get("stitch_edge_ratio", 0.35))
    if jump < -st_thr:                                            # 内部相关性骤降 → 拼接边界
        p_stitch = max(p_stitch, float(np.clip(-jump / (2.0 * st_thr), 0, 1)))
    if med > 0.6 and min_inner < 0.05:                            # 极低相关性层（更保守）
        p_stitch = max(p_stitch, 0.5)
    return float(np.clip(p_fake, 0.0, 1.0)), float(np.clip(p_stitch, 0.0, 1.0))


def _pv(pred: dict, key: str, default=None):
    """取规范字段的 predicted 值（字段可能是 {"predicted":…} 对象，也可能是标量）。"""
    v = pred.get(key, default)
    return v.get("predicted", default) if isinstance(v, dict) else v


def _present(pred: dict, key: str) -> bool:
    v = pred.get(key)
    return bool(v.get("present")) if isinstance(v, dict) else bool(v)


#: 结论文本用的中文映射。Prediction 里各字段是**英文枚举**，直接拼进中文会
#: 出现"左侧颞叶Irregular占位，RimEnhancing强化"这类中英混杂，因此统一转中文。
_LOC_CN = {"RightFrontal": "右侧额叶", "LeftFrontal": "左侧额叶", "RightTemporal": "右侧颞叶",
           "LeftTemporal": "左侧颞叶", "RightParietal": "右侧顶叶", "LeftParietal": "左侧顶叶",
           "RightOccipital": "右侧枕叶", "LeftOccipital": "左侧枕叶", "Brainstem": "脑干",
           "RightCerebellum": "右侧小脑半球", "LeftCerebellum": "左侧小脑半球",
           "RightBasalGanglia": "右侧基底节区", "LeftBasalGanglia": "左侧基底节区",
           "Other": "颅内其他部位", "NA": "颅内"}
_MORPH_CN = {"Regular": "形态规则", "Irregular": "形态不规则", "NA": ""}
_ENH_CN = {"None": "未见明确强化", "Ring": "环形强化", "RimEnhancing": "边缘强化",
           "Nodular": "结节状强化", "GroundGlass": "磨玻璃样强化",
           "Gyriform": "脑回样强化", "Multifocal": "多灶性强化", "Other": "其他形式强化"}


def make_conclusion(accession: str, pred: dict) -> str:
    """模板化结论（可解释；规范只要求是字符串，这里输出专业中文描述）。"""
    grade = _pv(pred, "WHO_Grade")
    loc = _pv(pred, "Location", "NA") or "NA"
    loc_cn = _LOC_CN.get(loc, loc)
    tp = pred.get("TumorProbability", 0.0)
    if tp is not None and tp < 0.5:                               # 阴性：非肿瘤性病变
        return f"{loc_cn}异常信号，考虑非肿瘤性病变（如脑梗死/脓肿），未见明确强化。"
    morph = _MORPH_CN.get(_pv(pred, "Morphology") or "", "")
    enh = _ENH_CN.get(_pv(pred, "EnhancementPattern") or "", "")
    nec = "伴坏死" if _present(pred, "Necrosis") else ""
    hem = "伴出血" if _present(pred, "Hemorrhage") else ""
    gtxt = f"（WHO {grade}级）" if grade else ""
    head = f"{loc_cn}{morph}占位"
    feats = f"{enh}{nec}{hem}"                                    # 可能为空 → 避免出现"，，"
    return f"{head}，{feats}，考虑脑胶质瘤{gtxt}。" if feats else f"{head}，考虑脑胶质瘤{gtxt}。"
