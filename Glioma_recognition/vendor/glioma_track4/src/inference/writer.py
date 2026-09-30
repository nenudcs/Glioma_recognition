"""答案写出与校验（严格按《赛事开发规范（赛道四）》）。

产出目录：
    {answer_root}/{evaluation_id}/
        duplicate_pairs.jsonl                 # 重复影像对（每例 ≤200 对）
        {AccessionNumber}/
            prediction.json
            {SeriesUid}/
                {SeriesUid}_core.nii.gz       # 任务A：T1增强核心区（**仿射/维度 == T1C 原图**）
                {SeriesUid}_flair.nii.gz      # 任务B：FLAIR/T2 总异常区（**仿射/维度 == FLAIR/T2 原图**）
                attention_{AccessionNumber}.nii.gz

强制规范（否则该例分割指标直接 0 分）：
1. 掩码体素值严格 0/1；
2. 掩码 affine 与空间维度必须与对应模态原图一致；
3. 掩码文件必须与 ``prediction.json`` 的 ``SegmentationMaskURI`` 一致。

**关键修正**：早期实现把源影像 header 整体复用（``Nifti1Image(arr, ref.affine, ref.header)``），
会继承 ``scl_slope/scl_inter``（DICOM 转换常见非 1 斜率）→ 掩码读出值 ≠ 0/1 →
该例分割直接 0 分。现在只保留几何（affine + qform/sform code + 单位），
并在写盘后**回读复核** 0/1、shape、affine。
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# 掩码回采样与写出
# --------------------------------------------------------------------------- #
def _resample_binary_to_grid(binary: np.ndarray, src_affine: np.ndarray, dst_shape: tuple,
                             dst_affine: np.ndarray) -> np.ndarray:
    """公共网格二值掩码 → 目标网格（最近邻，保证仍为 0/1）。"""
    import nibabel as nib
    from nibabel.processing import resample_from_to

    src = nib.Nifti1Image(np.asarray(binary, dtype=np.uint8),
                          np.asarray(src_affine, float))
    ref = nib.Nifti1Image(np.zeros(tuple(dst_shape), dtype=np.uint8),
                          np.asarray(dst_affine, float))
    out = resample_from_to(src, ref, order=0, mode="constant", cval=0.0)
    return (np.asanyarray(out.dataobj) > 0).astype(np.uint8)


def write_mask(mask_arr: np.ndarray, ref_image_path: str, out_path: str,
               post_check: bool = True) -> None:
    """以参考影像的 **affine/维度** 原样写出二值掩码（header 只保留几何，杜绝斜率污染）。"""
    import nibabel as nib

    ref = nib.load(ref_image_path)
    arr = (np.asarray(mask_arr) > 0).astype(np.uint8)
    assert tuple(arr.shape) == tuple(ref.shape), f"掩码维度 {arr.shape} != 参考影像 {ref.shape}"

    img = nib.Nifti1Image(arr, np.asarray(ref.affine, float))
    hdr = img.header
    hdr.set_data_dtype(np.uint8)
    hdr["scl_slope"] = 1.0                                        # 显式 1.0，避免任何缩放
    hdr["scl_inter"] = 0.0
    try:
        hdr.set_xyzt_units(*ref.header.get_xyzt_units())
    except Exception:                                             # noqa: BLE001
        pass
    qc = int(ref.header["qform_code"]) or 1
    sc = int(ref.header["sform_code"]) or 1
    img.set_qform(ref.affine, code=qc)
    img.set_sform(ref.affine, code=sc)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(img, out_path)

    if post_check:                                                # 写后回读复核（硬约束）
        chk = nib.load(out_path)
        a = np.asanyarray(chk.dataobj)
        u = np.unique(a)
        if not set(u.tolist()) <= {0, 1}:
            raise RuntimeError(f"掩码 {out_path} 体素值非 0/1：{u[:5].tolist()}")
        if tuple(chk.shape) != tuple(ref.shape):
            raise RuntimeError(f"掩码 {out_path} 维度 {chk.shape} != 参考 {ref.shape}")
        if not np.allclose(chk.affine, ref.affine, atol=1e-3):
            raise RuntimeError(f"掩码 {out_path} affine 与参考影像不一致")


# --------------------------------------------------------------------------- #
# prediction.json 组装
# --------------------------------------------------------------------------- #
def _norm(probs: list[float]) -> list[float]:
    """归一化到 sum=1，并做**残差补偿**保证求和精度。

    团队串联工程的 ``OutputValidator`` 要求 ``probabilities`` 之和满足
    ``isclose(total, 1.0, abs_tol=1e-4)``；若逐项 ``round(..., 4)`` 后直接返回，
    4 个类别的累积舍入误差最大可达 2e-4，会**直接判定输出非法**。
    这里把残差补到最大项上，使求和精确为 1.0。
    """
    finite = [float(p) if np.isfinite(p) else 0.0 for p in probs]
    s = float(sum(finite))
    if s <= 0:
        n = max(1, len(finite))
        vals = [round(1.0 / n, 4)] * n
    else:
        vals = [round(max(0.0, p) / s, 4) for p in finite]
    if vals:
        residual = round(1.0 - float(sum(vals)), 4)
        if residual:
            j = int(np.argmax(vals))
            vals[j] = round(max(0.0, min(1.0, vals[j] + residual)), 4)
    return vals


def _multiclass(classes: list[str], p, rule_val, output_classes: list | None = None) -> dict:
    """多分类字段：``{predicted, probabilities}``；保证 predicted 与概率自洽且归一。

    ``output_classes``：对外暴露的枚举子集（默认等于 ``classes``）。
    用于"模型训练时多一个兜底类、但规范示例只列了部分取值"的场景
    （如 Morphology 训练含 ``NA``，而规范示例仅 ``Regular``/``Irregular``）：
    **模型头保持不变（checkpoint 兼容），只在输出层裁剪并重新归一**，
    既严格对齐示例枚举，又不损失数值。
    """
    keep = [str(c) for c in (output_classes or classes)]
    keep = [c for c in keep if c in classes] or [str(c) for c in classes]

    if p is not None and np.size(p) == len(classes):
        probs = [float(x) for x in np.asarray(p, float).ravel()]
        pick = classes[int(np.argmax(probs))]
        if rule_val in classes and probs[classes.index(pick)] < 0.5:
            pick = str(rule_val)                                  # 低置信时规则接管
        probs = _norm(probs)
        if keep != [str(c) for c in classes]:                     # 裁剪到规范示例枚举
            idx = [classes.index(c) for c in keep]
            sub = _norm([probs[i] for i in idx])
            j = keep.index(pick) if pick in keep else int(np.argmax(sub))
            top = max(sub)
            if sub[j] < top - 1e-9:
                sub = [min(v, top) for v in sub]
                sub[j] = top
                sub = _norm(sub)
            return {"predicted": keep[j], "probabilities": dict(zip(keep, sub))}
        j, top = classes.index(pick), max(probs)
        if probs[j] < top - 1e-9:                                 # 令 predicted 与 probabilities 一致
            probs = [min(v, top) for v in probs]
            probs[j] = top
            probs = _norm(probs)                                  # 抬升后必须重新归一（SUM=1 硬约束）
        return {"predicted": pick, "probabilities": dict(zip(classes, probs))}
    if rule_val in keep:
        probs = _norm([3.0 if c == rule_val else 0.1 for c in keep])
        return {"predicted": str(rule_val), "probabilities": dict(zip(keep, probs))}
    return {"predicted": keep[-1],
            "probabilities": dict(zip(keep, _norm([1.0] * len(keep))))}


def assemble_prediction(accession: str, fields_cfg: list, probs: dict, rules: dict) -> dict:
    """把模型概率与规则结论融合成规范 prediction.json。"""
    pred: dict[str, Any] = {}
    tum_p = float(rules.get("tumor_prob_model", 0.0))
    tum_p = float(np.clip(0.8 * tum_p + 0.2 * (1.0 if rules.get("has_tumor") else 0.0), 0.0, 1.0))

    for f in fields_cfg:
        key = f["key"]
        p = probs.get(key)
        if key == "TumorProbability":
            pred[key] = round(tum_p, 4)
            continue
        if f["type"] == "binary":
            pm = float(np.asarray(p, float).ravel()[0]) if p is not None and np.size(p) == 1 else None
            rule_v = rules.get(key)
            if pm is None:
                val = bool(rule_v) if rule_v is not None else False
                conf = 0.75 if rule_v is not None else 0.5
            elif rule_v is None:
                val, conf = pm > 0.5, pm
            else:
                if pm >= 0.5 and bool(rule_v) == (pm > 0.5):
                    val, conf = pm > 0.5, min(0.99, 0.5 + 0.5 * abs(pm - 0.5) * 2)
                elif pm < 0.5 and not bool(rule_v):
                    val, conf = False, min(0.5, 1.0 - pm)
                else:                                             # 模型与规则冲突 → 取模型置信度高者
                    val = pm > 0.5 if abs(pm - 0.5) > 0.2 else bool(rule_v)
                    conf = max(pm, 1.0 - pm) * 0.9
            conf = float(np.clip(conf, 0.01, 0.99))
            val = bool(val) if conf >= 0.5 else bool(val)
            if key == "Margin":
                pred[key] = {"clear": val, "MarginClearProbability": round(conf, 4)}
            else:
                pred[key] = {"present": val, f"{key}Probability": round(conf, 4)}
        else:
            classes = [str(c) for c in f["classes"]]
            if key == "WHO_Grade" and tum_p < 0.5:
                pred[key] = {"predicted": None, "probabilities": {c: 0.0 for c in classes}}
                continue
            cat = _multiclass(classes, p, rules.get(key), f.get("output_classes"))
            if key == "WHO_Grade" and cat["predicted"] is not None:
                # 规范示例中 predicted 为**数字 4**（probabilities 的键仍为字符串 "1".."4"），
                # 且规范说明"WHO_Grade：predicted 为预测分级（1-4）"。
                # 其余类别字段（Morphology/EnhancementPattern/Signal_*）保持字符串。
                try:
                    cat["predicted"] = int(str(cat["predicted"]))
                except (TypeError, ValueError):
                    pass
            pred[key] = cat
    return pred


# --------------------------------------------------------------------------- #
# 病例答案写出
# --------------------------------------------------------------------------- #
def case_fingerprint(case: dict) -> dict:
    """手工指纹（重复影像辅助特征）：模态几何 + 强度直方图 + 空间缩略图。

    学习式嵌入负责语义相似度；手工指纹补足"同一检查被重复上传/前后复查"
    这类几何与强度几乎完全一致的强信号。

    **关键**：直方图/缩略图必须**按模态名索引**（不能用 list 顺序，
    否则不同病例的模态顺序不同会造成错配——实测会把 AUC 从 0.9+ 拉到 0.04）。
    """
    fp: dict[str, Any] = {"mods": sorted((case.get("images") or {}).keys()),
                          "hist": {}, "thumb": {}}
    for mod, meta in sorted((case.get("images") or {}).items()):
        try:
            import nibabel as nib
            img = nib.load(meta["path"])
            a = np.asanyarray(img.dataobj).astype(np.float32)
            flat = a.ravel()[::37]
            h, _ = np.histogram(flat, bins=8, range=(float(np.percentile(flat, 1)),
                                                     float(np.percentile(flat, 99)) + 1e-6))
            h = h / max(1, h.sum())
            fp["hist"][mod] = [round(float(x), 4) for x in h]
            # 空间缩略图（16³）：保留空间信息，对"同一数据重采样/重复上传"极敏感
            step = [max(1, int(s // 16)) for s in a.shape]
            t = np.asarray(a[::step[0], ::step[1], ::step[2]][:16, :16, :16], np.float32)
            fp["thumb"][mod] = np.round(np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0),
                                        3).tolist()
            fp[f"shape_{mod}"] = list(img.shape)
            fp[f"zoom_{mod}"] = [round(float(z), 3) for z in img.header.get_zooms()[:3]]
            fp[f"aff_{mod}"] = [round(float(x), 3) for x in np.asarray(img.affine).ravel()]
        except Exception:                                         # noqa: BLE001
            continue
    return fp


def write_case(answer_root: str, accession: str, masks: dict, pred_payload: dict,
               series_meta: dict, cfg: dict) -> dict:
    """写出一例的 prediction.json 与两个掩码。

    masks: ``{"core": (二值数组, 公共网格affine), "peri": (...)}``
    series_meta: ``{"core": {"series_uid","path","affine","shape"}, "peri": {...}}``
    """
    acc_dir = os.path.join(answer_root, str(accession))
    os.makedirs(acc_dir, exist_ok=True)

    mask_uri: dict[str, str] = {}
    for role, out_name in (("core", "core"), ("peri", "flair")):
        meta = series_meta.get(role)
        if meta is None or role not in masks:
            continue
        binary, common_aff = masks[role]
        arr = _resample_binary_to_grid(binary, common_aff, tuple(meta["shape"]), meta["affine"])
        uid = str(meta["series_uid"])
        sub = os.path.join(acc_dir, uid)
        out_p = os.path.join(sub, f"{uid}_{out_name}.nii.gz")
        write_mask(arr, meta["path"], out_p)
        mask_uri[out_name] = f"./{uid}/{uid}_{out_name}.nii.gz"

    if "attention" in masks and (series_meta.get("peri") or series_meta.get("core")):  # 可选注意力图
        meta = series_meta.get("peri") or series_meta["core"]
        binary, common_aff = masks["attention"]
        try:
            att = _resample_binary_to_grid((np.asarray(binary) * 255).astype(np.uint8),
                                           common_aff, tuple(meta["shape"]), meta["affine"])
            att = (np.asarray(att) > 0).astype(np.uint8) * 255
            uid = str(meta["series_uid"])
            att_p = os.path.join(acc_dir, uid, f"attention_{accession}.nii.gz")
            import nibabel as nib
            ref = nib.load(meta["path"])
            img = nib.Nifti1Image(att.astype(np.uint8), np.asarray(ref.affine, float))
            img.header.set_data_dtype(np.uint8)
            os.makedirs(os.path.dirname(att_p), exist_ok=True)
            nib.save(img, att_p)
            pred_payload.setdefault("Interpretation", {})["AttentionMapURI"] = \
                f"./{uid}/attention_{accession}.nii.gz"
        except Exception:                                         # noqa: BLE001
            pass

    pred_payload["AccessionNumber"] = accession
    pred_payload["SegmentationMaskURI"] = mask_uri
    with open(os.path.join(acc_dir, "prediction.json"), "w", encoding="utf-8") as f:
        json.dump(pred_payload, f, ensure_ascii=False, indent=1)
    return pred_payload


def write_fallback_case(answer_root: str, case: dict, cfg: dict,
                        reason: str = "inference_failed") -> dict:
    """兜底：推理异常时仍写出**几何合规**的全 0 掩码与合法 prediction.json。

    该例分割得 0 分不可避免，但保证"每个 AccessionNumber 都有 prediction.json"
    这一硬性完整性要求，避免其他字段（目标一~四）也被判缺失。
    """
    import nibabel as nib

    from ..data.dataset import pick_series
    from ..utils.config import load_config

    accession = str(case.get("accession"))
    acc_dir = os.path.join(answer_root, accession)
    os.makedirs(acc_dir, exist_ok=True)
    fields = load_config("labels.yaml")["fields"]
    pred: dict[str, Any] = {"AccessionNumber": accession, "IsNotHumanBodyProb": 0.0,
                            "IsStitchedProb": 0.0, "ProcessingTime_ms": 0,
                            "SegmentationMaskURI": {}, "Fallback": reason}
    pred["Prediction"] = assemble_prediction(accession, fields, {},
                                             {"has_tumor": False, "tumor_prob_model": 0.0})
    mask_uri: dict[str, str] = {}
    picked = {}
    try:
        picked = pick_series(case, cfg)
    except Exception:                                             # noqa: BLE001
        picked = {}
    for role, ch, name in (("core", "t1c", "core"), ("peri", "flair", "flair")):
        meta = picked.get(ch)
        if meta is None:
            continue
        try:
            ref = nib.load(meta["path"])
            uid = str(meta.get("series_uid") or os.path.basename(os.path.dirname(meta["path"])))
            arr = np.zeros(tuple(ref.shape), dtype=np.uint8)
            out_p = os.path.join(acc_dir, uid, f"{uid}_{name}.nii.gz")
            write_mask(arr, meta["path"], out_p)
            mask_uri[name] = f"./{uid}/{uid}_{name}.nii.gz"
        except Exception:                                         # noqa: BLE001
            continue
    pred["SegmentationMaskURI"] = mask_uri
    # 结论由已组装的 Prediction 生成，保持 Interpretation 与 Prediction 自洽。
    # 注意：这里必须整段包裹异常——兜底函数本身**绝不能再抛异常**，
    # 否则该例连 prediction.json 都写不出来，"每例必须有答案"的硬性要求会直接失败。
    try:
        from .structured import make_conclusion
        pred["Interpretation"] = {"Conclusion": make_conclusion(accession, pred["Prediction"])}
    except Exception:                                             # noqa: BLE001
        pred["Interpretation"] = {"Conclusion": "推理异常，已输出兜底结果。"}
    with open(os.path.join(acc_dir, "prediction.json"), "w", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False, indent=1)
    return pred


def write_duplicate_pairs(answer_root: str, pairs: list[dict],
                          fallback_accessions: list[str] | None = None) -> str:
    """写 ``duplicate_pairs.jsonl``（规范：每行 ``{"StudyUID","StudyUID_dup","PairProb"}``）。

    规范要求"提交文件必须至少包含一行有效记录"。``pairs`` 为空时**不能写空文件**
    （会被判为无效提交），因此若提供了 ``fallback_accessions``，就写一条自比对
    占位行来保护文件格式。
    """
    os.makedirs(answer_root, exist_ok=True)
    p = os.path.join(answer_root, "duplicate_pairs.jsonl")
    rows = list(pairs)
    if not rows and fallback_accessions:
        a = str(sorted(str(x) for x in fallback_accessions)[0])
        rows = [{"a": a, "b": a, "prob": 0.5}]
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"StudyUID": str(r["a"]), "StudyUID_dup": str(r["b"]),
                                "PairProb": round(float(r["prob"]), 6)},
                               ensure_ascii=False) + "\n")
    return p


# --------------------------------------------------------------------------- #
# 校验（提交前自检）
# --------------------------------------------------------------------------- #
def validate_answer(answer_root: str, known_accessions: set[str] | None = None,
                    max_pairs_per_study: int = 200, expect_accessions: set[str] | None = None) -> dict:
    """按强制规范校验答案目录；返回 ``{"ok","errors","warnings"}``。"""
    import nibabel as nib

    errs, warns = [], []
    root = os.path.abspath(answer_root)
    if not os.path.isdir(root):
        return {"ok": False, "errors": [f"答案目录不存在: {root}"], "warnings": []}

    n_case = 0
    for acc in sorted(os.listdir(root)):
        acc_dir = os.path.join(root, acc)
        if not os.path.isdir(acc_dir):
            continue
        n_case += 1
        pj = os.path.join(acc_dir, "prediction.json")
        if not os.path.isfile(pj):
            errs.append(f"{acc}: 缺 prediction.json")
            continue
        try:
            with open(pj, encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:                                    # noqa: BLE001
            errs.append(f"{acc}: prediction.json 非法 JSON（{e}）")
            continue
        uri = d.get("SegmentationMaskURI") or {}
        items = uri.items() if isinstance(uri, dict) else [("all", uri)]
        for key, rel in items:
            rel = str(rel).strip()
            p = os.path.normpath(os.path.join(acc_dir, rel))
            if not os.path.isfile(p):
                errs.append(f"{acc}: 掩码文件不存在 {rel}")
                continue
            img = nib.load(p)
            arr = np.asanyarray(img.dataobj)
            u = np.unique(arr)
            if not set(u.tolist()) <= {0, 1}:
                errs.append(f"{acc}: {rel} 体素值非 0/1（出现 {u[:5].tolist()}）")
            if not np.all(np.isfinite(img.affine)):
                errs.append(f"{acc}: {rel} affine 含非法值")
    dp = os.path.join(root, "duplicate_pairs.jsonl")
    if not os.path.isfile(dp):
        errs.append("缺 duplicate_pairs.jsonl（规范要求至少一行有效记录）")
    else:
        cnt: dict[str, int] = {}
        n_lines = 0
        with open(dp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:                                 # noqa: BLE001
                    errs.append(f"duplicate_pairs.jsonl 非法 JSON 行: {line[:60]}")
                    continue
                n_lines += 1
                if not all(k in r for k in ("StudyUID", "StudyUID_dup", "PairProb")):
                    errs.append(f"duplicate_pairs.jsonl 字段缺失: {line[:60]}")
                    continue
                if not (0.0 <= float(r["PairProb"]) <= 1.0):
                    errs.append(f"PairProb 越界: {r}")
                for k in ("StudyUID", "StudyUID_dup"):
                    cnt[str(r[k])] = cnt.get(str(r[k]), 0) + 1
                    if known_accessions and str(r[k]) not in known_accessions:
                        warns.append(f"duplicate_pairs.jsonl 出现测试集外 UID: {r[k]}")
        over = {k: v for k, v in cnt.items() if v > max_pairs_per_study}
        if over:
            errs.append(f"存在超过 {max_pairs_per_study} 对/检查的记录: {list(over.items())[:3]}")
        if n_lines == 0:
            errs.append("duplicate_pairs.jsonl 为空")
    if expect_accessions:
        missing = sorted(set(expect_accessions) - set(os.listdir(root)))
        if missing:
            errs.append(f"{len(missing)} 个测试检查缺答案目录，例如 {missing[:5]}")
    if n_case == 0:
        warns.append("答案目录下没有任何 {AccessionNumber} 子目录")
    return {"ok": not errs, "errors": errs, "warnings": warns, "n_cases": n_case}
