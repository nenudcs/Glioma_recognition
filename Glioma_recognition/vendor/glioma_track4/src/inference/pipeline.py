"""病例级推理流水线（赛道4）→ 规范答案。

流程（每个 AccessionNumber）：
1. 多序列 → 1mm 公共网格（缺失序列自动降级）→ 滑窗+TTA+多折集成：
   - 分割概率 [2,D,H,W]（core / peri）
   - 结构化字段概率 / 特殊影像概率（假人体、拼接）/ 重复影像嵌入
     （全局头使用**与训练同尺度的整脑视图**）；
2. 阈值（checkpoint 内验证集搜索值）→ 形态学后处理（连通域、peri ⊇ core）；
3. 两个掩码**回采样到各自源序列空间**并写出（仿射/维度一致 + 严格 0/1）；
4. 全批完成后写 ``duplicate_pairs.jsonl``（每例 Top-K + 相似度标定）。

**兜底**：任一病例推理异常时，仍按原始影像几何写出全 0 掩码与合规
``prediction.json``（该例至少不会因为"缺文件"而整例 0 分）。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np
import torch

from ..data.dataset import brain_center, build_case_volume, global_view, pick_series
from ..models.unet3d import build_model
from ..utils.config import data_source_tag, load_config, load_paths, resolve
from ..utils.logger import run_logger
from . import writer
from .duplicate import calibrate, fingerprint_similarity, gold_similarities, match_pairs
from .sliding import predict_volume
from .structured import derive_rules, make_conclusion, special_heuristics


def load_ensemble(ckpt_paths: list[str], device: str = "cuda") -> dict:
    """加载多折权重（EMA 优先），返回模型列表与元信息（arch / 阈值 / 全局视图尺寸）。"""
    pre = load_config("preprocess.yaml")
    models, cls_spec, thr, arch, gcfg = [], None, [], "mednext", {}
    special_trained = False
    for p in ckpt_paths:
        ck = torch.load(resolve(p), map_location="cpu", weights_only=False)
        spec = ck.get("cls_spec") or []
        mc = ck.get("model_cfg") or {}
        a = ck.get("arch") or mc.get("arch") or "mednext"
        m = build_model(spec, in_ch=len(pre["channels"]), arch=a,
                        base=int(mc.get("base", 32)), depth=int(mc.get("depth", 4)),
                        blocks_per_stage=int(mc.get("blocks_per_stage", 2)),
                        k=int(mc.get("k", 3)), expand=int(mc.get("expand", 2)),
                        dropout=0.0, aniso_z=bool(mc.get("aniso_z", False)),
                        max_ch=int(mc.get("max_ch", 320)),
                        plain_stages=int(mc.get("plain_stages", 0)),
                        dec_blocks=int(mc.get("dec_blocks", 1)))
        sd = ck.get("model_ema") or ck.get("model") or ck
        m.load_state_dict(sd, strict=False)
        m.eval().to(device)
        models.append(m)
        cls_spec = spec or cls_spec
        arch = a
        if ck.get("thresholds"):
            thr.append([float(x) for x in ck["thresholds"]])
        special_trained = special_trained or bool(ck.get("special_trained"))
        gcfg = {"size_mm": float(ck.get("global_size_mm", 192.0)),
                "out": int(ck.get("global_size", 96))}
    thr_mean = np.mean(thr, axis=0).tolist() if thr else [0.5, 0.5]
    return {"models": models, "cls_spec": cls_spec, "thresholds": thr_mean[:2],
            "arch": arch, "global_view": gcfg, "special_trained": special_trained}


# --------------------------------------------------------------------------- #
# 后处理
# --------------------------------------------------------------------------- #
def postprocess(core: np.ndarray, peri: np.ndarray, min_vox: int, spacing=(1.0, 1.0, 1.0),
                keep_n: int = 3, bridge_mm: float = 10.0):
    """连通域后处理（这类脑肿瘤分割任务的稳定涨点手段）。

    1. 按体积过滤碎片（< ``min_vox`` 体素）；
    2. core 保留最大的若干连通域（``keep_n``，默认 3）——**不能只留最大**，
       否则多灶/卫星强化灶会被删除而显著掉 Dice；
    3. peri 在此基础上再按"与 core 的距离 < ``bridge_mm``"合并邻近分量（卫星水肿），
       并强制 ``peri ⊇ core``。
    """
    from scipy import ndimage as ndi

    def _clean(m, ref=None, max_keep=keep_n, need_near=False):
        m = m.astype(bool)
        if not m.any():
            return m
        lab, n = ndi.label(m)
        if n <= 1:
            return m
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        order = np.argsort(-sizes[1:]) + 1
        dist = None
        if need_near and ref is not None and ref.any():
            dist = ndi.distance_transform_edt(~ref.astype(bool), sampling=spacing)
        keep = []
        for j in order:
            if sizes[j] < min_vox:
                continue
            if need_near and dist is not None and keep and dist[lab == j].min() > bridge_mm:
                continue
            keep.append(j)
            if len(keep) >= max_keep:
                break
        return np.isin(lab, keep)

    core = _clean(core)
    peri = _clean(peri | core, ref=core, need_near=True)
    peri = peri | core
    if core.sum() < min_vox:
        core = np.zeros_like(core)
    if peri.sum() < min_vox:
        peri = np.zeros_like(peri)
    return core.astype(np.uint8), peri.astype(np.uint8)


# --------------------------------------------------------------------------- #
# 主流水线
# --------------------------------------------------------------------------- #
class GliomaPipeline:
    def __init__(self, ckpt_paths: list[str], device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pre = load_config("preprocess.yaml")
        self.labels = load_config("labels.yaml")["fields"]
        self.net = load_ensemble(ckpt_paths, self.device)
        self.models = self.net["models"]
        self.thresholds = self.net["thresholds"]
        self.gcfg = self.net.get("global_view") or {}
        self.cls_spec = self.net["cls_spec"] or [(f["key"], 1 if f["type"] == "binary"
                                                  else len(f["classes"])) for f in self.labels]

    # ------------------------------------------------------------------ #
    def predict_prob(self, case: dict, vol: np.ndarray) -> dict:
        """滑窗 + TTA + 集成 → 概率图与全局头输出（不含阈值/后处理）。"""
        inf = self.pre["inference"]
        gsz = int(self.gcfg.get("out", inf.get("global_size", 96)))
        gmm = float(self.gcfg.get("size_mm", 192.0))
        gvol = global_view(vol, brain_center(vol), gmm, gsz)
        return predict_volume(
            self.models, vol, patch=tuple(inf["patch"]), overlap=float(inf["overlap"]),
            tta_flips=tuple(inf.get("tta_flips") or ()),
            seg_tta_flips=tuple(inf.get("seg_tta_flips") or inf.get("tta_flips") or ()),
            amp_dtype=torch.bfloat16, tta_batch=int(inf.get("tta_batch", 2)),
            global_size=gsz, global_vol=gvol,
            crop_brain=bool(self.pre["geometry"].get("crop_brain", True)),
            brain_margin=int(self.pre["geometry"].get("brain_margin_vox", 4)),
            device=self.device)

    # ------------------------------------------------------------------ #
    def process_case(self, case: dict, prepared: tuple | None = None) -> dict:
        t0 = time.time()
        log: list[str] = []
        if prepared is not None:
            vol, common_aff, _ = prepared
        else:
            vol, common_aff, _ = build_case_volume(case, self.pre, log)
        res = self.predict_prob(case, vol)
        inf = self.pre["inference"]

        spacing = tuple(float(np.linalg.norm(common_aff[:3, i])) for i in range(3))
        core = res["seg"][0] > float(self.thresholds[0])
        peri = res["seg"][1] > float(self.thresholds[1])
        core, peri = postprocess(core, peri, int(inf["min_tumor_voxels"]), spacing,
                                 keep_n=int(inf.get("keep_components", 3)),
                                 bridge_mm=float(inf.get("bridge_mm", 10.0)))

        ch_names = [c["name"] for c in self.pre["channels"]]
        rules = derive_rules(vol, ch_names, core, peri, spacing, int(inf["min_tumor_voxels"]),
                             affine=common_aff)
        rules["tumor_prob_model"] = 0.0
        probs = {f["key"]: res["cls"][i] for i, f in enumerate(self.labels) if i < len(res["cls"])}
        if probs.get("TumorProbability") is not None and len(probs["TumorProbability"]):
            rules["tumor_prob_model"] = float(probs["TumorProbability"][0])

        pred = writer.assemble_prediction(case["accession"], self.labels, probs, rules)
        pred["ProcessingTime_ms"] = int((time.time() - t0) * 1000)
        # 目标一/二：模型头 + 可解释启发式兜底（基础考核项，必须有保险）
        special = np.asarray(res["special"], float).ravel()
        p_fake_m = float(special[0]) if special.size else 0.5
        p_stitch_m = float(special[1]) if special.size > 1 else 0.5
        h_fake, h_stitch = special_heuristics(vol, self.pre)
        if self.net.get("special_trained"):
            # 模型头已训练 → 以模型为主，启发式仅在非常确定时补充（避免假阳性）
            p_fake = max(p_fake_m, h_fake if h_fake >= 0.9 else 0.0)
            p_stitch = max(p_stitch_m, h_stitch if h_stitch >= 0.9 else 0.0)
        else:                                                     # 未训练该头 → 以启发式为准
            p_fake, p_stitch = h_fake, h_stitch
            rules["special_heuristic_only"] = True
        pred["IsNotHumanBodyProb"] = round(float(np.clip(p_fake, 0, 1)), 4)
        pred["IsStitchedProb"] = round(float(np.clip(p_stitch, 0, 1)), 4)
        pred["Prediction"] = {k: pred.pop(k) for k in list(pred)
                              if k in {f["key"] for f in self.labels}}
        pred["Interpretation"] = {"Conclusion": make_conclusion(case["accession"], pred["Prediction"])}

        # 掩码写回各自源序列空间
        series_meta = {}
        for role, ch in (("core", "t1c"), ("peri", "flair")):
            meta = pick_series(case, self.pre).get(ch)
            if meta is None:
                continue
            import nibabel as nib
            img = nib.load(meta["path"])
            series_meta[role] = {
                "series_uid": meta.get("series_uid") or os.path.basename(os.path.dirname(meta["path"])),
                "path": meta["path"], "affine": np.asarray(img.affine, float),
                "shape": tuple(img.shape)}

        masks = {"core": core, "peri": peri}
        return {"masks": masks, "common_aff": common_aff, "pred": pred,
                "series_meta": series_meta, "embed": np.asarray(res["embed"], np.float32),
                "log": log, "special": res["special"]}

    # ------------------------------------------------------------------ #
    def run_batch(self, dataset_path: str, out_dir: str, cases: list[dict] | None = None,
                  gold_pairs: list[list[str]] | None = None, limit: int | None = None) -> dict:
        """dataset_path：测评数据集根（``<AccessionNumber>/<SeriesUid>/*``，NIfTI 或 DICOM）。"""
        from ..data.probe import scan_real
        paths = load_paths()
        logger = run_logger(resolve(paths["logs_dir"]), "inference")
        if cases is None:
            cases = scan_real(dataset_path)
        if limit:
            cases = cases[:limit]
        # ★ 目录级兜底（关键）：规范要求"每个 AccessionNumber 都必须有 prediction.json"。
        # 探针扫不到的病例（空目录、affine 全 0/含 NaN 导致解析失败、纯异常数据等）
        # 若被静默忽略，该例**全部指标直接 0 分**。这里把数据集根下所有一级子目录
        # 都纳入处理：无影像的病例会在 build_case_volume 阶段失败 → 自动走兜底写出。
        all_accs: set[str] = set()
        if os.path.isdir(dataset_path):
            all_accs = {d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d)) and d.lower() != "annotation"}
        covered = {str(c.get("accession")) for c in cases}
        for a in sorted(all_accs - covered):
            cases.append({"accession": a, "dir": os.path.join(dataset_path, a),
                          "images": {}, "masks": {}, "labels": {}, "orphan": True})
        os.makedirs(out_dir, exist_ok=True)
        # 测试集数据源标识：按 dataset_path 实际位置生成 official/test_v1 或 local/<name>/test
        ds_tag = data_source_tag(dataset_path, phase="test")

        embeds: dict[str, np.ndarray] = {}
        feats: dict[str, dict] = {}
        done, failed = 0, []
        # 预取流水线：DICOM→NIfTI、重采样、z-score 是 I/O/CPU 密集，
        # 与 GPU 推理重叠可显著缩短整批时间（DICOM 测试集尤其明显）。
        from concurrent.futures import ThreadPoolExecutor
        n_workers = max(1, int(self.pre["inference"].get("prefetch_workers", 2)))

        def _prep(c):
            try:
                log: list[str] = []
                vol, aff, _m = build_case_volume(c, self.pre, log)
                return (vol, aff, _m), log
            except Exception as e:                                # noqa: BLE001
                return e, []

        ex = ThreadPoolExecutor(max_workers=n_workers)
        order = list(range(len(cases)))
        window = n_workers + 1
        futs = {i: ex.submit(_prep, cases[i]) for i in order[:window]}
        nxt = window
        for i in order:                                               # 顺序消费 + 持续补位
            while nxt < len(order) and len(futs) < window:
                futs[nxt] = ex.submit(_prep, cases[nxt])
                nxt += 1
            case = cases[i]
            acc = case.get("accession")
            prepared, plog = futs.pop(i).result()
            try:
                if isinstance(prepared, Exception) or prepared is None:
                    raise RuntimeError(str(prepared))
                r = self.process_case(case, prepared=prepared)
                if plog:
                    r["log"] = list(plog) + list(r.get("log") or [])
                masks = {"core": (r["masks"]["core"], r["common_aff"]),
                         "peri": (r["masks"]["peri"], r["common_aff"]),
                         "attention": (r["masks"]["peri"], r["common_aff"])}
                writer.write_case(out_dir, acc, masks, r["pred"], r["series_meta"], self.pre)
                embeds[acc] = r["embed"]
                feats[acc] = writer.case_fingerprint(case)
                done += 1
                logger.log(phase="test", mode="inference", data_source=ds_tag,
                           checkpoint="", accession=acc,
                           has_tumor=bool(r["pred"]["Prediction"].get("TumorProbability", 0) > 0.5),
                           processing_ms=r["pred"]["ProcessingTime_ms"])
                if r["log"]:
                    print(f"[infer] {acc} 序列降级: {'; '.join(r['log'])}", flush=True)
                print(f"[infer] {acc} ok ({r['pred']['ProcessingTime_ms']}ms)", flush=True)
            except Exception as e:                                    # noqa: BLE001
                failed.append(acc)
                logger.log(phase="test", mode="inference", data_source=ds_tag,
                           accession=acc, error=str(e)[:200])
                print(f"[infer] ✗ {acc}: {e} —— 写兜底答案", flush=True)
                try:                                                  # 兜底：仍产出合规答案
                    writer.write_fallback_case(out_dir, case, self.pre)
                    done += 1
                except Exception as e2:                               # noqa: BLE001
                    print(f"[infer] ✗✗ {acc} 兜底失败: {e2}", flush=True)
        try:
            ex.shutdown(wait=False)
        except Exception:                                             # noqa: BLE001
            pass

        # 重复影像：学习式嵌入 + 手工指纹 融合 → 标定 → Top-K
        calib = None
        if gold_pairs:
            pos, neg = gold_similarities(embeds, gold_pairs)
            calib = calibrate(pos, neg)
            print(f"[infer] 重复影像标定: {calib}", flush=True)
        pairs = match_pairs(embeds, calib, topk=int(self.pre["inference"]["duplicate_topk"]),
                            per_study_k=int(self.pre["inference"].get("duplicate_per_study", 50)),
                            feats=feats, w_fp=float(self.pre["inference"].get("fp_weight", 0.5)))
        # 规范：文件至少一行有效记录。无有效对时由 writer 写一条自比对占位行。
        accs = sorted({str(c.get("accession")) for c in cases})
        writer.write_duplicate_pairs(out_dir, pairs, fallback_accessions=accs)

        # 校验基准 = 数据集目录全集（含被探针跳过的病例），确保"每例都有答案"被真正校验
        expect = all_accs or {str(c.get("accession")) for c in cases}
        rep = writer.validate_answer(out_dir, known_accessions=expect or None,
                                     expect_accessions=expect or None)
        print(f"[infer] 完成 {done}/{len(cases)} 例（兜底 {len(failed)} 例）；"
              f"答案自检 ok={rep['ok']} errors={len(rep['errors'])}", flush=True)
        for e in rep["errors"][:5]:
            print(f"[infer][校验] {e}", flush=True)
        return {"n_done": done, "n_cases": len(cases), "pairs": len(pairs),
                "validate": rep, "failed": failed}
