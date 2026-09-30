"""DICOM 序列读取与 NIfTI 化（**规避"全盘 0 分"风险的关键模块**）。

背景冲突（必须以两种格式都能工作为准）：
- 《公共数据集格式说明》赛道四：*影像原始数据为 nii 格式*
- 《赛事开发规范（赛道四）》：*影像原始数据为 dicom 格式*

因此推理服务必须同时支持 NIfTI 与 DICOM。DICOM 通过 SimpleITK 读取，
仿射使用 SimpleITK 的 (origin, spacing, direction) 严格转成 nibabel 的 RAS affine，
保证**写出的 NIfTI 与 DICOM 世界坐标一一对应**，从而掩码的 affine/shape
能通过"与原始输入影像一致"的强制校验。

缓存：转换结果写到 ``DICOM_CACHE``（默认 ``{WORKSPACE}/cache_nifti`` 或系统临时目录），
避免每次测评重复解析；缓存的键为源目录的绝对路径哈希。
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

DICOM_EXTS = (".dcm", ".dicom", ".ima", ".img", ".raw")
NII_EXTS = (".nii", ".nii.gz")


def has_nifti(d: str) -> bool:
    for _dp, _dn, fns in os.walk(d):
        for fn in fns:
            if fn.lower().endswith(NII_EXTS):
                return True
    return False


def _looks_like_dicom(path: str) -> bool:
    """DICM magic（偏移 128）或常见扩展名。"""
    low = path.lower()
    if low.endswith((".dcm", ".dicom")):
        return True
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def sitk_image_to_ras(img):
    """SimpleITK 图像 → (nibabel 轴序数组, RAS affine)。

    用 ``sitk.GetArrayFromImage``（明确返回 ``(z, y, x)``）而非 ``np.asarray``
    （后者维度语义依赖 SimpleITK 版本，会得到 2D/4D 而失败）。
    """
    import SimpleITK as sitk

    arr_sitk = sitk.GetArrayFromImage(img)                                # (z, y, x)
    if arr_sitk.ndim == 2:                                                # 单层图像
        arr_sitk = arr_sitk[None]
    if arr_sitk.ndim > 3:                                                 # 多分量 → 取首分量
        arr_sitk = arr_sitk[..., 0]
    arr = np.transpose(arr_sitk.astype(np.float32), (2, 1, 0))             # -> (x, y, z)
    d = np.array(img.GetDirection(), float).reshape(3, 3)                  # LPS 方向
    sp = np.array(img.GetSpacing(), float)
    org = np.array(img.GetOrigin(), float)
    lps2ras = np.array([-1.0, -1.0, 1.0])
    aff = np.eye(4)
    for i in range(3):
        aff[:3, i] = d[:, i] * sp[i] * lps2ras
    aff[:3, 3] = org * lps2ras
    return arr, aff


def cache_root() -> str:
    root = os.environ.get("DICOM_CACHE")
    if not root:
        ws = os.environ.get("WORKSPACE", "")
        root = os.path.join(ws, "cache_nifti") if ws and os.path.isdir(ws) \
            else os.path.join(os.path.expanduser("~"), ".cache", "glioma4_nifti")
    os.makedirs(root, exist_ok=True)
    return root


def _series_out_dir(src_dir: str) -> str:
    key = hashlib.md5(os.path.abspath(src_dir).encode("utf-8")).hexdigest()[:16]
    out = os.path.join(cache_root(), key)
    os.makedirs(out, exist_ok=True)
    return out


def read_dicom_series(src_dir: str, min_slices: int = 3) -> list[dict]:
    """递归读取 ``src_dir`` 下所有 DICOM 序列 → [{arr, affine, series_uid, desc, meta}, ...]。

    仅依赖 SimpleITK（GDCM 内置），不要求 pydicom。
    """
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    out: list[dict] = []
    seen: set[str] = set()
    for dp, _dn, fns in os.walk(src_dir):
        dcms = [os.path.join(dp, f) for f in fns if _looks_like_dicom(os.path.join(dp, f))]
        if not dcms:
            continue                                                  # 目录内无 DICOM 文件
        try:
            sids = reader.GetGDCMSeriesIDs(dp)
        except Exception:                                             # noqa: BLE001
            sids = []
        if sids:
            files_by_sid = {sid: list(reader.GetGDCMSeriesFileNames(dp, sid)) for sid in sids}
            # 兜底：按 SeriesInstanceUID 分组后若没有任何"够长"的序列
            # （UID 缺失/被重写的二次导出、逐张重新导出的数据很常见），
            # 就把整个目录视作一个序列，交给 ImageSeriesReader 按几何排序。
            if not any(len(v) >= min_slices for v in files_by_sid.values()):
                files_by_sid = {os.path.basename(dp) or dp: sorted(dcms)}
        else:
            files_by_sid = {os.path.basename(dp) or dp: sorted(dcms)}
        for sid, files in files_by_sid.items():
            if sid in seen or len(files) < min_slices:
                continue
            try:
                r = sitk.ImageSeriesReader()
                r.SetFileNames(files)
                r.MetaDataDictionaryArrayUpdateOn()
                img = r.Execute()
            except Exception:                                         # noqa: BLE001
                continue
            arr, aff = sitk_image_to_ras(img)
            meta = {}
            try:
                for key, tag in (("desc", "0008|103e"), ("modality", "0008|0060"),
                                 ("thickness", "0018|0050"), ("study", "0020|000d")):
                    if r.HasMetaDataKey(0, tag):
                        meta[key] = r.GetMetaData(0, tag)
            except Exception:                                         # noqa: BLE001
                pass
            seen.add(sid)
            out.append({"arr": arr, "affine": aff, "series_uid": sid,
                        "desc": meta.get("desc", ""), "meta": meta})
    return out


def ensure_series_nifti(src_dir: str, force: bool = False) -> list[dict]:
    """把 ``src_dir`` 下的 DICOM 序列转成 NIfTI 落盘并返回元信息列表。

    返回 ``[{"path","series_uid","desc","arr_shape","affine"}...]``；
    已有 NIfTI 时不转换（返回空列表，由上层按 NIfTI 流程处理）。
    """
    import nibabel as nib

    if has_nifti(src_dir):
        return []
    outdir = _series_out_dir(src_dir)
    recs = read_dicom_series(src_dir)
    out: list[dict] = []
    for i, s in enumerate(recs):
        uid = str(s["series_uid"]).replace("/", "_")[:64] or f"series{i}"
        p = os.path.join(outdir, f"{uid}.nii.gz")
        if force or not os.path.isfile(p):
            nib.save(nib.Nifti1Image(s["arr"], s["affine"]), p)
        out.append({"path": p, "series_uid": s["series_uid"], "desc": s.get("desc", ""),
                    "shape": list(s["arr"].shape), "spacing": [
                        float(np.linalg.norm(s["affine"][:3, k])) for k in range(3)]})
    return out


def describe_dir(src_dir: str, limit: int = 12) -> dict:
    """给探针用的目录概览（不落盘）。"""
    recs = read_dicom_series(src_dir)
    return {"n_series": len(recs),
            "series": [{"uid": r["series_uid"][:40], "desc": r.get("desc", ""),
                        "shape": list(r["arr"].shape)} for r in recs[:limit]]}
