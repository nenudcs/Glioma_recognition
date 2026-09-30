"""数据探针：扫描比赛训练集/测评集 → 自动识别结构并生成 manifest。

识别内容（不依赖任何预设目录布局）：
1. 特殊影像目录 ``annotation/{Composition,fake,duplicate}`` 与重复影像金标准（每行 ``src,desc``）；
2. 真实影像：一级目录 = 检查号；其下按序列分目录或直接放影像；
   **同时支持 NIfTI 与 DICOM**（DICOM 自动转 NIfTI 并缓存，见 ``data/dicom.py``）；
3. 掩码角色：**按掩码所在序列的模态判定**（core=T1C 上的肿瘤瘤体；peri=FLAIR/T2 上的
   瘤体∪水肿 或全肿瘤），同一角色多个掩码取并集；
4. 结构化金标准表（csv/xlsx）→ 规范字段；
5. 抽样读取 shape/spacing/orientation（QC 用）。

用法：
    python -m src.data.probe --root <数据根> [--out data/manifest.json] [--limit-cases 200]
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from ..utils.config import data_source_tag, load_paths, resolve, val_root
from .labels import (SERIES_TYPE_TABLE, build_uid_index, desc_index_from_records,
                     find_named_table,
                     find_official_labels, find_structured_tables,
                     guess_modality, has_input_modality, has_series_type_table,
                     has_strict_mask_hint, id_key, is_explicit_other, is_hard_skip,
                     is_official_accession, is_official_mask_name,
                     lookup_series_type, mask_role_for, norm_key,
                     read_abnormal_table,
                     read_duplicate_pairs, read_mask_table, read_series_desc_index,
                     read_series_types,
                     read_structured_table, read_text_any_encoding,
                     series_uid_candidates, sidecar_desc,
                     structured_from_row)

IMG_EXT = (".nii.gz", ".nii")
SKIP_NAME_KW = ("dicomdir", "license", "readme", "vht", ".mhd")

#: 平台 ``/2026aicompetition/datasets`` 下的阶段目录名。
#: 数据根必须精确到其中一个（训练用 ``training``），不能停在父目录。
_PLATFORM_PHASES = frozenset({
    "training", "evaluation_first", "evaluation_second",
    "evaluation_finals", "verification",
})


def assert_case_root(root: str | os.PathLike) -> None:
    """拦截"数据根误指向 ``datasets/`` 父目录"。

    ``/2026aicompetition/datasets`` 下是 5 个阶段目录（``training`` /
    ``evaluation_first`` / ``evaluation_second`` / ``evaluation_finals`` /
    ``verification``），数据根要精确到其中一个。误传父目录时旧实现会把阶段名
    当成检查号：

    * 清单里出现 5 个假病例（``evaluation_first``…），把 5 份数据的像素混在一起；
    * 金标准一张也对不上，``no_labels`` 全空，分类头实际从未收到有效监督；
    * 全程不报错 —— 训练能跑完、指标能打印，直到提交才发现完全跑偏。

    因此直接失败，并在报错里给出应该填的路径。
    """
    if not os.path.isdir(root):
        return
    children = sorted(e for e in os.listdir(root)
                      if os.path.isdir(os.path.join(root, e)))
    if len(children) < 2 or not {c.casefold() for c in children} <= _PLATFORM_PHASES:
        return
    raise ValueError(
        f"数据根 {root} 指向数据集父目录，其下是平台阶段目录 {children}。"
        f"请把数据根设为具体阶段（训练应为 {os.path.join(root, 'training')}）；"
        f"否则这些目录名会被当作检查号，训练数据完全错误却不报错。"
    )
#: 纯模态名（这些是**影像**而非掩码；避免 "flair.nii.gz" 被误判成掩码）
PURE_MODALITY_STEMS = {"t1c", "t1ce", "t1_ce", "t1", "t1w", "t1wi", "t2", "t2w", "t2wi",
                       "flair", "t2flair", "t2_flair", "dwi", "adc", "swi", "bold", "seg",
                       "image", "img", "volume", "scan", "series", "mri", "brain"}


def _is_img(fn: str) -> bool:
    return fn.lower().endswith(IMG_EXT)


def _stem(fn: str) -> str:
    low = fn.lower()
    for ext in IMG_EXT:
        if low.endswith(ext):
            return low[: -len(ext)]
    return low


def _probe_nifti(path: str) -> dict:
    try:
        import nibabel as nib
        img = nib.load(path)
        return {"shape": list(img.shape),
                "spacing": [round(float(x), 3) for x in img.header.get_zooms()[:3]],
                "orient": "".join(nib.aff2axcodes(img.affine))}
    except Exception as e:                                        # noqa: BLE001
        return {"error": str(e)[:80]}


# --------------------------------------------------------------------------- #
# 特殊影像
# --------------------------------------------------------------------------- #
def scan_special(root: str) -> dict:
    """扫描 ``annotation/{Composition,fake,duplicate}`` 与重复影像金标准。

    返回 ``composition`` / ``fake`` 的**病例标识集合**（用于目标一二的监督），
    ``gold_pairs`` 为重复影像金标准对（**取自 ``duplicate/`` 目录下的金标准文件本身** ——
    格式说明口径："每一行为 src_img, desc_img"）；``gold_files`` 是命中的金标准文件清单，
    供上层在"有文件却没解析出对"时报出具体是哪个文件。
    """
    out: dict = {"annotation_dir": None, "composition": [], "fake": [], "duplicate": [],
                 "gold_pairs": [], "gold_files": [],
                 "composition_cases": [], "fake_cases": []}
    # 目录名两套写法都要认：本地模拟集用 `Composition`，**官方用 `compositing`**
    # （见天坛 `AIRecongition/src/data/paths.py`）。只认前者会让"拼接"这一类
    # 正样本整批找不到 → 目标二-A 的头没有监督信号，而且不报错。
    for cand in (os.path.join(root, "annotation"), root):
        if any(os.path.isdir(os.path.join(cand, name))
               for name in ("Composition", "compositing", "fake")):
            out["annotation_dir"] = cand
            break
    if not out["annotation_dir"]:
        return out
    ann = out["annotation_dir"]

    def _identifiers(d: str) -> list[str]:
        """把特殊影像目录下的条目规整为"病例标识"（目录名，或文件名去扩展名）。"""
        ids = []
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isdir(p):
                ids.append(name)
            elif _is_img(name) or name.lower().endswith((".dcm", ".dicom")):
                ids.append(_stem(name))
        return ids

    # 每个用途可能对应多个目录名（本地 `Composition` / 官方 `compositing`）
    for names, key in ((("Composition", "compositing"), "composition"),
                       (("fake",), "fake"), (("duplicate",), "duplicate")):
        for cls in names:
            d = os.path.join(ann, cls)
            if not os.path.isdir(d):
                continue
            items = _identifiers(d)
            merged = list(dict.fromkeys(out[key] + items))[:500]   # 去重且保序
            out[key] = merged
            out[f"{key}_cases"] = merged

    # 重复影像金标准（csv/txt，每行 `src_img, desc_img` —— 两个值是**检查号**）。
    # 找到的文件路径记进 ``gold_files``：目录里**有**文件却解析出 0 对时，
    # 上层才报得出"是哪个文件、格式不对"，而不是只剩一句"0 对"。
    for dirpath, _dirs, files in os.walk(os.path.join(ann, "duplicate")):
        for fn in files:
            if not fn.endswith((".csv", ".txt")):
                continue
            path = os.path.join(dirpath, fn)
            out["gold_files"].append(path)
            try:
                # 不写死 utf-8：金标准 csv 常是 Excel/WPS 导出的 ANSI(GBK) 或
                # 「Unicode 文本」(UTF-16)，按 utf-8 解直接 UnicodeDecodeError →
                # ``gold_pairs`` 变 0 对，看着像"数据没给金标准"
                # （见 :data:`labels.TEXT_ENCODINGS`）。
                for line in read_text_any_encoding(path).splitlines():
                    line = line.strip()
                    if not line or line.lower().startswith(("src", "#")):
                        continue
                    parts = [p.strip() for p in line.replace("\t", ",").split(",") if p.strip()]
                    if len(parts) >= 2:
                        out["gold_pairs"].append([parts[0], parts[1]])
            except Exception as exc:                              # noqa: BLE001
                # 读取异常**不能静默**：吞掉之后 gold_pairs 为 0，
                # 看起来像"数据没给金标准"，实际是文件读不了（编码/权限/损坏）。
                print(f"[probe][告警] 读重复金标准失败 {path}："
                      f"{type(exc).__name__}: {exc}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# 真实影像
# --------------------------------------------------------------------------- #
#: 官方数据里"异常影像"的子目录名（`AIRecongition/src/data/paths.py` 约定）：
#: ``<根>/fake/<检查号>/…``、``<根>/compositing/<检查号>/…``、``<根>/duplicate/<检查号>/…``。
#: 它们与主目录**同构**，因此绝不能被当成检查号 —— 否则会多出三个名叫 fake/compositing/
#: duplicate 的"病例"，而它们的"序列"是几百上千个真实病例目录。
SPECIAL_SOURCE_DIRS = ("fake", "compositing", "composition", "duplicate")

#: 允许自动下钻的中间层：``annotation`` / ``original``（影像/标注表所在层）+ 平台阶段名。
#: 平台实际布局比"数据集根"多这一层，见 ``README.md`` §2.2「数据布局」。
#:
#: ``original`` 是**验证集**的实测布局：``verification/original/<检查号>/<序列>/``，
#: 影像、``SeriesType.xlsx`` 与标注表都在 ``verification/original/``。漏了它，
#: 数据根填 ``…/verification`` 时 ``original`` 会被当成检查号 —— 扫描结果是
#: "病例数正常、却报无任何可用序列"，且表也找不到（候选目录里没有它）。
_DESCEND_DIRS = frozenset({"annotation", "original"}) | _PLATFORM_PHASES
#: 顶层非病例目录：本层出现其中任何一个，说明"还没到病例层"
_NON_CASE_DIRS = (frozenset({"annotation", "original", "cache", "runs", "folds",
                             "labels", "logs", "checkpoints", "weights"})
                  | frozenset(SPECIAL_SOURCE_DIRS))


def resolve_case_root(root: str) -> str:
    """把"填高了一层"的数据根下钻到真正含病例目录的那一层。

    平台实测（2026-09-24）——**训练集与验证集的中间层名字不同**：

    ```text
    /2026aicompetition/datasets/training/annotation/<检查号>/…      ← 容器叫 annotation
    /2026aicompetition/datasets/verification/original/<检查号>/…    ← 容器叫 original
    ```

    因此容器名单（:data:`_DESCEND_DIRS`）同时含 ``annotation`` / ``original``
    （外加平台阶段名）。评测集 ``evaluation_*`` 的容器名还未知 ——
    幸好 ``SeriesType.xlsx`` **随数据下发、与检查号目录同层**，所以最后还有一条
    **名字无关**的兜底："唯一子目录里直接放着这张表"就下钻。

    规则（按序，命中即停）：

    1. 唯一子目录是"容器"（已知容器名，**或其内直接放着 SeriesType.xlsx**）
       → 下钻并告警；
    2. 本层存在"非白名单"的子目录 → 病例层（本地模拟集、无表的布局）；
    3. 已知容器名的唯一候选（本层全是白名单目录时）→ 下钻。

    多阶段父目录（``/2026aicompetition/datasets``）仍由 :func:`assert_case_root`
    在扫描前报错，不下钻、不猜。
    """
    if not os.path.isdir(root):
        return root
    children = sorted(e for e in os.listdir(root)
                      if os.path.isdir(os.path.join(root, e)))
    if not children:
        return root

    def _is_container(name: str) -> bool:
        if name.casefold() in _DESCEND_DIRS:
            return True
        # 名字无关的判定：容器里直接放着数据信息表（与检查号目录同层）
        return has_series_type_table(os.path.join(root, name))

    if len(children) == 1 and _is_container(children[0]):   # ① 唯一子目录是容器
        print(f"[probe][告警] 数据根 {root} 下没有病例目录，已自动下钻到 "
              f"{children[0]}/（若不对请用 DATASET_ROOT 显式指定）", flush=True)
        return os.path.join(root, children[0])
    if any(c.lower() not in _NON_CASE_DIRS for c in children):
        return root                                        # ② 本层已有病例目录
    cands = [c for c in children if c.casefold() in _DESCEND_DIRS]
    if len(cands) == 1:                                    # ③ 已知容器名的唯一候选
        print(f"[probe][告警] 数据根 {root} 下没有病例目录，已自动下钻到 "
              f"{cands[0]}/（若不对请用 DATASET_ROOT 显式指定）", flush=True)
        return os.path.join(root, cands[0])
    return root


#: 掩膜相关的告警：**同类只打前几条**。上万例数据里同一个问题会刷屏，而
#: "打不出来"比"刷屏"坏得多——掩膜认错角色是**不报错**的（只是指标悄悄偏低）。
_MASK_WARNED: set[str] = set()
_MASK_WARN_COUNT: dict[str, int] = {}
_MASK_WARN_LIMIT = 3


def _warn_mask_once(kind: str, detail: str, msg: str) -> None:
    """掩膜类告警去重打印（同类最多 :data:`_MASK_WARN_LIMIT` 条 + 一条收尾提示）。"""
    if f"{kind}|{detail}" in _MASK_WARNED:
        return
    n = _MASK_WARN_COUNT.get(kind, 0)
    if n >= _MASK_WARN_LIMIT:
        return
    _MASK_WARNED.add(f"{kind}|{detail}")
    _MASK_WARN_COUNT[kind] = n + 1
    tail = "（同类告警不再重复打印）" if n + 1 == _MASK_WARN_LIMIT else ""
    print(f"[probe][告警] {msg}{tail}", flush=True)


#: 靠**标注表的「序列描述」**（而不是 ``SeriesType.xlsx``）认出来的序列：``{检查号|序列UID}``。
#: 这是"类型表没到手、旁证那一路顶上了"的唯一可见指标 —— 不报出来就分不清
#: "这批数据本来就没有模态信息"和"我们的兜底没接上"（后者修起来完全不一样）。
_desc_used: set[str] = set()


def _report_desc_fallback() -> None:
    """汇报"序列描述旁证"的命中量（0 条时**不出声**，避免无表数据也刷一行）。"""
    if not _desc_used:
        return
    n_acc = len({s.split("|", 1)[0] for s in _desc_used})
    print(f"[probe] 已用**标注表的「序列描述」**作模态旁证："
          f"{len(_desc_used)} 路 / {n_acc} 例"
          f"（{SERIES_TYPE_TABLE} 里没有这些 UID；列名见 README §7.2）", flush=True)


def _collect_nifti(cdir: str, accession: str = "",
                   series_types: dict | None = None,
                   mask_names: dict | None = None,
                   uid_index: dict | None = None,
                   desc_index: dict | None = None) -> tuple[dict, list, list]:
    """扫描一个检查目录下的 NIfTI：返回 ``(images, mask_entries, unknown)``。

    - images: ``{modality: {"path","series_uid","file"}}``
    - mask_entries: ``[(role, modality, path, series_uid), ...]``（同一角色可多条 → 取并集）
    - unknown: **认不出模态的序列全表**（``[{...}]``）

    ``unknown`` 为什么必须单独返回：数据的模态来自数据信息 ``SeriesType.xlsx``
    （评测集在正式测试时才随测试数据下发；表没到手或表里没有这个检查号时
    就一条都认不出来），
    序列目录名是 DICOM UID，关键词一个都命中不了 —— 此时唯一的出路是
    **读体素用统计模型判模态**（``data.modality_model``）。而原先的实现
    只把**第一个**认不出的序列塞进 ``images["other"]``、其余直接丢弃，
    兜底模型最多只能看到 1 个序列，且拿不到完整候选。

    不能把列表塞进 ``images``（如 ``images["unknown"] = [...]``）：
    ``inference/writer.py`` 会 ``for mod, meta in images.items()`` 后取
    ``meta["path"]``，遇到 list 会直接崩。

    ``series_types`` 是官方 ``SeriesType.xlsx`` 解析出的
    ``{(检查号, 序列号): 序列类型}``。**官方数据必须靠它**：序列目录名是
    DICOM UID、文件名也是 UID，任何"按名字猜模态/掩膜"的关键词都命中不了——
    探针会表现为"病例数正常、模态全是 other、掩膜一个没有"，
    而训练侧更直接：``无任何可用序列``。

    官方布局是**扁平**的，影像与掩膜同在 ``<检查号>/`` 下
    （``<序列UID>.nii.gz`` / ``<序列UID>_<RoiName>_<RoiNumber>_mask.nii.gz``）：
    掩膜的文件名主干**不是**裸 UID，若直接拿去查类型表就查不到模态，而
    ``瘤体`` 的角色**依赖模态**（FLAIR/T2→peri，T1/T1CE→core）→ 会落成 core、
    进错任务空间（见 :func:`labels.series_uid_candidates`）。所以这里按候选
    逐级查表，并用 ``_mask`` 后缀把"认不出角色的掩膜"挡在输入通道之外。
    """
    images: dict[str, dict] = {}
    masks: list[tuple[str, str | None, str, str]] = []
    unknown: list[dict] = []
    for dirpath, _dirs, files in os.walk(cdir):
        sdir = os.path.basename(dirpath)
        for fn in sorted(files):
            if not _is_img(fn) or any(k in fn.lower() for k in SKIP_NAME_KW):
                continue
            full = os.path.join(dirpath, fn)
            stem = _stem(fn)
            is_mask_name = is_official_mask_name(fn)
            # 序列 UID 的三种来源（**掩膜必须走 ③**，否则 role 会静默判错）：
            #   ① 每序列一个子目录的老布局 → 目录名就是 UID；
            #   ② 扁平布局的影像 → 主干即裸 UID；
            #   ③ 扁平布局的掩膜 → 主干是 `<UID>_<RoiName>_<RoiNumber>_mask`，
            #      第一个 `_` 之前才是 UID（DICOM UID 只含数字与点、不含下划线）。
            if sdir and sdir != os.path.basename(cdir):
                series_uid = sdir
            elif is_mask_name:
                series_uid = stem.split("_", 1)[0] or stem
            else:
                series_uid = stem
            # 查表候选：解析出的 UID 优先，再补主干/各前缀（列名口径不一、SUID 单键回退用）
            uid_candidates = tuple(dict.fromkeys(
                (series_uid,) + series_uid_candidates(stem)))
            # 序列类型，四级兜底（**越靠前越权威**）：
            #   ① ``SeriesType.xlsx``：官方主力；查不到精确键时按 SeriesUid 单键回退
            #      （检查号列与磁盘目录名口径不一致时，UID 是两边唯一必然同源的键）；
            #   ② **标注表 ROI 级别的「序列描述」**（``SeriesDescription`` /
            #      ``DetailDescription``）：它承载的就是那 5 类模态取值。这是
            #      "类型表拿不到"时唯一还能批量判模态的正规线索 —— 没有它就只能
            #      整批落到体素判别模型（或干脆 unknown）；
            #   ③ 同名 json sidecar；
            #   ④ 文件名 / 目录名（模拟集、以及名字里就带模态的公开数据）。
            desc = lookup_series_type(series_types, accession,
                                      uid_candidates, uid_index)
            if not desc and desc_index:
                for _u in uid_candidates:
                    hit = desc_index.get(norm_key(_u))
                    if hit:
                        desc = hit
                        _desc_used.add(f"{accession}|{series_uid}")
                        break
            desc = desc or (sidecar_desc(full) or "")
            mod = guess_modality(desc) or guess_modality(stem) or guess_modality(sdir)
            is_pure = stem.strip() in PURE_MODALITY_STEMS
            role = None if is_pure else mask_role_for(fn, mod)
            # 类型表给出的掩膜：文件名是 UID，只能靠"类型 + 严格线索"识别。
            # 严格线索必不可少——"T1增强"这类**影像**名里也含"增强"，
            # 直接送进 mask_role_for 会被判成 core 掩膜。
            if role is None and desc and has_strict_mask_hint(desc):
                role = mask_role_for(f"{desc} {fn}", mod)
            # 官方 `4_masklabel.xlsx` 指定的掩膜：文件名是**任意的**（如 core.nii.gz），
            # 靠关键词认不出。不同步排除的话，掩膜会被当成一路"影像"混进输入通道。
            if role is None and mask_names:
                for _key in uid_candidates:
                    if fn in (mask_names.get(_key) or []):
                        role = mask_role_for(f"{fn} {desc}", mod) or "core"
                        break
            # `瘤体` 的角色**依赖模态**：FLAIR/T2 上是 peri（任务B），T1/T1CE 上是
            # core。类型表里没有这一路序列时模态为空、只能落到 core —— 这是整条链上
            # 唯一还能被救回来的静默错，必须响（另一种是表里根本没这个 UID）。
            if role == "core" and mod is None and "瘤体" in fn and "肿瘤瘤体" not in fn:
                _warn_mask_once(
                    "mask_modality", fn,
                    f"掩膜 {accession or '?'}/{fn} 所在序列的模态**没解析出来**"
                    f"（{SERIES_TYPE_TABLE} 里没有这个 UID）→ `瘤体` 只能按 core 处理，"
                    f"若它来自 FLAIR/T2 应为 peri（任务B）")
            if role:
                masks.append((role, mod, full, series_uid))
            elif is_mask_name:
                # 官方掩膜命名、但 ROI 名认不出角色（关键词表之外的取值）：
                # **绝不**当影像丢进输入通道（`series_uid` 是 UID，体素模型会把它
                # 猜成 t1c/t2 —— 标签当输入）。跳过并明确报数。
                _warn_mask_once(
                    "mask_role", fn,
                    f"掩膜 {accession or '?'}/{fn} 的 ROI 名认不出角色 → **已跳过**："
                    f"既不算掩膜、也不进输入通道。ROI 名取值见 README §2.2"
                    f"（瘤体/水肿/肿瘤瘤体/全肿瘤/异常信号）")
            else:
                meta = {"path": full, "series_uid": series_uid, "file": fn}
                key = mod or "other"
                if key not in images:
                    images[key] = dict(meta)          # 兼容旧语义：仍是"第一个"
                if mod is None and is_explicit_other(desc):
                    # 类型表**明确写了"其他"**（如平台 SeriesType.xlsx 的 其他）：
                    # 这是权威结论"它不是 T1/T2/FLAIR/T1CE 中的任何一个"，
                    # 不是"没认出来"。丢给体素模型猜只会把 DWI/ADC 判成 t2
                    # 填进通道（比空通道更有害），所以标记后不进 unknown。
                    images[key]["declared_other"] = True
                elif mod is None:
                    # 认不出模态的序列**全部保留**（见 docstring：评测期要靠模型回头判）
                    unknown.append({**meta, "reason": desc or stem or sdir})
    return images, masks, unknown


def _collect_dicom(cdir: str, log: list | None = None) -> dict:
    """把 DICOM 序列转成 NIfTI（缓存）→ ``{modality: {...}}``。"""
    from .dicom import ensure_series_nifti
    images: dict[str, dict] = {}
    try:
        recs = ensure_series_nifti(cdir)
    except Exception as e:                                        # noqa: BLE001
        if log is not None:
            log.append(f"{os.path.basename(cdir)}: DICOM 读取失败 {e}")
        return images
    for r in recs:
        mod = guess_modality(r.get("desc") or "") or guess_modality(os.path.basename(
            os.path.dirname(r["path"]))) or guess_modality(r["path"])
        if mod is None:
            continue
        if mod not in images:
            images[mod] = {"path": r["path"], "series_uid": r.get("series_uid") or mod,
                           "file": os.path.basename(r["path"]), "from_dicom": True}
    return images


def scan_real(root: str, limit_cases: int | None = None,
              struct_tables: dict[str, dict] | None = None,
              log: list | None = None,
              series_types: dict | None = None,
              mask_by_acc: dict | None = None,
              dropped_out: list | None = None,
              desc_index: dict | None = None) -> list[dict]:
    """扫描真实影像：一级目录 = 检查号；其下收集影像（NIfTI/DICOM）与掩码。

    平台下发的 ``training/annotation/SeriesType.xlsx``（与病例目录**同层**）
    会被一次性读入并用于识别**模态与掩膜**：官方数据的序列目录名与文件名都是
    UID（``2.25.*``），只靠关键词会得到"整批 other、掩膜全无"，
    而病例数与目录结构看起来完全正常。

    传入 ``dropped_out``（一个 list）时，**被剔除**的病例（有影像目录、却一路可用
    序列都没认出来）会追加进去，供调用方写进报告：剔除必须可追溯，
    不能只留一行 stdout（否则"病例数突然少了 N 例"无人能解释）。
    """
    cases: list[dict] = []
    #: 有影像目录、但一路可用模态都没认出来的病例（官方「备注」多为 `序列缺失跳过`
    #: / `构建失败跳过`）—— 它们不进清单，只用于最后汇报（见 :func:`_report_skip_reasons`）。
    _dropped_no_series: list[dict] = []
    root = resolve_case_root(root)                 # 填高一层（如 .../training）时自动下钻
    if not os.path.isdir(root):
        return cases
    assert_case_root(root)
    if series_types is None:
        series_types = read_series_types(root)
    if series_types:
        print(f"[probe] 已读取序列类型映射：{len(series_types)} 条（来源："
              f"{find_named_table(SERIES_TYPE_TABLE, root) or SERIES_TYPE_TABLE}）",
              flush=True)
    # UID 单键回退索引：一次建好、全病例复用（表可能上万行，别放进每病例的循环里）
    uid_index = build_uid_index(series_types)
    entries = sorted(e for e in os.listdir(root)
                     if os.path.isdir(os.path.join(root, e))
                     and is_official_accession(e)                # 只认大赛检查号（见下）
                     and e.lower() not in SPECIAL_SOURCE_DIRS)   # fake/compositing/duplicate 是"来源"不是检查号
    # ---- 大赛数据闸门 ----
    # 平台上 ``<数据根>/<检查号>/…`` 的检查号必然是 32 位十六进制。非大赛目录
    # （别的数据集、随手堆着 nii 的目录）在这里就被挡在**读取之前**：读了会污染
    # 训练与指标；报出来会把排查方向带偏。所以这里只拦、不列出它们的名字。
    if not entries:
        _dirs = sorted(e for e in os.listdir(root)
                       if os.path.isdir(os.path.join(root, e)))
        if not _dirs:
            print(f"[probe] ⚠️ 数据根 {root} 下没有子目录 → 0 例"
                  f"（影像存储可能没挂上，见 docs/DATASET_ROOT_TROUBLESHOOT.md）",
                  flush=True)
            return cases
        raise ValueError(
            f"数据根 {root} 不是大赛数据布局：一级子目录里没有 32 位十六进制的检查号"
            f"（如 0050d79429cf4d86907dc8c4a34cbf04）。当前 {len(_dirs)} 个子目录、"
            f"其中 {sum(1 for e in _dirs if e.lower() in _NON_CASE_DIRS)} 个是本工程"
            f"自己的目录（annotation/original/fake/compositing/duplicate 等）。"
            f"官方布局：训练集根=…/datasets/training（其下 annotation/）、"
            f"验证集根=…/datasets/verification（其下 original/），"
            f"病例目录名就是检查号本身。")
    skipped_sources = [e for e in sorted(os.listdir(root))
                       if os.path.isdir(os.path.join(root, e))
                       and e.lower() in SPECIAL_SOURCE_DIRS]
    if skipped_sources:
        print(f"[probe] 已按官方约定跳过异常影像目录：{skipped_sources}"
              f"（其内容与主目录同构，按来源区分而非当成检查号）", flush=True)
    for acc in entries:
        cdir = os.path.join(root, acc)
        images, mask_entries, unknown = _collect_nifti(
            cdir, acc, series_types,
            (mask_by_acc or {}).get(acc.casefold()) or (mask_by_acc or {}).get(acc),
            uid_index, desc_index=desc_index)
        if not images:                                            # 纯 DICOM 检查
            images = _collect_dicom(cdir, log)
        if not images and not mask_entries:
            continue
        # 掩码 → 角色并集
        masks: dict[str, dict] = {}
        for role, mod, path, uid in mask_entries:
            e = masks.setdefault(role, {"paths": [], "metas": [], "modality": mod})
            e["paths"].append(path)
            e["metas"].append({"path": path, "series_uid": uid, "modality": mod})
        if not struct_tables and not masks and not images:
            continue
        labels = {}
        if struct_tables:
            # 查表要覆盖全部等价写法：原样 / 大小写折叠 / 去前导零 / 归一化键。
            # 只查原样时，目录名 `C0E1F8F2-53BA-45BE` 与表里 `c0e1f8f2-53ba-45be`
            # 互相看不见 —— 表现为"表解析出 N 行，但每例 labels 全空"，
            # 报告里 `label_field_counts: {}` 而 `n_structured_rows` 正常，最难查。
            for key in (acc, acc.casefold(), acc.lstrip("0"),
                        acc.lstrip("0").casefold(), id_key(acc)):
                if key in struct_tables:
                    labels = structured_from_row(struct_tables[key])
                    break
        # unknown_series：认不出模态的序列清单。评测集没有标注表时，
        # 数据集侧（``dataset.pick_series``）会读它们的体素用统计模型判模态。
        # 「备注」（`STUDY->CLINICAL->备注`）提到病例级：它是官方"该检查是否被
        # 跳过"的标记（见 :data:`labels.SKIP_REASONS`），原先全工程零引用 ——
        # 提到顶层后，报告能统计、数据集侧也能说明这例为什么被跳过。
        skip = str((labels or {}).get("SkipReason") or "")
        # ★ 无可用序列的病例**不进清单**：`images` 为空意味着这个检查的序列一路
        #   都没认出模态（官方备注里 `序列缺失跳过` / `构建失败跳过` 就是这类），
        #   任何任务都用不了它；留在清单里只会在训练取样时由
        #   `dataset.build_case_volume` 抛 RuntimeError 中断整跑。
        #   这里剔除并**响亮报数**（不静默），原因见 `_report_skip_reasons`。
        if not images:
            _dropped_no_series.append({"accession": acc, "skip_reason": skip})
            continue
        # ★★ `images` 非空 **不等于** 有可用输入通道 —— 这是"训练跑到一半随机崩"的根因：
        #    类型表里**明写 `其他`** 的病例，`images` 是 `{"other": {...}}`（非空！
        #    而且它被**刻意**排除在 `unknown_series` 之外，因为"其他"是权威排除、
        #    不是"没认出来"），只有 DWI/ADC/SWI 的病例同理。两者都让
        #    `dataset.pick_series` 一个通道都挑不出 → `build_case_volume` 抛
        #    RuntimeError（在 DataLoader worker 里、epoch 中途，整跑一起挂；
        #    shuffle 之下看起来像随机崩）。
        #    口径（**全放开**）：这类病例**照常进清单、照常进训练** —— `build_case_volume`
        #    会借该例任意一路影像的几何把 4 个通道置零（掩膜仍是真值）。这里只**打标记 +
        #    报数**，让"有多少例输入侧是全空的"始终看得见（见 README §7.2）。
        no_input = not has_input_modality(images, unknown)
        cases.append({"accession": acc, "dir": cdir, "images": images,
                      "masks": masks, "labels": labels,
                      **({"unknown_series": unknown} if unknown else {}),
                      **({"skip_reason": skip} if skip else {}),
                      **({"no_input_channel": True} if no_input else {})})
        if limit_cases and len(cases) >= limit_cases:
            break
    _report_skip_reasons(cases, _dropped_no_series)
    if dropped_out is not None:
        dropped_out.extend(_dropped_no_series)
    return cases


def _report_skip_reasons(cases: list[dict], dropped: list[dict]) -> None:
    """汇报「备注」（``STUDY->CLINICAL->备注``）、"组不出序列"与"无输入通道"的统计。

    为什么必须报：官方按 `备注` 剔除病例（``序列缺失跳过`` / ``构建失败跳过`` / …），
    本地若照单全收，训练与评测的**分母**就与线上不一致 —— 而"多算/少算了几例"
    本身不会以任何形式报错。这里只统计事实，不改动清单内容。
    """
    reasons = Counter(str(c["skip_reason"]) for c in cases if c.get("skip_reason"))
    lost = Counter(str(c.get("skip_reason") or "模态未识别（无 SeriesType 记录 / 影像缺失）")
                   for c in dropped)
    # `images` 非空、却一个输入通道都凑不出的病例（类型表明写 `其他` / 只有 DWI 等）。
    # 它们**留在清单里、也照常进训练**（全放开口径），这里只是把数量报出来。
    no_ch = [c["accession"] for c in cases if c.get("no_input_channel")]
    if not reasons and not lost and not no_ch:
        return
    if reasons:
        hard = [f"{r}({n})" for r, n in reasons.items() if is_hard_skip(r)]
        print(f"[probe] 备注标记「已跳过」的病例 {sum(reasons.values())} 例：{dict(reasons)}"
              f"；其中影像不可用的原因 {hard or '无'} —— "
              f"官方按备注剔除，本地**仍进清单**（阴性数据还要当检测负样本用），"
              f"若线上口径是全部剔除，评测分母会与线上不一致", flush=True)
    if lost:
        print(f"[probe] 已从清单剔除「组不出序列」的病例 {sum(lost.values())} 例："
              f"{dict(lost)} —— 这类病例一个通道都凑不齐（`pick_series` 会抛 "
              f"RuntimeError 中断训练）；若本该有影像，见 README §7.2", flush=True)
    if no_ch:
        print(f"[probe] 清单内有 {len(no_ch)} 例**没有真输入通道**"
              f"（序列被类型表明写为 `其他`，或只有 DWI/ADC/SWI，例如 {no_ch[:3]}）："
              f"按**全放开**口径照常进训练 —— `build_case_volume` 借该例任意一路影像的"
              f"几何把 4 个通道置零（掩膜仍是真值、任务空间不受影响）。"
              f"代价是这一例回传近噪声梯度，数量见报告 `cases_without_input_channel`"
              f"（见 README §7.2）", flush=True)


def merge_special_cases(cases: list[dict], special: dict, log: list | None = None,
                        series_types: dict | None = None,
                        desc_index: dict | None = None) -> list[dict]:
    """把 ``annotation/{fake,Composition}`` 中**未出现在真实影像目录**的病例补进清单。

    否则目标一/二的正样本可能一例都匹配不上（``SpecialImageDataset`` 找不到影像），
    特殊影像头仍然训不起来。
    """
    by = {c["accession"]: c for c in cases}
    ann = special.get("annotation_dir")
    if not ann:
        return cases
    uid_index = build_uid_index(series_types)          # UID 单键回退索引（见 _collect_nifti）
    for cls, key in (("fake", "fake_cases"), ("Composition", "composition_cases")):
        base = os.path.join(ann, cls)
        for ident in (special.get(key) or []):
            if ident in by or not is_official_accession(str(ident)):
                continue                       # 同上：非大赛检查号不读
            d = os.path.join(base, str(ident))
            if not os.path.isdir(d):
                continue
            imgs, masks, unknown = _collect_nifti(d, str(ident), series_types,
                                                  uid_index=uid_index,
                                                  desc_index=desc_index)
            if not imgs:
                imgs = _collect_dicom(d, log)
            if not imgs:
                continue
            c = {"accession": ident, "dir": d, "images": imgs, "masks": {},
                 "labels": {}, "special": cls,
                 **({"unknown_series": unknown} if unknown else {}),
                 # 特殊影像病例同样可能"没有真输入通道"：它们要的是整脑视图
                 # （`SpecialImageDataset`），全放开口径下也照收（零通道整脑视图）。
                 **({} if has_input_modality(imgs, unknown) else {"no_input_channel": True})}
            cases.append(c)
            by[ident] = c
    return cases


def probe(root: str, limit_cases: int | None = None, sample_geometry: int = 8,
          phase: str = "train") -> dict:
    log: list[str] = []
    _desc_used.clear()                          # 同一进程内多次探测（测试）时不串场
    # 先把根定到病例层：否则下面读标注表、列检查号都会落在空的父目录上，
    # 报告里出现"0 例 + 0 张表"，看起来像数据没挂载，实际只是根填高了一层。
    root = resolve_case_root(root)
    # "按取值找检查号列"需要磁盘上真实存在的检查号（列名叫什么都不影响），
    # 这里先轻量列一次目录名，口径与 scan_real 一致（一级子目录、排除 annotation）。
    # 目录名统一归一化后再传：表内取值会经 id_key 归一化，两侧不同口径会
    # 一条都对不上（哈希型检查号 C0E1F8F2-53BA-45BE 就是这么栽的）。
    known_ids = ({id_key(e) for e in os.listdir(root)
                  if os.path.isdir(os.path.join(root, e))
                  and is_official_accession(e)}            # 只认大赛检查号（见 labels）
                 if os.path.isdir(root) else set())
    # 天坛参考实现那几张表（`1_abnormal` / `2_duplicate` / `4_masklabel` /
    # `5_characteristics`）**不是赛道四数据集的内容**，只在附近有（如团队工作区
    # labels/）时顺手用上：字段金标准、掩膜名表都在这里，靠目录名或关键词猜不出来。
    # 它们的 `工作区兼容表` **不再参与**（模态只认数据集自带的 `SeriesType.xlsx`，
    # 上面 read_series_types 已读；工作区那份取值更粗，读了会把 T2WI/T2-Flair 压平）；
    # 数据集里的字段金标准在 `annotation/脑胶质瘤标注结果-训练集.xlsx`
    # （下面 find_structured_tables 会找到）。
    # `2_duplicate.xlsx` 同理**只在数据集没给金标准文件时兜底**（见下方 special 段）：
    # 格式说明写明重复金标准就放在 `annotation/duplicate/` 目录里，优先读它。
    label_files = find_official_labels(root)
    if label_files:
        print("[probe] 官方标注表：" + ", ".join(
            f"{k}={os.path.basename(v)}" for k, v in label_files.items()), flush=True)

    chars = label_files.get("characteristics")                    # 字段金标准（官方列名）
    tables = ([chars] if chars else []) + [t for t in find_structured_tables(root)
                                          if t != chars]
    struct = {}
    for t in tables:
        try:
            struct.update(read_structured_table(t, known_ids=known_ids))
        except Exception as exc:                                  # noqa: BLE001
            # 读表异常**不能静默**：吞掉之后报告里只剩 `label_field_counts: {}`，
            # 看起来像"表里没数据"，实际是解析期就失败了（缺 openpyxl / 文件损坏 /
            # 加密 xlsx）—— 不打印异常就只能靠反复猜。
            print(f"[probe][告警] 读金标准表失败 {os.path.basename(t)}："
                  f"{type(exc).__name__}: {exc}", flush=True)
    # 字典里同一行会有多个键（原值 / 去前导零 / 大小写折叠），
    # 直接 len() 会把"行数"报成实际的两倍以上，把诊断带偏 —— 按**唯一记录**计数。
    n_struct_rows = len({id(v) for v in struct.values()})

    # 模态的**第二条来源**（旁证）：标注表序列级 / ROI 级别的「序列描述」。它承载的
    # 就是那 5 类模态取值，且行键是 ``序列UID``（与磁盘文件名主干同源，能直接对上）。
    # `SeriesType.xlsx` 拿不到时（或表里缺这个 UID）这是唯一还能**批量**判模态的正规
    # 线索 —— 剩下的只能一路走到体素判别模型，甚至整批 unknown。
    desc_index = desc_index_from_records({id(v): v for v in struct.values()}.values())
    if not desc_index:
        # 直读兜底：序列级 / ROI 级的行只有在"**检查号列能对上磁盘上的病例**"时才会挂进
        # 记录（见 :func:`labels._find_case_column`）。检查号口径不一致时（目录名是哈希、
        # 表里是原始检查号）整批子行挂不上 → 记录里 `__roi_rows__` 是空的、旁证一条都拿不到，
        # 而表其实就在旁边。``序列UID → 序列描述`` **与检查号无关**（UID 两边同源），
        # 所以这里直接读文件仍能救回来。
        for _t in tables:
            if str(_t).lower().endswith((".xlsx", ".xlsm")):
                desc_index = read_series_desc_index(_t)
                if desc_index:
                    print(f"[probe] 「序列描述」旁证改从 {os.path.basename(_t)} **直读**："
                          f"{len(desc_index)} 条序列 UID（子行没挂到病例上，"
                          f"但 UID→描述 与检查号无关）", flush=True)
                    break
    if desc_index:
        print(f"[probe] 已从标注表读到「序列描述」（模态旁证）：{len(desc_index)} 条序列 UID"
              f"（列名 DetailDescription / SeriesDescription / 序列描述）", flush=True)

    series_types = read_series_types(root)

    # 掩膜：官方 `4_masklabel.xlsx` 的 Maskname（文件名任意，靠关键词认不出）
    mask_by_acc: dict[str, dict[str, list[str]]] = {}
    if label_files.get("mask"):
        for (acc, uid), names in read_mask_table(label_files["mask"]).items():
            slot = mask_by_acc.setdefault(acc.casefold(), {})
            slot[uid] = names
            slot[uid.casefold()] = names
    # 异常影像（fake/compositing/duplicate）的标注来自官方 `1_abnormal.xlsx` 的 Label
    abnormal = (read_abnormal_table(label_files["abnormal"])
                if label_files.get("abnormal") else {})

    special = scan_special(root)
    # 重复金标准的来源顺序：**数据集自带的优先** —— `annotation/duplicate/` 下的金标准
    # 文件（每行 `src_img, desc_img`，两个值是重复影像的**检查号**），这是赛道四格式
    # 说明的原文口径；**只有数据集里一个金标准文件都没有**时，才退到团队工作区的
    # `2_duplicate.xlsx`。
    # ⚠️ 反过来（工作区表覆盖数据集）是**串表**：`2_duplicate.xlsx` 是天坛参考实现的表，
    #    里面是**别的数据集**的检查号 —— 表现是"有正样本、却一条都没匹配上"，
    #    比"缺正样本"更难查（缺正样本至少 evaluate 会明确提示）。
    if special["gold_pairs"]:
        print(f"[probe] 数据集自带重复金标准：{len(special['gold_pairs'])} 对"
              f"（{', '.join(os.path.basename(f) for f in special['gold_files'])}）", flush=True)
    elif special["gold_files"]:
        print(f"[probe][告警] duplicate/ 下有疑似金标准文件、但没解析出任何检查号对："
              f"{', '.join(special['gold_files'])}；"
              f"预期格式为每行 `src_img, desc_img`", flush=True)
    if not special["gold_pairs"] and label_files.get("duplicate"):
        official_pairs = [[a, b] for a, b in read_duplicate_pairs(label_files["duplicate"])]
        if official_pairs:
            special["gold_pairs"] = official_pairs
            print(f"[probe][告警] 数据集内没找到重复金标准，已退用工作区表 "
                  f"{os.path.basename(label_files['duplicate'])}：{len(official_pairs)} 对"
                  f"（⚠️ 它可能属于别的数据集，检查号对不上时重复任务的正样本会全废）",
                  flush=True)

    # `_no_series`：被剔除的"组不出序列"病例（有目录、却一路模态都没认出来）。
    # 必须进报告：否则"扫到了 30 例检查、清单里只有 27 例"这件事无人能解释。
    _no_series: list = []
    cases = scan_real(root, limit_cases, struct, log, series_types, mask_by_acc,
                      _no_series, desc_index=desc_index)
    cases = merge_special_cases(cases, special, log, series_types,
                                desc_index=desc_index)

    mod_counter, mask_counter, label_counter = Counter(), Counter(), Counter()
    geom_samples = []
    n_unknown_series = n_unknown_cases = n_declared_other = n_no_input = 0
    for c in cases:
        mod_counter.update(c["images"].keys())
        mask_counter.update(c["masks"].keys())
        label_counter.update(c["labels"].keys())
        # 认不出模态的序列数：评测集（无标注表、UID 目录名）会整批落在这里。
        # 这是"评测期要不要靠模型判模态"的唯一可见指标 —— 不报出来就只能等
        # 训练时崩「无任何可用序列」才发现。
        n_u = len(c.get("unknown_series") or [])
        n_unknown_series += n_u
        n_unknown_cases += int(n_u > 0)
        # 被类型表**明确标为"其他"**的病例数。它与"认不出"必须分开统计：
        # 前者是权威结论（该排除），后者才需要模型兜底；混在一起会让人
        # 以为"表没接上"，然后跑去重配 labels_dir 白折腾。
        n_declared_other += int(bool((c["images"].get("other") or {}).get("declared_other")))
        # `images` 非空、但一个**真输入通道**都凑不出的病例数（全放开：照训，仅报数）。
        n_no_input += int(bool(c.get("no_input_channel")))
        if len(geom_samples) < sample_geometry:
            for mod, meta in list(c["images"].items())[:1]:
                geom_samples.append({"accession": c["accession"], "modality": mod,
                                     **_probe_nifti(meta["path"])})

    _report_desc_fallback()                    # 「序列描述」旁证的命中量（0 条不出声）

    # 结构化字段金标准为空时**说清是哪一种空**。
    # 三种情形在本地看起来都是 `label_field_counts: {}`，但处理方式完全不同：
    # 没表（数据根偏了一层）/ 有表但认不出检查号列 / 有表有检查号但列名没映射。
    labels_hint = ""
    if label_counter:
        labels_hint = ""
    elif not tables and phase == "val":
        # 官方验证集**没有**字段金标准表（实测 ``verification/original/`` 下只有
        # ``SeriesType.xlsx``）→ 验证集 labels 全空是**预期**，不是配置错。
        # 不说清的话，下面那条"表在别处"的提示会把人引去满磁盘找一张不存在的表。
        labels_hint = ("这是**验证集**：官方验证集实测只有 SeriesType.xlsx、**没有**字段金标准表"
                       " → 本级 labels 全空属**预期**，不必去找表。分类头只由训练折监督；"
                       "官方评估口径是 Dice/NSD/HD95 + 重复影像（scripts/04_eval.sh --split external）。"
                       "若平台后续单独发布验证集标注表：放进验证集目录或 "
                       "export GLIOMA_LABELS_DIR=<含表目录>，重跑本探针即自动接上")
    elif not tables:
        labels_hint = ("没找到 csv/xlsx 金标准表（已搜数据根/父/祖父，含 annotation / "
                       "original / 标注结果 这类子目录）；若表确在别处："
                       "export GLIOMA_LABELS_DIR=<含表的目录>（**显式**指定 —— "
                       "不做任何隐式搜索，隐式翻别处会串表），"
                       "或把数据根定到与表同级的那一层")
    elif not struct:
        # 0 行**不一定**是表坏了：扫描范围里的"杂物表"（本地伪造的对照表、临时导出
        # 的样本、别的赛道的表）同样会被当成候选金标准表 —— 它们连行键列都没有，
        # 解析必然 0 行；此时只说"列名有问题"会让人去改一张根本没用的表。
        # 所以把**所在目录**一并报出：目录是不是"数据该在的地方"，一眼可见。
        labels_hint = (f"找到 {len(tables)} 个表但一行都没解析出来："
                       f"表里需要有 检查号/AccessionNumber/PatientId 之类的列"
                       f"（候选：{', '.join(os.path.basename(t) for t in tables[:3])}；"
                       f"所在目录：{', '.join(sorted({os.path.dirname(t) for t in tables[:3]}))}。"
                       f"若这些是本地伪造/临时导出的表，把它们移出扫描范围即可）")
    else:
        # 有表、也解析出了行，但**没有一例因此拿到字段**。两种原因的处理方式完全不同：
        # 表属于另一份数据（检查号一条都对不上，例如把训练集的 `5_characteristics.xlsx`
        # 也搜进来了）vs 检查号对上了但列名没映射到规范字段。
        # 只报后一种会把前者说成"列名有问题"，让人反复改列名 —— 先看有没有一行落到磁盘上。
        n_hit = len(known_ids & set(struct))
        if not n_hit:
            labels_hint = (f"表解析出 {n_struct_rows} 行，但与本数据集的检查号**一条都对不上**："
                           f"表多半属于另一份数据/另一个阶段（候选："
                           f"{', '.join(os.path.basename(t) for t in tables[:3])}）——"
                           f"先确认数据根与表是不是配套的")
        else:
            labels_hint = (f"表解析出 {n_struct_rows} 行、命中 {n_hit} 个磁盘检查号，"
                           f"但列名没映射到规范字段；需要 病理结果 / location_of_lesion / "
                           f"lesion_morphology / tumor_t2wi_signal_intensity 这类列")

    # ★ 键口径自检：查表失败只有三种成因（检查号对不上 / 序列号对不上 / 列认错了）。
    #   把"表里的键"和"磁盘上的名字"各抽几个摆在一起 + 算命中率 ——
    #   2026-09-24 验证集就栽在这里：表读到了 1735 条，但报告只说 0 命中，
    #   看不出是键对不上还是取值不对，只能反复猜。
    _tbl_accs = sorted({k[0] for k in series_types
                        if isinstance(k, tuple) and k[0]})[:2]
    _tbl_uids = sorted({k[1] for k in series_types
                        if isinstance(k, tuple) and len(k) > 1})[:2]
    _disk_accs = [c["accession"] for c in cases[:2]]
    _disk_uids = [m.get("series_uid") for c in cases[:2]
                  for m in (c.get("unknown_series")
                            or [{"series_uid": m.get("series_uid")}
                                for m in (c.get("images") or {}).values()])[:3]]
    _tbl_acc_set = {k[0] for k in series_types if isinstance(k, tuple)}
    _tbl_uid_set = {k[1] for k in series_types if isinstance(k, tuple) and len(k) > 1}
    _hit_acc = sum(1 for c in cases[:50] if id_key(c["accession"]) in
                   {id_key(a) for a in _tbl_acc_set})
    _n_series_50 = _hit_uid = 0
    for c in cases[:50]:
        for m in (c.get("unknown_series")
                  or [{"series_uid": mm.get("series_uid")}
                      for mm in (c.get("images") or {}).values()]):
            uid = m.get("series_uid")
            if not uid:
                continue
            _n_series_50 += 1
            _hit_uid += int(any(id_key(uid) == id_key(u) for u in _tbl_uid_set))

    report = {
        "root": os.path.abspath(root),
        "n_cases": len(cases),
        # 「扫到了检查目录、却一路可用序列都没认出来」而被剔除的病例。必须与
        # "压根没扫到病例"分开：前者要去解决 SeriesType.xlsx / 模态判别模型，
        # 后者才该查 --root。官方备注里的 `序列缺失跳过` / `构建失败跳过` 就是前者。
        "cases_dropped_no_series": len(_no_series),
        "no_series_reasons": dict(
            Counter(str(d.get("skip_reason") or "模态未识别（无 SeriesType 记录）")
                    for d in _no_series)),
        "no_series_samples": [str(d.get("accession")) for d in _no_series][:20],
        "structured_tables": tables,
        "n_structured_rows": n_struct_rows,
        # 官方 5 张标注表的命中情况（看不到某个键 = 那类标注没找到）
        "official_label_files": {k: os.path.basename(v)
                                 for k, v in label_files.items()},
        "n_abnormal_rows": len(abnormal),
        "abnormal_label_counts": dict(Counter(abnormal.values())),
        # 序列类型表命中数：0 且在官方数据上 → 模态/掩膜必然认不出，
        # 先解决这个再谈训练（"病例数正常但全 other"就是这个原因）
        "series_type_rows": len(series_types),
        # ★ 键口径自检（表读到了、但一条都没查到时靠它定位）：
        #   检查号/序列号命中数 + 两边各自的样例
        "series_type_key_hits": {"acc": _hit_acc, "uid": _hit_uid,
                                  "n_series": _n_series_50},
        "series_type_acc_samples": _tbl_accs,
        "series_type_uid_samples": _tbl_uids,
        "disk_acc_samples": _disk_accs,
        "disk_series_samples": _disk_uids,
        "modality_counts": dict(mod_counter),
        # 认不出模态的序列总数 / 涉及病例数。评测集没有标注表时它会等于"序列总数"，
        # 此时全靠 data/modality_model.json 兜底（见 README.md §7.2）
        "unknown_series_total": n_unknown_series,
        "cases_with_unknown_series": n_unknown_cases,
        # 类型表里写着"其他"的病例数（`SeriesType.xlsx` 常见）：**不是**缺表信号，
        # 这些序列不参与模态判别（见 labels.EXPLICIT_OTHER_VALUES）
        "cases_with_declared_other_series": n_declared_other,
        # `images` 非空、却一个**输入通道**都凑不出的病例数（类型表明写"其他"、或只有
        # DWI/ADC/SWI）。它们留在清单里供推理写出合规空掩码，训练侧会剔除并报数 ——
        # 原先这类病例会在取样时于 DataLoader worker 里抛 RuntimeError 打断整跑。
        "cases_without_input_channel": n_no_input,
        # 「序列描述」旁证读到的 UID 数：`SeriesType.xlsx` 不在手时的第二条模态来源。
        # 0 且 series_type_rows=0 → 两路都没接上，模态只能靠体素判别模型。
        "series_desc_rows": len(desc_index),
        "mask_role_counts": dict(mask_counter),
        "label_field_counts": dict(label_counter),
        "labels_hint": labels_hint,
        "special": {k: (v if not isinstance(v, list) else f"{len(v)} items")
                    for k, v in special.items()},
        "geometry_samples": geom_samples,
        "missing_t1c": [c["accession"] for c in cases if "t1c" not in c["images"]][:20],
        "missing_flair": [c["accession"] for c in cases
                          if "flair" not in c["images"] and "t2" not in c["images"]][:20],
        "no_labels": [c["accession"] for c in cases if not c["labels"]][:20],
        "dicom_log": log[:20],
    }
    return {"report": report, "cases": cases, "special": special,
            # phase="val" 时记 `local/<验证集目录名>/val` —— 验证集清单不参与训练，
            # 但同样要可追溯数据来源（评估侧 assert_data_source(phase="val") 会比对）。
            "data_source": data_source_tag(root, phase=phase)}


def main() -> None:
    ap = argparse.ArgumentParser()
    paths = load_paths()
    ap.add_argument("--phase", choices=("train", "val"), default="train",
                    help="train=训练集（默认，写 data/manifest.json）；"
                         "val=官方验证集（写 data/manifest_val.json，"
                         "供评估侧 external 分支使用）")
    ap.add_argument("--root", default=None,
                    help="数据根；train 默认 DATASET_ROOT / raw.track4，"
                         "val 默认 VAL_ROOT / raw.val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-cases", type=int, default=None)
    a = ap.parse_args()

    if a.phase == "val":
        a.root = a.root or val_root()
        if not a.root:
            raise SystemExit(
                "[probe] ✗ 未配置验证集数据根：export VAL_ROOT=<验证集目录>，"
                "或在 configs/paths.yaml 的 raw.val 填写（验证集布局见该处注释）")
        a.out = a.out or paths.get("manifest_val") or "data/manifest_val.json"
    else:
        a.root = a.root or os.environ.get("DATASET_ROOT") or paths["raw"]["track4"]
        a.out = a.out or paths["manifest"]

    res = probe(a.root, a.limit_cases, phase=a.phase)
    out = resolve(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cases": res["cases"], "special": res["special"], "report": res["report"],
                   # 数据源标识：训练前会与本机数据源比对，防止"用本地/公开数据清单训练官方数据"
                   "data_source": res.get("data_source") or data_source_tag(a.root, phase=a.phase),
                   "data_root": os.path.abspath(a.root)},
                  f, ensure_ascii=False, indent=1)

    print(json.dumps(res["report"], ensure_ascii=False, indent=1))
    _kind = "验证集清单" if a.phase == "val" else "训练清单"
    print(f"\n[probe] {_kind} -> {out}  病例 {res['report']['n_cases']}")
    if a.phase == "val":
        print("[probe] ℹ️ 已生成验证集清单：scripts/04/14/15/16 会自动切到 external 分支"
              "（全折集成，不做留一；最终指标以官方验证集为准）。"
              "想回退折内 val：删除该清单或清空 raw.val/VAL_ROOT。")
    if res["report"]["n_cases"] == 0:
        _dropped = res["report"]["cases_dropped_no_series"]
        if _dropped:
            # 与"压根没扫到病例"是**两个完全不同的故障**：这里的目录结构是对的，
            # 只是序列一路都没认出模态。指错方向会让人白改 --root。
            print(f"[probe] ⚠️ 扫到 {_dropped} 例检查，但**一路可用序列都没认出来**，"
                  f"已全部剔除：{res['report']['no_series_reasons']} —— 先看 "
                  f"series_type_rows={res['report']['series_type_rows']}"
                  f"（为 0 就是数据信息表没接上，见 README §7.2），"
                  f"而不是去改 --root")
        else:
            print("[probe] ⚠️ 未找到病例：请确认 --root 指向含'检查号目录'的数据根（其内应有 NIfTI 或 DICOM）")
    if not res["report"]["label_field_counts"]:
        # 目标三/目标四的监督信号全在这里；为 0 就意味着分类头学不到东西，
        # 而训练照样能跑完（loss 只统计有 mask 的样本）——必须显式提醒。
        # 验证集没有字段金标准表是**预期**（官方只给 SeriesType.xlsx），
        # 用 ⚠️ 会让人以为自己配错了，去翻一张不存在的表。
        _mark = "ℹ️" if a.phase == "val" else "⚠️"
        print(f"[probe] {_mark} 结构化字段金标准为空（label_field_counts={{}}）："
              f"{res['report']['labels_hint']}")
    if res["report"]["modality_counts"].get("other") and not res["report"]["series_type_rows"]:
        # 序列类型表只在**数据集里**（与病例目录同层），走到这里就是没找到 ——
        # 而不是我们没看那几个目录（工作区那份 工作区兼容表 已不参与）。
        print(f"[probe] ⚠️ 有序列落到 other 且没读到数据信息表（{SERIES_TYPE_TABLE}）："
              "先确认数据根指向的是含 annotation/ 的那一层（表与病例目录同层）；"
              "表确在别处就 export GLIOMA_LABELS_DIR=<含该表的目录>（**显式**指定）"
              "再重跑本探针；表也没有时走体素判别兜底（见下一条）。"
              "排查步骤：README.md §7.2")
    elif (res["report"]["modality_counts"].get("other")
          and res["report"]["series_type_rows"]
          and not ({"t1c", "flair", "t2", "t1"} & set(res["report"]["modality_counts"]))):
        # ★ 表读到了、却一条都没查到 —— 2026-09-24 验证集的真实故障：
        #   表 1735 条读进来了，但 (检查号,序列号) 精确键和 UID 单键全部落空。
        #   只有三种成因：检查号对不上 / 序列号对不上 / 列认错了（acc↔uid 认反）。
        #   把两边的样例和命中率摆出来，一眼即可定位。
        kh = res["report"].get("series_type_key_hits") or {}
        print(f"[probe] ⚠️ 数据信息表读到了 {res['report']['series_type_rows']} 条，"
              f"却没有一路序列因此认出模态（modality_counts="
              f"{res['report']['modality_counts']}）。\n"
              f"       键命中（前 50 例）：检查号 {kh.get('acc')} 例 / "
              f"序列号 {kh.get('uid')}/{kh.get('n_series')} 路\n"
              f"       表里样例：检查号 {res['report'].get('series_type_acc_samples')}"
              f" / 序列号 {res['report'].get('series_type_uid_samples')}\n"
              f"       磁盘样例：检查号 {res['report'].get('disk_acc_samples')}"
              f" / 序列目录 {res['report'].get('disk_series_samples')}\n"
              "       → 命中为 0 = 键口径不一致（检查号或序列号列对不上磁盘目录名；"
              "把上面两行样例贴出来即可定位）。\n"
              "         两边样例能对上 = 列认错了（acc/uid 认反）或取值不是"
              "T1/T1CE/T2-Flair/T2WI —— 用 scripts/30_inspect_table.py 摊开看前几行。",
              flush=True)
    if res["report"].get("cases_with_declared_other_series"):
        print(f"[probe] ℹ️ {res['report']['cases_with_declared_other_series']} 例含被类型表标为"
              f"『其他』的序列（不属于 T1/T2-FLAIR/T1CE），已排除、不交给体素模型猜；"
              f"这是正常现象，不必去补标注表")
    if res["report"].get("unknown_series_total"):
        # 评测集没有标注表 → 关键词必然全失效。这条路是**预期**的，
        # 关键是别让它静默：说清有多少路要走模型判别、模型在不在。
        from .modality_model import load_default_model
        has_model = load_default_model() is not None
        print(f"[probe] ℹ️ {res['report']['cases_with_unknown_series']} 例共 "
              f"{res['report']['unknown_series_total']} 路序列模态未知"
              f"（无标注表时的正常现象）→ 体素统计模型："
              f"{'已就绪 data/modality_model.json' if has_model else '❌ 缺失，请先跑 python3 scripts/31_train_modality_model.py --root <数据根>'}")


if __name__ == "__main__":
    main()
