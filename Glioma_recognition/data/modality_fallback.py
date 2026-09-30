"""模态识别兜底：**挑不出任何序列时**，改用随数据下发的 ``SeriesType.xlsx`` 重贴模态。

为什么需要它
------------
提交侧的模态来源是「``SeriesType.xlsx`` → 同名 sidecar → 目录名猜关键词」
（见 :mod:`data.loader`）。后两者在真实评测数据上**可能落空**：

* 序列目录名是 DICOM UID（``2.25.135...``），任何关键词都命中不了；
* sidecar 也不保证带 ``SeriesDescription`` / ``ProtocolName``。

两者落空、而 loader 又没读到表时，``select`` 挑不出序列 → 上层抛「无任何可用序列」
（规范 §9.1 不可降级错误）→ **整批评测失败**。此时唯一还能救的就是
``SeriesType.xlsx`` 本身——它**随数据一起下发**（训练/验证/测试集都带），
与病例目录**同层**；loader 正常路径已读过一次，这里在"要报错时"再兜一次。

⚠️ **不再读 ``3_serieslabel.xlsx``**（旧版的行为）：那张表与赛道四数据集无关，
读它只会把 ``T2WI``/``T2-Flair`` 静默压平成 ``T2``；且它躺在工作区 ``labels/`` 里，
一旦被用作兜底，就是"拿另一个目标的标签改写本任务的模态"——比报错更糟。

工作方式（只在"要报错"时介入，不报错则完全不动）
------------------------------------------------
:func:`data.series_selector.select` 先按原逻辑挑：挑到就**原样返回**——不读盘、
零额外开销。**只有挑不出任何序列**（= 上层即将抛「无任何可用序列」或降级推理）
时，才调本模块：

1. 定位 ``SeriesType.xlsx``（:func:`find_label_table`）——**数据优先**：
   从序列文件所在目录上溯 4 层，逐层查 ``<层>/SeriesType.xlsx`` 与
   ``<层>/{annotation,original}/SeriesType.xlsx``；都不在才退
   ``$GLIOMA_SERIES_TYPE_XLSX`` / ``$GLIOMA_LABELS_DIR`` 显式指定。
   **不扫工作区**（那只会捡到别的数据集/别的项目的同名表 → 串表）；
2. 读表并**按表路径缓存**（含"没找到"这一结论；表换了路径会自动重读）；
3. 两级匹配：``(检查号, 序列号)`` 精确键 → **``SeriesUid`` 单键回退**（表里的
   检查号与磁盘目录名口径不一致时，只有 UID 必然一致）；
4. 命中的序列把 ``Series.modality`` 换成表里的取值（如 ``T1CE(增强)``），
   交给原逻辑重挑。

只改 ``modality``（描述），**不动** ``metadata``：``series_selector._key_of`` 优先读
``metadata['modality']`` 且**不走关键词匹配**（直接 lower 当键用），把 ``T1CE`` 写进去
会得到 ``t1ce``，反而认不出来。

关掉它：``GLIOMA_MODALITY_FALLBACK=0``（默认开启）。
"""
from __future__ import annotations

import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from data.structures import Series, Study

__all__ = [
    "enabled",
    "find_label_table",
    "describe_sources",
    "recover_study",
]

#: 序列类型表文件名（数据集自带的数据信息）；大小写不敏感，另容忍 ``*SeriesType*.xlsx`` 改名
_OFFICIAL_NAME = "SeriesType.xlsx"

#: 数据目录里表可能待的位置：与检查号目录**同层**（平台契约）。
#: 容器名训练集是 ``annotation``、验证集是 ``original``，评测集的名字还未知 ——
#: 三个都试，而不是只认其中一个。
_DATA_TABLE_SUBDIRS = ("", "annotation", "original")

#: 从序列文件所在目录**上溯**的层数：表可能与影像同层，也可能在数据根那一层
_DATA_TABLE_MAX_UP = 4

#: 表头别名（归一化后做**子串**匹配，与 ``data.loader._read_series_types`` 同一风格）
_HEADER_ALIASES = {
    "acc": ("accessionnumber", "accession", "检查号", "检查编号", "病例号"),
    "uid": ("seriesinstanceuid", "seriesuid", "序列号", "序列uid"),
    "lab": ("seriestype", "序列类型", "模态", "序列描述", "序列名称"),
}

_CACHE: dict[str, Any] = {}


def enabled() -> bool:
    """兜底开关：``GLIOMA_MODALITY_FALLBACK=0`` 可关（默认开启）。"""
    return os.environ.get("GLIOMA_MODALITY_FALLBACK", "").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _norm(value: Any) -> str:
    return re.sub(r"\s+", "", str(value if value is not None else "")).casefold()


# --------------------------------------------------------------------------- #
# 表定位（**数据优先**，绝不扫工作区）
# --------------------------------------------------------------------------- #
def _table_in(folder: Path) -> Path | None:
    """``folder`` 下**直接**放着的序列表（大小写不敏感，容忍 ``SeriesType*.xlsx`` 改名）。"""
    exact = folder / _OFFICIAL_NAME
    if exact.is_file():
        return exact
    if not folder.is_dir():
        return None
    try:
        for candidate in sorted(folder.glob("*[Ss]eries[Tt]ype*.xlsx")):
            if candidate.is_file():
                return candidate
    except OSError:
        return None
    return None


def _levels(source_paths: Iterable[Path]) -> list[Path]:
    """序列文件 → 它所在目录及其上溯 ``_DATA_TABLE_MAX_UP`` 层（去重、保序）。"""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in source_paths:
        try:
            start = Path(raw).expanduser().resolve().parent
        except OSError:
            continue
        for folder in [start, *list(start.parents)[:_DATA_TABLE_MAX_UP]]:
            if folder not in seen:
                seen.add(folder)
                out.append(folder)
    return out


def find_label_table(source_paths: Iterable[Path] = ()) -> Path | None:
    """定位 ``SeriesType.xlsx``；``source_paths`` 传序列文件路径 —— **数据优先**。

    顺序：① ``source_paths`` 各自所在目录及其上溯 4 层，逐层查
    ``<层>/SeriesType.xlsx`` 与 ``<层>/{annotation,original}/SeriesType.xlsx``；
    ② 都落空才退显式指定：``$GLIOMA_SERIES_TYPE_XLSX``（表本身）→
    ``$GLIOMA_LABELS_DIR``（表所在目录）。

    **不扫工作区**：``$COMPETITION_WORKSPACE`` 下的 ``labels/`` 里可能躺着
    **另一个数据集**的同名表，捡到它 = 拿别的数据的检查号来查本数据
    （表读得出几千条、却一条都匹配不上，且日志里表的路径指向 ``labels/`` 而非数据目录）。
    """
    for level in _levels(source_paths):
        for sub in _DATA_TABLE_SUBDIRS:
            hit = _table_in(level / sub if sub else level)
            if hit:
                return hit
    explicit = os.environ.get("GLIOMA_SERIES_TYPE_XLSX", "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path
    env_dir = os.environ.get("GLIOMA_LABELS_DIR", "").strip()
    if env_dir:
        hit = _table_in(Path(env_dir).expanduser())
        if hit:
            return hit
    return None


# --------------------------------------------------------------------------- #
# 表读取
# --------------------------------------------------------------------------- #
def _read_rows(path: Path) -> dict[tuple[str, str], str]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        out: dict[tuple[str, str], str] = {}
        for sheet in workbook.worksheets:
            rows = list(sheet.iter_rows(values_only=True))
            if not rows:
                continue
            header = [_norm(cell) for cell in rows[0]]
            idx: dict[str, int] = {}
            for want, aliases in _HEADER_ALIASES.items():
                for i, cell in enumerate(header):
                    if any(alias in cell for alias in aliases):
                        idx[want] = i
                        break
            if not {"acc", "uid", "lab"} <= idx.keys():
                continue                              # 该 sheet 不是映射表
            for row in rows[1:]:
                try:
                    acc, uid, label = row[idx["acc"]], row[idx["uid"]], row[idx["lab"]]
                except IndexError:
                    continue
                if acc is None or uid is None or label is None:
                    continue
                key = (_norm(acc), _norm(uid))
                value = str(label).strip()
                if not value:
                    continue
                # 同一键冲突：保留**先出现**的取值（官方表偶有重复行，
                # 这里不 fail-fast——兜底模块的职责是尽力救回检查，
                # 而不是让整个评测因为一行脏标注失败；冲突会打进日志）
                if key in out and out[key] != value:
                    print(f"[selector][模态回退] 表中同键冲突 {key}: "
                          f"{out[key]!r} vs {value!r}（保留前者）", flush=True)
                    continue
                out[key] = value
        return out
    finally:
        workbook.close()


def _table_maps(table: Path | None) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """读表 → ``(精确键表, UID 单键索引)``；**按表路径缓存**（含"没找到"这一结论）。

    表换了路径（= 换成另一个数据集的表）会自动重读，不会拿旧内容继续匹配。
    """
    key = str(table) if table is not None else ""
    cache: dict[str, tuple[dict[tuple[str, str], str], dict[str, str]]] = \
        _CACHE.setdefault("tables", {})
    if key in cache:
        return cache[key]
    rows: dict[tuple[str, str], str] = {}
    index: dict[str, str] = {}
    if table is not None:
        try:
            rows = _read_rows(table)
        except Exception as exc:                       # noqa: BLE001
            print(f"[selector][模态回退] 读官方序列表失败 {table}: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            rows = {}
        for (_, uid), label in rows.items():
            index.setdefault(uid, label)
        print(f"[selector][模态回退] 已读官方序列表 {table}："
              f"{len(rows)} 条（UID 单键索引 {len(index)} 条）", flush=True)
    cache[key] = (rows, index)
    return cache[key]


def describe_sources(source_paths: Iterable[Path] = ()) -> str:
    """一句话自检"模态来源现在什么状态"（供上层报错文案使用）。

    ``source_paths`` 传**该病例各序列的文件路径**，报出的才是真正会被用到的那张表；
    不传时只能报"按环境变量找没找到"（候选退到 ``$GLIOMA_LABELS_DIR`` 等）。
    """
    table = find_label_table(source_paths)
    if table is None:
        return ("未找到数据集自带的 SeriesType.xlsx（已按数据优先搜过序列文件所在目录"
                "及其上溯 4 层的 SeriesType.xlsx / annotation/ / original/，"
                "再退 $GLIOMA_SERIES_TYPE_XLSX、$GLIOMA_LABELS_DIR；"
                "不扫工作区，以免串到别的数据集的同名表）")
    return f"数据集自带 SeriesType.xlsx={table}"


def _uid_candidates(series: Series) -> tuple[str, ...]:
    """按可靠性降序：磁盘目录名（官方契约的 ``{SeriesUid}``）→ sidecar UID → 文件名主干。

    ⚠️ **``sidecar`` 的 ``SeriesInstanceUID`` 必须仍然是一个候选**：
    ``data.loader`` 现在把 ``Series.series_uid`` 定为**磁盘推导值**
    （官方契约要求 —— 掩膜 URI 的 ``{SeriesUid}`` 要能解析回输入数据的目录名），
    于是 ``series.series_uid`` 与 ``parent.name``、``stem`` **三者变成同一个值**，
    去重后只剩一个候选。

    但**官方表里的 ``SeriesUid`` 列写的是哪一边，数据方并没有承诺** ——
    实测两种形态都存在（带装饰的 ``*2.25.…*`` 与纯 ``2.25.…``）。
    少列一个候选 = 把本来能匹配上的检查直接变成「挑不出序列」，
    所以这里只做**排序**、绝不做**裁剪**。
    """
    name = series.source_path.name
    stem = name[:-7] if name.lower().endswith(".nii.gz") else Path(name).stem
    sidecar = (getattr(series, "metadata", None) or {}).get("SeriesInstanceUID")
    return tuple(dict.fromkeys(
        str(value) for value in (
            series.series_uid,                 # = 磁盘目录名（loader 已按契约设定）
            series.source_path.parent.name,    # 同一值，保留以兼容 2 层/1 层布局
            stem,                              # 同一值，保留同上
            sidecar,                           # ← 关键：sidecar 原始 UID 仍要试
        ) if value
    ))


def _match(exact: dict[tuple[str, str], str], index: dict[str, str],
           accession: str, series: Series) -> str:
    """两级匹配：``(检查号, 序列号)`` 精确键 → **``SeriesUid`` 单键回退**。

    表里的检查号与磁盘目录名口径不一致时（前导零 / 大小写 / 全角），
    只有 DICOM UID 必然一致，所以精确键落空后必须退到 UID 单键。
    """
    uids = _uid_candidates(series)
    acc = _norm(accession)
    for uid in uids:
        value = exact.get((acc, _norm(uid)))
        if value:
            return value
    for uid in uids:
        value = index.get(_norm(uid))
        if value:
            return value
    return ""


def recover_study(study: Study) -> Study:
    """兜底入口：用 ``SeriesType.xlsx`` 给"认不出模态"的序列重贴描述。挑不出就原样返回。

    只处理**描述认不出模态**的序列；已有可用描述的序列一律不碰
    （保守起见，避免把本来能用的判断改坏）。
    """
    from data.series_selector import guess_modality

    if not enabled():
        return study
    sources = [series.source_path for series in study.series]
    table = find_label_table(sources)
    exact, index = _table_maps(table)
    if not exact and not index:
        if not _CACHE.get("warned_missing"):
            _CACHE["warned_missing"] = True
            print(f"[selector][模态回退] 挑不出序列，且{describe_sources(sources)}；"
                  f"保持原报错（可 export GLIOMA_LABELS_DIR=<表所在目录> 后重跑）"
                  f"【本进程只报这一次：后续同类检查不再打印，"
                  f"它描述的是**首次触发的那一例**，不代表整批都这样】",
                  flush=True)
        return study

    accession = study.accession_number
    changed: dict[str, str] = {}
    pending: list[tuple[Series, tuple[str, ...]]] = []
    for series in study.series:
        if guess_modality(series.modality) is not None:
            continue                                   # 原描述已能用 → 不动
        label = _match(exact, index, accession, series)
        if label:
            changed[series.series_uid] = label
        else:
            pending.append((series, _uid_candidates(series)))
    if not changed:
        # ⚠️ 必须把**两种完全不同的情况**分开报。旧版统一写成
        # 「挑不出序列，序列表在 X 但按 (检查号,序列号) / UID 都匹配不到」，
        # 于是「整例只有 DWI/ADC/SWI、原描述全都认得出来」这种**与表无关**的情况
        # 也被说成"表匹配不到"，把排查方向直接带偏。
        if not pending:
            # 没有任何序列需要重贴：原描述都认得出来，只是都不在目标模态集合里。
            if not _CACHE.get("warned_no_target"):
                _CACHE["warned_no_target"] = True
                print(
                    f"[selector][模态回退] 挑不出序列：{len(study.series)} 条序列的"
                    f"**原描述全都认得出来**，但都不在目标模态集合里"
                    f"（示例 study={accession!r} "
                    f"modality={[s.modality for s in study.series][:4]}）→ "
                    f"属于**该检查本来就没有目标模态**，与序列表匹配无关",
                    flush=True,
                )
            return study

        if not _CACHE.get("warned_unmatched"):
            _CACHE["warned_unmatched"] = True
            first_series, first_cands = pending[0]
            acc_rows = sum(1 for (a, _u) in exact if a == _norm(accession))
            # 候选**长度**单独列出：``*`` 在 Markdown 里会被吃掉，
            # 长度才是粘贴到聊天里之后仍然可信的信息。
            print(
                f"[selector][模态回退] 挑不出序列，序列表在 {table} 但 "
                f"{len(pending)}/{len(study.series)} 条序列按 "
                f"(检查号,序列号) / UID 都匹配不到"
                f"（示例 study={accession!r} "
                f"uid={[s.series_uid for s in study.series][:3]}）"
                f"；该检查号在表里命中 {acc_rows} 行"
                f"；首个未命中 desc={first_series.modality!r} "
                f"候选长度={[len(c) for c in first_cands]} "
                f"候选={list(first_cands)} "
                f"sidecar={(first_series.metadata or {}).get('SeriesInstanceUID')!r}"
                f"【本进程只报这一次：它描述的是**首次触发的那一例**；"
                f"若后面还能看到「已按官方表重贴」，说明其余检查匹配正常】",
                flush=True,
            )
        return study

    recovered = tuple(
        replace(series, modality=changed[series.series_uid])
        if series.series_uid in changed else series
        for series in study.series
    )
    # 同一检查的 select 会被调多次（建体积 → 指纹 → 降级路径），只记一条日志；
    # 这里**不缓存 Study 对象**——它会拖住 Series.image，几千例下来就是内存暴涨。
    logged: set = _CACHE.setdefault("logged", set())
    if accession not in logged:
        logged.add(accession)
        print(f"[selector][模态回退] study {accession!r} 原描述认不出模态，"
              f"已按官方表重贴：{changed}（共 {len(study.series)} 条序列）", flush=True)
    return replace(study, series=recovered)
