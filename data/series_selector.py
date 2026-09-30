"""序列选择：从 ``Study.series`` 中挑出各任务需要的模态。

规范要求把"序列选择"从 Loader 中独立出来（§5、§26），因为它是**任务相关**的：
Goal5 需要 T1 增强与 FLAIR/T2，Goal2 duplicate 需要全部序列做指纹。

本模块只做"从已有 Series 中挑选"，**常规路径不读盘**、不重采样、不做几何变换。

唯一例外是 :func:`select` 里的**兜底判断**：一个好序列都挑不出来时（上层即将抛
「无任何可用序列」或降级推理），转模态识别重挑一次（:mod:`data.modality_fallback`）。
挑得到时这一步完全不发生——行为与开销与改动前一致。
"""
from __future__ import annotations

import re
from typing import Iterable

from data.structures import Series, Study

#: 模态关键词。顺序敏感：t1c/flair 必须早于 t1/t2（子串包含关系）。
_MODALITY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("t1c", ("t1c", "t1ce", "t1_ce", "t1+c", "t1wi+c", "postcontrast", "post_contrast",
             "post contrast", "post", "enhance", "增强", "ce+", "+c", "gd")),
    ("flair", ("flair", "t2flair", "t2_flair", "t2-f", "dark_fluid", "darkfluid",
               "压水", "tirm")),
    ("dwi", ("dwi", "diffusion", "trace", "epi", "弥散")),
    ("adc", ("adc", "apparent_diffusion")),
    ("swi", ("swi", "susceptibility", "swan", "t2star")),
    ("t2", ("t2wi", "t2w", "t2_wi", "t2")),
    ("t1", ("t1wi", "t1w", "t1_wi", "t1")),
)


def guess_modality(text: str | None) -> str | None:
    """从序列描述/UID 猜模态；先判 t1c/flair/dwi（它们也含 t1/t2 子串）。"""
    low = (text or "").lower()
    for key, kws in _MODALITY_KEYWORDS:
        for kw in kws:
            if kw in low:
                return key
    return None


#: 内部模态键（``_key_of`` 的合法返回值）
_INTERNAL_KEYS = frozenset({"t1c", "flair", "dwi", "adc", "swi", "t2", "t1"})


def _key_of(series: Series) -> str | None:
    """优先用显式标记的模态，其次从描述/UID 推断。

    ⚠️ 显式模态值**必须先归一化**：``metadata['modality']`` 可能是官方表的**原值**
    （如 ``T1CE(增强)`` / ``T2WI`` / ``T2-Flair``）。旧实现直接 ``str(explicit).lower()``
    当键用 → 得到 ``t1ce(增强)``，永远匹配不上内部键 ``t1c``；更糟的是它属于**最高优先级**，
    会把"从描述推断"和后面两条模态兜底**全部短路掉** ——
    实测症状就是：日志里官方表读得到几千条，兜底却一条都改不动、`select()` 恒为空。
    这里先判"是不是内部键"，不是就走关键词归一。
    """
    explicit = (series.metadata or {}).get("modality")
    if explicit:
        low = str(explicit).strip().lower()
        if low in _INTERNAL_KEYS:
            return low
        guessed = guess_modality(low)
        if guessed:
            return guessed
    return guess_modality(series.modality) or guess_modality(series.series_uid)


def _select_picked(study: Study, wanted: tuple[str, ...]) -> dict[str, Series]:
    """纯挑选：只读 ``study.series`` 里已有的描述，不碰磁盘。"""
    out: dict[str, Series] = {}
    for s in study.series:
        key = _key_of(s)
        if key is None or key not in set(wanted):
            continue
        cur = out.get(key)
        if cur is None or s.image.size > cur.image.size:
            out[key] = s
    return out


def select(study: Study, wanted: Iterable[str]) -> dict[str, Series]:
    """按优先级返回 ``{模态: Series}``；同一模态取体素最多的那一个。

    体素最多通常意味着覆盖最完整（少切片/局部序列会被排除）。

    **兜底判断（只在"要报错"时生效）**：一个好序列都挑不出来，说明上层马上要抛
    「无任何可用序列」（规范 §9.1 不可降级）或被迫降级推理——此时自动转模态识别：
    用数据集自带的 ``SeriesType.xlsx`` 给认不出模态的序列重贴描述后再挑一次
    （:func:`data.modality_fallback.recover_study`，含 UID 单键回退）。
    重挑仍为空 → 返回空，让上层按**原逻辑**报错；**挑到了就继续**。

    挑得到序列时整个兜底不触发：不读盘、零额外开销，行为与改动前一致
    （除 ``wanted`` 会被物化成 tuple——顺带修掉"传生成器时只对第一条序列生效"的隐患）。
    """
    wanted = tuple(wanted)
    out = _select_picked(study, wanted)
    if out or not wanted:
        return out
    from data.modality_fallback import recover_study

    out = _select_picked(recover_study(study), wanted)
    if out:
        return out
    # 第三条腿：**表也匹配不上时**用体素统计判模态重挑。
    # 这是唯一不依赖"表里的键与磁盘一致"的兜底 —— 序列描述认不出、序列表对不上时仍能救回通道。
    # 缺了它，Goal5 的输入通道会全零、掩膜退化写入参考序列 → 该例分割必然 0 分。
    from data.voxel_modality import guess_excluded_enabled, recover_study as recover_by_voxels

    out = _select_picked(recover_by_voxels(study), wanted)
    if out:
        return out
    # 最后一遍：连"权威排除（表里明写 `其他`/`正常`/`平扫`）"的序列也允许猜。
    #
    # **支持它的事实**：验证集上存在**整例序列全被标成 `其他`** 的情况 —— 此时尊重
    # "权威排除"就等于必然 0 分（Goal5 输入通道全零、掩膜退化）。
    #
    # **反对它的事实（实测后已改为默认关闭）**：
    #   ① `其他` 序列最可能是定位像 / 非脑之类的**异常序列**，体素判别对这类 OOD 输入会给出
    #      **概率饱和**（置信 1.0 / 次优 0.0）→ `_MIN_CONF = 0.5` 形同虚设；
    #   ② `scripts/voxel_consistency.py` 实测的一致率**低于三分类随机水平**。
    # 把定位像当 T1CE 填进通道，**比留空通道更有害**（留空是训练时见过的"缺失"标记）。
    #
    # 所以改成**开关控制**：`GLIOMA_VOXEL_GUESS_EXCLUDED=0` 关闭（`start.sh` 默认关闭）。
    # 若重训后把一致率测到 >=80%，再设回 `1` 打开。
    if not guess_excluded_enabled():
        return {}
    return _select_picked(recover_by_voxels(study, allow_excluded=True), wanted)


def select_first(study: Study, wanted: Iterable[str]) -> Series | None:
    """按 ``wanted`` 的顺序返回第一个命中的序列（用于"必需模态"）。"""
    got = select(study, wanted)
    for w in wanted:
        if w in got:
            return got[w]
    return None


def _sanitize_uid(uid: str) -> str:
    """清洗 UID 中的路径分隔符与控制字符（**用 translate 而非正则**）。

    这里踩过一个坑：写成 ``re.sub(r"[\\\\/\\x00-\\x1f]", "_", uid)`` 时，
    raw string 里的 ``\\x00`` 是**字面字符**而不是控制字符转义，
    字符类的实际范围与预期完全不同，结果把 UID 里的普通字符也替换掉了
    （实测 ``T1CE`` → ``____``，进而导致输出目录名全错）。
    """
    table = {ord(c): "_" for c in "\\\\/"}
    table.update({i: "_" for i in range(32)})                     # 控制字符
    return uid.translate(table)


def infer_uid_from_path(series) -> str:
    """稳定的序列标识：**磁盘推导值优先**（官方契约的 ``{SeriesUid}`` 就是它）。

    官方 README「当前规范解释」规定：
    ``SegmentationMaskURI`` 相对于病例目录，格式为 ``./{SeriesUid}/{SeriesUid}.nii.gz`` ——
    平台要按这个相对路径解析回**输入数据**的同名目录，所以 ``{SeriesUid}`` 必须是
    ``<AccessionNumber>/<SeriesUid>/<SeriesUid>.nii.gz`` 里的那一层目录名。

    优先级因此固定为：``series_uid``（Loader 已按磁盘推导：3 层取目录名、≤2 层取文件主干）
    → 目录名 → sidecar 的 ``SeriesInstanceUID``。

    ⚠️ **不能反过来（旧行为：sidecar 优先）**：sidecar 与目录名不一致时会写出一个输入数据里
    不存在的目录，而本地 ``OutputValidator`` 只拿答案目录名去内存里的 ``Study`` 查
    （writer/validator 共用同一个 uid，必然自洽），**查不出这个错** —— 本地全绿、平台取不到。
    """
    uid = str(getattr(series, "series_uid", "") or "").strip()
    if uid:
        return _sanitize_uid(uid)
    parent = getattr(getattr(series, "source_path", None), "parent", None)
    if parent is not None and parent.name:
        return _sanitize_uid(parent.name)
    sidecar = str((getattr(series, "metadata", None) or {}).get("SeriesInstanceUID") or "").strip()
    return _sanitize_uid(sidecar) if sidecar else ""
