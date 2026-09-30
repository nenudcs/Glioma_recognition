"""结构化字段的中文→规范英文枚举映射，以及 csv/xlsx 金标准读取。

规范枚举（《赛事开发规范（赛道四）》prediction.json）：
  Location: Brainstem/Right|Left{Parietal,Frontal,BasalGanglia,Temporal,Cerebellum,Occipital}/Other/NA
  Morphology: Regular/Irregular/NA
  WHO_Grade: 1/2/3/4
  EnhancementPattern: None/Ring/RimEnhancing/Nodular/GroundGlass/Gyriform/Multifocal/Other
  Signal_*: Low/Iso/High
数据集中文（《公共数据集格式说明》赛道4）：
  病灶位置 15 类、病灶形态 3 类、边缘分叶/边界 4 类、坏死/囊变/出血/钙化 3 类、
  T2WI/T2-FLAIR 信号 4 类、T1WI+C 强化 3 类、强化形态 8 类
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any


def _fold(text: Any) -> str:
    """列名 / 枚举取值 / 文件名的归一化：**全角→半角 + 去空白 + 大小写无关**。

    同一列表头在不同版本、不同人手里会写成
    ``AccessionNumber`` / ``ACCESSIONNUMBER`` / ``accession_number`` /
    ``ＡｃｃｅｓｓｉｏｎＮｕｍｂｅｒ``（中文输入法打出的全角）/ ``Accession Number``（多个空格）——
    人工看是同一列，**精确比较**却是五个不同字符串。一旦掉进"包含匹配"轮，
    同一张表里的 ``StudyUid`` 就可能被别的关键词先命中：整表按错列建索引、
    行键形如 ``1.2.3.xxxx``，与磁盘检查号一条都对不上（静默丢行，见
    :data:`ID_COLUMN_KEYWORDS` 的说明）。所以列名匹配统一先过这里。

    NFKC 把全角字母/数字/标点折成半角（``Ｔ１`` → ``T1``、``ＮＡ／ＵＮＫ`` → ``NA/UNK``），
    对中文是恒等变换，不影响中文列名。

    与 :func:`_norm_key` 的分工：``_norm_key`` 用于**取值键**（检查号 / 序列号），
    只去空白 + casefold；本函数额外做 NFKC，用于列名、枚举取值、文件名的比较。
    """
    s = unicodedata.normalize("NFKC", str(text if text is not None else ""))
    return re.sub(r"\s+", "", s).casefold()

# ---- 位置：中文 → 英文枚举（规范 15 类）----
LOCATION_MAP = {
    "脑干": "Brainstem",
    "右侧顶叶": "RightParietal", "右侧额叶": "RightFrontal",
    "右侧基底节区": "RightBasalGanglia", "右侧颞叶": "RightTemporal",
    "右侧小脑半球": "RightCerebellum", "右侧枕叶": "RightOccipital",
    "左侧顶叶": "LeftParietal", "左侧额叶": "LeftFrontal",
    "左侧基底节区": "LeftBasalGanglia", "左侧颞叶": "LeftTemporal",
    "左侧小脑半球": "LeftCerebellum", "左侧枕叶": "LeftOccipital",
    "其他": "Other", "NA/UNK": "NA", "NA": "NA", "": "NA",
}

# ---- 强化形态：中文 → 规范 8 类 ----
ENHAN_PATTERN_MAP = {
    "多灶状": "Multifocal", "花环状": "RimEnhancing", "环形": "Ring",
    "结节状": "Nodular", "毛玻璃样": "GroundGlass", "脑回状": "Gyriform",
    "其他": "Other", "无": "None", "NA/UNK": "Other", "": "Other",
}

# ---- 形态 ----
MORPH_MAP = {"规则": "Regular", "不规则": "Irregular", "NA/UNK": "NA", "": "NA"}

# ---- 三分类信号：1低/2等/3高 ----
SIGNAL_MAP = {"1低": "Low", "2等": "Iso", "3高": "High", "低": "Low", "等": "Iso", "高": "High",
              "无": "Low", "NA/UNK": None, "": None}

# ---- 有/无（含 4 类变体）----
YESNO_MAP = {"有": 1, "无": 0, "有/清": 1, "无/不清": 0, "无病灶": 0, "NA/UNK": None, "": None}

# ---- 病理结果 → WHO 分级 / 是否胶质瘤 ----
GRADE_MAP = {"脑胶质瘤1级": "1", "脑胶质瘤2级": "2", "脑胶质瘤3级": "3", "脑胶质瘤4级": "4"}
NON_GLIOMA = {"其他肿瘤或病变", "脑转移", "脑脓肿", "脑梗死", "病因不明", "无", "NA/UNK", ""}
#: 级别单独成列（`WHO分级` / `分级`）时可能出现全角数字；罗马数字见 ``_ROMAN_GRADE``
_CIRCLED_GRADE = {"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4"}
_ROMAN_GRADE = {"i": "1", "ii": "2", "iii": "3", "iv": "4"}

# ---- 掩码文件角色识别（《公共数据集格式说明》赛道4 的 ROI 命名）----
MASK_ROLE_KEYWORDS = {
    "core": ["肿瘤瘤体", "瘤体", "core", "et", "enhanc", "增强"],
    "peri": ["全肿瘤", "水肿", "whole", "edema", "flair", "异常信号", "abnormal", "abn"],
    "abn": ["异常信号", "abnormal", "abn"],
}

# ---- 序列模态识别（文件名/序列描述关键词）----
MODALITY_KEYWORDS = {
    "t1c": ["t1c", "t1ce", "t1_ce", "t1+c", "t1wi+c", "t1_wi+c", "postcontrast", "post_contrast",
            "post contrast", "post", "enhance", "增强", "ce+", "+c", "gd", "t1_mprage_post"],
    "flair": ["flair", "t2flair", "t2_flair", "t2-f", "dark_fluid", "darkfluid", "压水", "tirm"],
    "dwi": ["dwi", "diffusion", "diff", "trace", "epi", "弥散"],
    "adc": ["adc", "apparent_diffusion"],
    "swi": ["swi", "susceptibility", "swan", "t2star", "t2*"],
    "t2": ["t2wi", "t2w", "t2_wi", "t2"],
    "t1": ["t1wi", "t1w", "t1_wi", "t1"],
}


def guess_modality(text: str) -> str | None:
    """从文件名/序列描述猜模态；先判 t1c/flair/dwi（它们也含 t1/t2 子串）。

    关键词比对**大小写 / 全角 / 空格无关**（见 :func:`_fold`）：``T1CE（增强）``、
    ``t1ce``、``Ｔ１ＣＥ（增强）``（全角）是同一件事，`T1 CE` 也按 ``t1ce`` 认。
    漏认的表现是"这一路模态丢了"：掩膜会被归到错误的任务空间，
    而且不报错、只是结果偏。
    """
    low = _fold(text)
    for key in ("t1c", "flair", "dwi", "adc", "swi", "t2", "t1"):
        for kw in MODALITY_KEYWORDS[key]:
            if _fold(kw) in low:
                return key
    return None


def guess_mask_role(text: str) -> str | None:
    """只看勾画名猜掩膜角色（**大小写/全角无关**；更稳的判定见 :func:`mask_role_for`）。"""
    name = _fold(text)
    for role in ("core", "peri", "abn"):
        for kw in MASK_ROLE_KEYWORDS[role]:
            if _fold(kw) in name:
                return role
    return None


def mask_role_for(filename: str, modality: str | None = None) -> str | None:
    """**结合掩码所在序列的模态**判定掩码角色（关键修正）。

    仅看文件名会把 FLAIR/T2 序列目录下的"瘤体"误判为任务A的 core，
    进而把掩码写到错误的空间。规范语义：
    - 任务A(core) = **T1 增强**序列上的"肿瘤瘤体"；
    - 任务B(peri) = **FLAIR/T2** 序列上的"瘤体 ∪ 水肿"，或整体勾画的"全肿瘤"；
    - 非肿瘤性病变的"异常信号"按所在序列归位（FLAIR/T2 → peri）。

    返回 ``core`` / ``peri`` / ``abn`` / ``None``。
    """
    import re

    name = unicodedata.normalize("NFKC", filename or "")           # 全角→半角，中文不受影响
    low = name.casefold()
    has = lambda kws: any(k in name for k in kws)                   # noqa: E731
    tok = lambda kws: any(re.search(rf"(^|[^a-z0-9]){re.escape(k)}([^a-z0-9]|$)", low) # noqa: E731
                          for k in kws)
    is_f = modality in ("flair", "t2")

    if has(["异常信号"]) or tok(["abnormal", "abn"]):
        return "peri" if (modality is None or is_f) else "core"
    if has(["全肿瘤", "水肿"]) or tok(["whole", "edema", "peritumoral"]):
        return "peri"
    if has(["肿瘤瘤体", "增强"]) or tok(["core", "et", "enhancing", "enhancement"]):
        return "core"
    if "瘤体" in name:
        return "peri" if is_f else "core"                          # FLAIR/T2 上的瘤体 → 总异常区
    if (has(["肿瘤", "病灶"]) or tok(["tumor", "lesion", "mask", "roi"])) and modality is not None:
        return "peri" if is_f else "core"
    return None


def to_enum(value: Any, mapping: dict) -> Any:
    """宽松匹配：先精确，再包含匹配（处理 "有/清"、"2高" 之类）；大小写/全角无关。

    两轮都按 :func:`_fold` 归一后比较：映射里写着 ``NA/UNK``，表里写 ``na/unk``、
    ``Na/Unk`` 或全角 ``ＮＡ／ＵＮＫ``，在人工看来是同一件事，
    精确比较却会漏 —— 漏掉的表现是"字段没读到"，
    下游把它当"这一例没有金标准"，分母悄悄变小（见 :func:`structured_from_row`）。
    """
    if value is None:
        return None
    s = _fold(value)
    for k, v in mapping.items():
        if _fold(k) == s:                                          # ① 精确（含映射里的空串键）
            return v
    for k, v in mapping.items():
        if k and _fold(k) in s:                                    # ② 包含
            return v
    return None


#: 金标准表里"检查号/病例号"列的关键词（大小写无关；含中文与常见变体）。
#:
#: **顺序即优先级**（精确 → 包含两轮，都按此顺序取第一个命中）：
#: 越具体的检查号命名越靠前，泛化的"记录号/序号"放最后，
#: 避免一张表里同时存在行列号时把行号当成了检查号。
#:
#: ⚠️ 各表的列名统一是 ``AccessionNumber``，它**必须排在最前**：
#: 只写后面那些变体时它进不了精确轮，而同一行的 ``StudyUid`` 能精确命中 ——
#: 于是整张检查级别 sheet 按 **StudyUid** 建索引（键形如 ``1.2.3.xxxx``），
#: 与磁盘上的检查号目录名一条都对不上，序列级/ROI级子行也全部挂不上（静默丢行）。
#:
#: 后面的 ``accession_number`` / ``accessionumber``（少一个 n）/ ``accessionno`` …
#: 是**改版与历史排版**的兜底（官方表里的列名被改过若干次拼写），
#: 归一化后都是 ``accessionnumber`` 前缀，留着只增不减兼容面、不影响正常命中。
ID_COLUMN_KEYWORDS = ("accessionnumber", "accession_number", "accession_no",
                      "accessionumber", "accession_num", "accessionno", "accession",
                      "patientid", "patient_id", "record_uuid", "studyuid",
                      "study_instance_uid", "study_id", "studyid",
                      "检查号", "检查编号", "病例号", "患者号", "检查id",
                      "检查序号", "记录号")


#: 只按**精确相等**匹配的短列名。
#:
#: 不能并进上面的"包含"匹配：``id`` 会命中 ``SeriesUid``，
#: 于是整表按**序列号**建索引 —— 检查号永远对不上、字段全空，
#: 而且它看起来"解析成功了"（有行数、无字段），比认不出更难查。
ID_COLUMN_EXACT = ("id", "编号", "序号", "流水号")

#: 判定"这行是表头"用的字段线索（命中越多越像表头）
FIELD_HINTS = ("病理", "glioma", "location", "lesion", "morpholog", "tumor",
               "signal", "enhan", "坏死", "囊变", "出血", "钙化", "强化", "水肿")

#: 已告警过的表（避免每次探测/训练都刷屏）
_WARNED_TABLES: set[str] = set()


# --------------------------------------------------------------------------- #
# 数据集金标准表的**三张工作表**（`脑胶质瘤标注结果-训练集.xlsx`）与"级别"
# --------------------------------------------------------------------------- #
#: 实测排版：三个 sheet 依次是 ``检查级别`` / ``序列级别`` / ``ROI级别``。
#:
#: ⚠️ **不要假设表头在第几行**：三个 sheet 的表头行位置各不相同，第 1~3 行
#: 都可能是"索引信息"（标题 / 字段说明 / 空行）。表头行一律由
#: :func:`_detect_header` 在**前 20 行里扫描**确定（要求含该级别的行键列），
#: 写死"第 N 行"会在官方换一版排版时整表解析出 0 行。
#:
#: 为什么必须分级别处理：三个 sheet 的**行键不是同一个** ——
#: 检查级别 = 检查号（一例一行）、序列级别 = 检查号 + 序列号（一例多行）、
#: ROI 级别 = 检查号 + 序列号 + ROI 名（一例更多行）。
#: 全部按检查号合并会让同病例的后续行**互相覆盖**：看起来"读到了 N 行"、
#: 字段却来自最后一行（序列级/ROI 级），病例级字段全丢 —— 而且不报任何错。
_SHEET_LEVEL_RULES: tuple[tuple[str, str], ...] = (
    ("检查", "case"), ("case", "case"), ("study", "case"), ("exam", "case"),
    ("序列", "series"), ("series", "series"), ("serie", "series"),
    ("roi", "roi"), ("病灶", "roi"), ("掩膜", "roi"), ("mask", "roi"),
)

#: 各级别**行键**的列名候选（合并时用它区分同一病例下的多行）
LEVEL_KEY_COLUMNS: dict[str, tuple[str, ...]] = {
    "case": ID_COLUMN_KEYWORDS + ID_COLUMN_EXACT,
    "series": ("序列号", "序列编号", "序列id", "序列uid", "seriesuid", "series_uid",
               "seriesinstanceuid", "seriesid", "series"),
    # ROI 名**显式列出无下划线拼写并排最前**：官方表头是 ``RoiName``，只写 ``roi名称`` /
    # ``roi`` 时它仅被"前缀"轮的 ``roi`` 兜住 —— 而同一行还有 AB 列 ``ROIUid``（官方表的
    # 实测拼写），位置更靠前、又同样满足 ``roi`` 前缀，于是**先撞上它**：行键变成 ROI UID，
    # ROI 名（瘤体/水肿/肿瘤瘤体/全肿瘤，掩膜角色 core/peri 的唯一来源）退化成普通列。
    # 这不是理论隐患：``ROIUid`` 这种拼写靠 ``roi`` 前缀兜不住，必然被截走。
    "roi": ("roiname", "roi_name", "roi名称", "roi名", "roi编号", "roi号", "roi",
            "掩膜名", "掩膜文件", "掩膜", "maskname", "mask_name", "mask",
            "病灶名", "病灶"),
}

#: 序列级 / ROI 级的行**原样挂**在病例记录的这两个键下（不参与字段映射）。
#: 用双下划线包起来是为了与真实列名不可能撞名（列名都是中文或英文单词）。
LEVEL_NESTED_KEY: dict[str, str] = {"series": "__series_rows__", "roi": "__roi_rows__"}

#: 合并顺序：检查级别在前（病例级字段以它为准），序列/ROI 级只做嵌套保留
_LEVEL_ORDER: dict[str, int] = {"case": 0, "series": 1, "roi": 2, "unknown": 3}

#: 在**序列级 / ROI 级**表里认"检查号列"用的名字（比 :data:`ID_COLUMN_KEYWORDS` 更严）。
#:
#: 这里不能用那套宽松关键词：序列级表的行键是 ``序列号``、ROI 级是 ``ROI名称``，
#: 宽松匹配里的 ``id`` / ``编号``（包含匹配）会把 ``SeriesUid`` / ``序列编号`` 认成检查号，
#: 于是整张表的子行被挂到"序列号当检查号"的假病例上 —— 有结果、全错位。
_CASE_COLUMN_STRICT = ("accessionnumber", "accession_number", "accession_no",
                       "accessionumber", "accession_num", "accessionno", "accession",
                       "检查号", "检查编号", "病例号", "患者号", "检查序号", "检查id",
                       "studyid", "study_id", "study_instance_uid")


def _sheet_level(name: str) -> str:
    """由工作表名判"级别"：``检查级别``→case、``序列级别``→series、``ROI级别``→roi。

    名字认不出来时返回 ``"unknown"``（单张 csv、``Sheet1``、空名…）——
    **不是**"按检查号合并"：:func:`_sheet_plan` 会按表头列回退判级
    （ROI 名 → 序列号 → 检查号）。这里只负责"表名这一条线索"。
    """
    low = _fold(name)                                              # 大小写/全角无关
    for kw, level in _SHEET_LEVEL_RULES:
        if _fold(kw) in low:
            return level
    return "unknown"


def _find_id_column(header) -> str | None:
    """在**表头单元格序列**里定位"检查号"列，返回命中的列名（原样）。

    入参是表头**各单元格的值**（不是整行、也不是 dict）。这里踩过一次坑：
    传 ``dict(enumerate(row))`` 时键变成了下标，于是"列名"永远匹配不上，
    所有表都解析出 0 行 —— 连本来正常的小写 csv 也一起失效。

    两轮匹配：① 精确（含 ``id``/``编号`` 这类短名）→ ② 包含
    （**只用长关键词**，避免 ``id`` 命中 ``SeriesUid`` 而错把序列号当检查号）。

    列名比较**大小写 / 全角 / 空格无关**（见 :func:`_fold`）：
    ``AccessionNumber``、``ACCESSIONNUMBER``、``ＡｃｃｅｓｓｉｏｎＮｕｍｂｅｒ``
    是同一列，任一写法都要能落到"精确"轮 ——
    漏进"包含"轮就可能被同表的 ``StudyUid`` 抢走行键。
    """
    cells = [str(c).strip() for c in header if str(c).strip()]
    table = {_fold(c): c for c in cells}
    for kw in ID_COLUMN_KEYWORDS + ID_COLUMN_EXACT:                # ① 精确
        key = _fold(kw)
        if key in table:
            return table[key]
    for kw in ID_COLUMN_KEYWORDS:                                  # ② 包含（不用短名）
        key = _fold(kw)
        for key_lower, name in table.items():
            if key in key_lower:
                return name
    return None


# --------------------------------------------------------------------------- #
# 天坛参考实现 `AIRecongition/` 的 5 张表（⚠️ **不是赛道四数据集的内容**）
# --------------------------------------------------------------------------- #
#: 参考实现标注表文件名 → 用途
#:
#: ⚠️ **这 5 张表与赛道四数据集没有一点关系**（属工作区里另一个目标的产物）：
#: 赛道四的数据信息就是数据集自带的 ``SeriesType.xlsx``（见
#: :data:`SERIES_TYPE_TABLE`）与训练集的 ``脑胶质瘤标注结果-训练集.xlsx``（字段金标准）。
#: 下面这套仍保留读取，**只为兼容**：表在附近时能用上就用，读不到属正常、不是配置问题。
#:
#: | 文件 | 关键列 | 用途 |
#: |---|---|---|
#: | ``1_abnormal.xlsx`` | AccessionNumber, SeriesUid, **Label** ∈ {true,fake,compositing,duplicate} | 目标一/二的正样本（**逐序列**） |
#: | ``2_duplicate.xlsx`` | src_img, desc_img | 重复影像 pair |
#: | ~~``工作区兼容表``~~ | ~~AccessionNumber, SeriesUid, SeriesLabel~~ | **已删除，不再读**（模态只认数据集里的 ``SeriesType.xlsx``；读它只会把 ``T2WI``/``T2-Flair`` 静默压平成 ``T2``） |
#: | ``4_masklabel.xlsx`` | AccessionNumber, SeriesUid, **Maskname** | 掩膜文件名（任意名，关键词认不出） |
#: | ``5_characteristics.xlsx`` | AccessionNumber + 14 个英文列 | 结构化字段金标准（**兼容**；数据集里那份是中文表头的 ``脑胶质瘤标注结果-训练集.xlsx``） |
#:
#: 这套约定是**唯一权威**。在此之前我们按中文列名去猜，于是出现
#: "病例数正常、一例都挑不出模态""字段金标准为空"——表一直都在，
#: 只是文件名和列名都不是我们猜的那套。
#:
#: 因此本字典里**没有** ``series`` 这个键（:data:`OFFICIAL_LABEL_FILES`）——
#: 找表、打印自检时都不会再出现 ``工作区兼容表``。
OFFICIAL_LABEL_FILES = {
    "abnormal": "1_abnormal.xlsx",
    "duplicate": "2_duplicate.xlsx",
    "mask": "4_masklabel.xlsx",
    "characteristics": "5_characteristics.xlsx",
}

#: 序列类型表在**赛道四数据集里**的文件名（**唯一来源**）。
#:
#: 它的位置与影像同层：``<阶段>/annotation/SeriesType.xlsx``，
#: 训练/验证集的数据里都有（``training`` / ``verification``），
#: ``evaluation_*`` 评测集在正式测试时**随测试数据一起下发**。
#: 列：``AccessionNumber`` + ``SeriesUid`` + ``SeriesType``；
#: 取值共 **5 类**：``T1`` / ``T1CE（增强）`` / ``T2-Flair`` / ``T2WI`` / ``其他``。
SERIES_TYPE_TABLE = "SeriesType.xlsx"

#: 查找/自检文案里出现的表名（**只有一个名字**：数据集自带的那张）。
#: 以前这里还有 ``工作区兼容表`` 做兜底 —— 已删除：它与本赛道数据集无关，
#: 而且取值更粗（只写 ``T2``），会把数据集里的 ``T2WI`` / ``T2-Flair`` 静默压平。
SERIES_TYPE_FILENAMES = (SERIES_TYPE_TABLE,)

#: 官方标注表所在目录的环境变量（对应官方 config 的 ``paths.labels_dir``）
LABELS_DIR_ENV = "GLIOMA_LABELS_DIR"


#: **表的搜索范围 = 数据根（+其父/祖父 + 像标注容器的子目录）。**
#:
#: 这里曾经还会去 ``<工程>/labels`` 与 ``$WORKSPACE`` 下 3 层翻 ``labels/`` 目录
#: （"零配置读到团队工作区那几张表"）。**已移除**，原因有两条：
#:
#: 1. **串表**：工作区里若残留**另一份数据**的表，它会先于当前数据自己的表被命中 ——
#:    表现是"表读到了几千条、却一条都查不中"（拿训练集的键查验证集），日志里表路径
#:    指向 ``labels/`` 而不是数据目录，一眼看不出串了；
#: 2. **本赛道数据集自带全部所需**：模态在 ``SeriesType.xlsx``、掩膜靠 ``_mask`` 文件、
#:    字段金标准是 ``脑胶质瘤标注结果-*.xlsx``、特殊/重复影像靠 ``{fake,compositing,
#:    duplicate}/`` 目录 —— 那 5 张工作区表**不是本赛道数据集的内容**，翻它们只会带偏。
#:
#: 仍然保留**显式**入口（见 :data:`LABELS_DIR_ENV`）：只有你亲手指定时才用它，
#: 不做任何隐式搜索。


#: 候选目录里**值得下钻一层**的子目录名（大小写无关）：标注/结果类容器。
#:
#: 为什么不无脑扫全部子目录：平台数据根下有 **3000+ 病例目录**（32 位哈希），
#: 逐个 stat 既慢，又会在备份/缓存目录里撞到同名旧表。而"标注放在哪个容器目录"
#: 其实是个有限集合 —— 平台那份在 ``training/annotation/``（英文）
#: 或下载后的 ``标注结果/``（中文），下面这些名字把两种情况都覆盖了。
#: "像标注容器"的子目录名。
#:
#: ``original`` 是**验证集**的中间层：``verification/original/`` 里既放影像也放
#: ``SeriesType.xlsx`` / ``脑胶质瘤标注结果-验证集.xlsx``。漏了它，数据根填
#: ``…/verification``（而不是 ``…/verification/original``）时表就一条都搜不到，
#: 报错只说"未找到"，看不出是差了一层目录。
_LABEL_SUBDIR_NAMES = frozenset({
    "labels", "label", "annotation", "annotations", "标注结果", "标注", "结果",
    "original", "gold", "groundtruth", "ground_truth", "gt", "meta", "metadata",
    "results",
})

#: 往下钻几层。2 层是为了覆盖"数据根填高一层（``training/``）**且**
#: 表还在容器子目录里（``annotation/标注结果/``）"这种叠加情况。
_LABEL_SUBDIR_DEPTH = 2


def _subdir_candidates(base: Path) -> list[Path]:
    """``base`` 下"像标注容器"的一级子目录（读不了就返回空，不抛）。

    先按名字过滤再 ``is_dir()``：数据根下可能有 3000+ 病例目录，
    对每个条目都 stat 一次会白花几十毫秒（这里只在名字命中时才 stat）。
    """
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return []
    return [p for p in entries
            if p.name.lower() in _LABEL_SUBDIR_NAMES and p.is_dir()]


def label_search_dirs(root: str | os.PathLike | None = None,
                      labels_dir: str | os.PathLike | None = None) -> list[Path]:
    """标注表候选目录（**顺序即优先级**，命中即止）。

    1. 显式传入的 ``labels_dir``（对应官方 config 的 ``paths.labels_dir``）
    2. 环境变量 ``GLIOMA_LABELS_DIR`` —— **显式**指定，只有你亲手设了才用它
    3. 数据根自身、父目录、祖父目录 —— 官方把 ``SeriesType.xlsx`` / 字段金标准放在
       **病例目录那一层**（``training/annotation/``），第 3 条就是为它准备的；
       传进来的若是**病例目录**，父/祖父两级正好覆盖到 ``annotation/``
    4. 上述每个目录下"像标注容器"的一级子目录（见 :data:`_LABEL_SUBDIR_NAMES`）——
       防止表藏在 ``annotation/`` 的下一层（下载解压后常见的 ``标注结果/``）

    ⚠️ **不再搜 ``<工程>/labels`` 与 ``$WORKSPACE``**：隐式翻别处的表会**串表**
    （工作区里残留另一份数据的表 → 先于当前数据被命中，"表读到几千条却一条都查不中"），
    而本赛道数据集自带全部所需（模态 `SeriesType.xlsx`、掩膜 `_mask` 文件、
    字段 `脑胶质瘤标注结果-*.xlsx`、特殊/重复影像的 `fake/compositing/duplicate` 目录）。
    需要指向别处的表时，用 ``GLIOMA_LABELS_DIR`` **显式**说明。
    """
    cands: list[Path] = []
    if labels_dir:
        cands.append(Path(labels_dir).expanduser())
    if os.environ.get(LABELS_DIR_ENV):
        cands.append(Path(os.environ[LABELS_DIR_ENV]).expanduser())
    if root is not None:
        base = Path(str(root)).expanduser()
        try:
            base = base.resolve()
        except OSError:
            base = base.absolute()
        cands.extend([base, base.parent, base.parent.parent])

    # 去重（保留优先级顺序）后，再为每个目录补上"值得下钻的子目录"（逐层展开，见
    # :data:`_LABEL_SUBDIR_DEPTH`）。用广度优先是为了保持"先浅后深"的优先级：
    # 数据根那一层永远排在自己的子目录前面。
    out: list[Path] = []
    seen: set[str] = set()
    for folder in cands:
        frontier = [folder]
        for _ in range(_LABEL_SUBDIR_DEPTH + 1):
            nxt: list[Path] = []
            for cand in frontier:
                key = str(cand)
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
                nxt.extend(_subdir_candidates(cand))
            frontier = nxt
    return out


def _find_file(folder: Path, filename: str) -> str | None:
    """在**单个目录内**按文件名找文件，**大小写 / 全角 / 空格无关** → 路径。

    为什么不直接 ``(folder / filename).is_file()``：那是**精确字符串**比较。
    Linux 文件系统区分大小写，平台解压、人工改名或从 Windows 拷过来之后
    出现 ``seriestype.xlsx`` / ``SERIESTYPE.XLSX`` / ``ＳｅｒｉｅｓＴｙｐｅ.xlsx`` 时，
    精确比较会**直接判为"表不存在"** —— 而这类失败是静默的：
    一路降级成"没有模态表/没有字段金标准"，不报错、只是数字变小。

    先试精确（绝大多数情况一次命中、不扫目录），再退回大小写无关扫描。
    """
    direct = folder / filename
    if direct.is_file():
        return str(direct)
    want = _fold(filename)
    try:
        entries = list(folder.iterdir())
    except OSError:
        return None
    for entry in entries:
        if entry.is_file() and _fold(entry.name) == want:
            return str(entry)
    return None


def find_named_table(filename: str, root: str | os.PathLike | None = None,
                     labels_dir: str | os.PathLike | None = None) -> str | None:
    """在候选目录里按**文件名**找一张表 → 路径（找不到返回 ``None``）。

    为什么不复用 :func:`find_official_labels`：它只认工作区那几张固定名字的表
    （``1_abnormal.xlsx`` 等），而赛道四数据集里那张叫 ``SeriesType.xlsx`` ——
    名字不同，必须在**同一批候选目录**里分别找，才能既认数据集又兼容工作区。

    文件名比较走 :func:`_find_file`：``SeriesType.xlsx`` / ``seriestype.xlsx`` /
    ``SERIESTYPE.XLSX`` 都算命中。
    """
    for folder in label_search_dirs(root, labels_dir):
        found = _find_file(folder, filename)
        if found:
            return found
    return None


def find_official_labels(root: str | os.PathLike | None = None,
                         labels_dir: str | os.PathLike | None = None) -> dict[str, str]:
    """定位团队工作区那几张标注表 → ``{用途: 路径}``（找不到的键不出现）。

    搜索顺序见 :func:`label_search_dirs`。这些表放在**工程/工作区目录**而不是数据集里 ——
    这也是"数据根下找不到金标准"的原因之一。``root=None`` 时只搜 1~4
    （用于报错时做"表到底在不在"的自检）。

    ⚠️ 里面**没有模态表**：``工作区兼容表`` 已从 :data:`OFFICIAL_LABEL_FILES`
    移除，模态只在数据集的 ``SeriesType.xlsx`` 里认（见 :func:`read_series_types`）。
    """
    cands = label_search_dirs(root, labels_dir)
    found: dict[str, str] = {}
    for kind, name in OFFICIAL_LABEL_FILES.items():
        for folder in cands:
            hit = _find_file(folder, name)                         # 文件名大小写/全角无关
            if hit:
                found[kind] = hit
                break
    return found


def has_series_type_table(folder: str | os.PathLike | None) -> bool:
    """该目录下是否**直接**放着序列类型表（大小写/全角无关）。

    专供 :func:`src.data.probe.resolve_case_root` 判"这一层是不是病例层"用：
    ``SeriesType.xlsx`` 随数据下发、与检查号目录**同层**，所以
    "该层直接有这张表"就是确定性信号 —— 不依赖容器目录名
    （训练集叫 ``annotation``、验证集叫 ``original``，评测集的名字还未知）。
    """
    if folder is None:
        return False
    return _find_file(Path(str(folder)), SERIES_TYPE_TABLE) is not None


#: 数据目录里序列类型表可能待的位置：**与检查号目录同层**（平台契约），
#: 容器名训练集是 ``annotation``、验证集是 ``original``。
_DATA_TABLE_SUBDIRS = ("", "annotation", "original")


def find_series_type_table_in_data(root: str | os.PathLike | None) -> str | None:
    """**只在数据目录里**找序列类型表 → 路径或 ``None``。

    为什么需要它（防**跨数据集串表**）：:func:`find_named_table` 的候选顺序是
    ``显式 labels_dir/$GLIOMA_LABELS_DIR → 数据根/父/祖父``。
    早先这一轮还会去 ``<工程>/labels`` 与 ``$WORKSPACE/**/labels``，若那里残留了一份
    **另一个数据集**的 ``SeriesType.xlsx``（比如把训练集的表拷过去过），它会**先于**
    当前数据自己的表被命中 ——
    表现正是"表读到了几千条、却一条都查不到"：拿训练集的检查号/序列号去查
    验证集的数据（2026-09-24 排查过的故障形态），而且日志里表的路径指向
    ``labels/`` 而不是数据目录，一眼看不出串了。

    所以模态表**数据优先**：先查 ``root`` 本身、``root/annotation``、
    ``root/original``、``root 的父目录``（都不在时才退回通用搜索）。
    其余四张表（``1_abnormal`` 等）仍走通用搜索 —— 它们本来就在工作区。
    """
    if root is None:
        return None
    base = Path(str(root)).expanduser()
    try:
        base = base.resolve()
    except OSError:
        base = base.absolute()
    for sub in _DATA_TABLE_SUBDIRS:
        folder = base / sub if sub else base
        hit = _find_file(folder, SERIES_TYPE_TABLE)
        if hit:
            return hit
    return _find_file(base.parent, SERIES_TYPE_TABLE)


#: 分层列名的分隔符：官方表的列名就是**字段路径**（``Study->CLINICAL->病理结果``）。
#:
#: ⚠️ 只认这两种箭头：普通连字符列名（``T2-Flair``、``t1wi_c_enhan``）不能拆。
_COLUMN_PATH_SEPS: tuple[str, ...] = ("->", "→")


def _column_leaf(name: Any) -> str:
    """取**分层列名的末段**：``Study->CLINICAL->病理结果`` → ``病理结果``。

    实测排版：检查级别 sheet 的病例级字段全写成路径形式
    （``Study->CLINICAL->病理结果``、``Study->DICOM->StudyDate``），**末段才是字段名**。

    只按整串匹配时字段能不能命中全靠"包含"，父段里出现关键词就会被抢走
    （``...->病理结果`` 与 ``...->病理类型`` 谁在前面谁得）。因此匹配顺序统一成
    **先末段、再整串** —— 只增精度，不改旧行为（非分层列名的末段就是它自己）。
    """
    text = str(name if name is not None else "").strip()
    for sep in _COLUMN_PATH_SEPS:
        if sep in text:
            text = text.rsplit(sep, 1)[-1].strip()
    return text


def _find_col_in_list(header: list[str], keywords: tuple[str, ...]) -> int | None:
    """在表头列表里找列，返回**列号**：先按**分层列名末段**、再按整串；每轮 精确→前缀→包含。

    末段优先的理由见 :func:`_column_leaf`；
    列名与关键词都先过 :func:`_fold`（**大小写 / 全角 / 空格无关**）——
    ``SeriesUid`` / ``SERIESUID`` / ``ＳｅｒｉｅｓＵｉｄ`` / ``Series Uid``
    必须落在同一轮次里，否则同一张表换个写法就"整表 0 行"。
    """
    full = {_fold(c): i for i, c in enumerate(header) if str(c).strip()}
    leaf: dict[str, int] = {}
    for i, cell in enumerate(header):
        text = str(cell).strip()
        if text:
            leaf.setdefault(_fold(_column_leaf(text)), i)
    for table in (leaf, full):                                     # 末段 → 整串
        for kw in keywords:                                        # ① 精确
            key = _fold(kw)
            if key in table:
                return table[key]
        for kw in keywords:                                        # ② 前缀
            key = _fold(kw)
            for low, idx in table.items():
                if low.startswith(key):
                    return idx
        for kw in keywords:                                        # ③ 包含
            key = _fold(kw)
            for low, idx in table.items():
                if key in low:
                    return idx
    return None


def read_official_triples(path: str,
                          id_kws: tuple[str, ...] = ("accessionnumber", "accession", "检查号"),
                          uid_kws: tuple[str, ...] = ("seriesuid", "series_uid", "序列号"),
                          value_kws: tuple[str, ...] = ("label",),
                          ) -> dict[tuple[str, str], list[str]]:
    """读"检查号 + 序列号 + 取值"三类表 → ``{(检查号, 序列号): [取值, ...]}``。

    ``1_abnormal`` / ``4_masklabel`` 都是这个形状
    （掩膜表同一序列可能有多行 → 取值列表）。取值保持原样大小写，
    调用方按需 ``.upper()`` / ``.lower()`` 比较。
    """
    out: dict[tuple[str, str], list[str]] = {}
    for rows in _sheet_rows(path):
        head = _detect_header(rows)
        if head is None:
            continue
        header = [str(c).strip() for c in rows[head]]
        i_id = _find_col_in_list(header, id_kws)
        i_uid = _find_col_in_list(header, uid_kws)
        i_val = _find_col_in_list(header, value_kws)
        if i_id is None or i_uid is None or i_val is None:
            continue
        for row in rows[head + 1:]:
            def _cell(idx: int) -> str:
                return str(row[idx]).strip() if idx < len(row) else ""
            acc, uid, value = _cell(i_id), _cell(i_uid), _cell(i_val)
            if not acc or not uid or not value:
                continue
            for key in {(acc, uid), (acc.casefold(), uid.casefold())}:
                bucket = out.setdefault(key, [])
                if value not in bucket:
                    bucket.append(value)
    return out


def read_mask_table(path: str) -> dict[tuple[str, str], list[str]]:
    """读 ``4_masklabel.xlsx`` → ``{(检查号, 序列号): [Maskname, ...]}``。"""
    return read_official_triples(
        path,
        value_kws=("maskname", "mask_name", "mask", "掩膜", "标注文件"),
    )


def read_duplicate_pairs(path: str) -> list[tuple[str, str]]:
    """读 ``2_duplicate.xlsx`` → ``[(src_img, desc_img), ...]``（重复影像正对）。

    ⚠️ **这是兜底表，不是赛道四数据集的金标准**：格式说明写明金标准就在
    ``annotation/duplicate/`` 目录下（每行 ``src_img, desc_img``，两个值是检查号），
    由 :func:`src.data.probe.scan_special` 直接读取。``2_duplicate.xlsx`` 属于天坛
    参考实现，检查号可能来自**别的数据集** —— 只在数据集里没有金标准文件时才用它
    （见 :func:`src.data.probe.probe`）。
    """
    pairs: list[tuple[str, str]] = []
    src_kws = ("src_img", "src", "image1", "检查号1", "studyuid")
    dst_kws = ("desc_img", "desc", "image2", "检查号2", "studyuid_dup")
    for rows in _sheet_rows(path):
        # ⚠️ 不能复用通用表头识别：它要求"检查号列"，而这张表只有 src/desc 两列，
        # 于是永远返回 None → 重复金标准恒为 0 对（不报错）。
        head = None
        for idx, row in enumerate(rows[:20]):
            header = [str(c).strip() for c in row]
            if (_find_col_in_list(header, src_kws) is not None
                    and _find_col_in_list(header, dst_kws) is not None):
                head = idx
                break
        if head is None:
            continue
        header = [str(c).strip() for c in rows[head]]
        i_src = _find_col_in_list(header, src_kws)
        i_dst = _find_col_in_list(header, dst_kws)
        if i_src is None or i_dst is None or i_src == i_dst:
            continue
        for row in rows[head + 1:]:
            src = str(row[i_src]).strip() if i_src < len(row) else ""
            dst = str(row[i_dst]).strip() if i_dst < len(row) else ""
            if src and dst:
                pairs.append((src, dst))
    return pairs


def read_abnormal_table(path: str) -> dict[tuple[str, str], str]:
    """读 ``1_abnormal.xlsx`` → ``{(检查号, 序列号): Label}``。

    ``Label ∈ {true, fake, compositing, duplicate}``：它同时告诉我们两件事 ——
    这条序列是不是异常影像，以及它的影像在哪个子目录下
    （``true`` → 数据根；``fake``/``compositing``/``duplicate`` → 同名子目录）。
    """
    triples = read_official_triples(path, value_kws=("label", "标签"))
    return {k: v[0].lower() for k, v in triples.items() if v}


#: 读文本类表（csv/txt）时依次尝试的编码。
#:
#: 为什么不能只按 ``utf-8-sig`` 打开：中文 Windows / WPS / Excel 的
#: 「CSV（逗号分隔）」默认写 **ANSI（GBK/GB18030）**、「Unicode 文本」写
#: **UTF-16LE（带 BOM）**。这两种文件按 utf-8 解必然抛 ``UnicodeDecodeError``
#: （实测报错形如 ``'utf-8' codec can't decode byte 0xcf in position 3``），
#: 而表本身是完好的 —— 表现是"表就在磁盘上、读金标准却全部失败"，
#: 训练侧等于**没有标签**（分类头学不到东西，且只在告警里出现一次）。
#:
#: 顺序有讲究：``utf-8-sig`` 必须排在 ``gb18030`` 之前。gb18030 能"成功"解码
#: 绝大多数 UTF-8 字节序列，只是解成乱码 —— 先试它会把正常 UTF-8 表读成乱码，
#: 行数对、值全错，比直接报错更难查。
TEXT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "big5")

#: 带 BOM 的编码靠嗅探识别（它们**不能**放进盲试列表：``utf-16`` 无 BOM 时会猜错字节序）。
_BOM_ENCODINGS: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xfe", "utf-16"),                                       # Excel「Unicode 文本」
    (b"\xfe\xff", "utf-16"),
    (b"\xef\xbb\xbf", "utf-8-sig"),
)


def read_text_any_encoding(path: str) -> str:
    """读文本表 → ``str``：**BOM 嗅探 + 多编码回退**（见 :data:`TEXT_ENCODINGS`）。

    全部失败时抛出**列明试过哪些编码**的异常。绝不 ``errors="replace"`` 硬解：
    那样中文会变成 ``锟斤拷``，行还在、值全错，比读不到难查得多
    （与 :func:`_detect_header` 拒绝"猜表头"同一原则）。
    """
    with open(path, "rb") as f:
        raw = f.read()
    for bom, enc in _BOM_ENCODINGS:
        if raw.startswith(bom):
            return raw.decode(enc)
    errors: list[str] = []
    for enc in TEXT_ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as exc:                           # noqa: PERF203
            errors.append(f"{enc}: {exc}")
    raise ValueError(
        f"读表失败 {os.path.basename(path)}：试过 {'/'.join(TEXT_ENCODINGS)} 都解不开"
        f"（把这个表另存为 UTF-8 或 .xlsx 即可正常读取）。首个错误：{errors[0]}")


def _sheet_frames(path: str) -> list[tuple[str, list[list[str]]]]:
    """把 csv/xlsx 读成 ``[(工作表名, 原始行), ...]``（**不做任何表头假设**）。

    为什么不直接用 ``pandas.read_excel`` 的默认行为：它把**第一行**当表头、
    且**只读第一个 sheet**。本项目两类表的真实排版都跟默认行为对不上：

    * ``脑胶质瘤标注结果-训练集.xlsx``：**3 个 sheet（检查级别/序列级别/ROI级别）**，
      每个 sheet 的表头行位置还不一样（第 1~3 行都可能是索引信息）；
    * 中文标注表的常见排版：``标题 → 空行/说明 → 真正的表头``。

    默认行为下列名会变成"标题/Unnamed"，检查号列认不出来，整表 0 行。
    工作表名要留着：它是"这张表是哪个级别"的**第一条线索**（见 :func:`_sheet_level`）；
    名字认不出来时级别由表头列决定（见 :func:`_sheet_plan`）。
    csv 的编码见 :func:`read_text_any_encoding`（中文 Excel/WPS 导出的
    ANSI(GBK) 与「Unicode 文本」(UTF-16) 都能读）。
    """
    if path.endswith((".xlsx", ".xls")):
        import pandas as pd
        try:
            sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=str)
        except ImportError as exc:
            # 读 `.xls` 要 xlrd、读 `.xlsx` 要 openpyxl。缺依赖时 pandas 的原文
            # 只有一句 "Missing optional dependency"，分不清是"表坏了"还是
            # "这张表格式本工程读不了"；这里把**怎么办**写进异常。
            raise ImportError(
                f"读表失败 {os.path.basename(path)}：缺少读取该格式的依赖（{exc}）。"
                f"安装 requirements.txt 里的 xlrd / openpyxl，"
                f"或把这张表另存为 .xlsx 再重试") from exc
        return [(str(name), [["" if v is None else str(v).strip() for v in row]
                             for row in frame.fillna("").values.tolist()])
               for name, frame in sheets.items()]
    import csv
    # 不写死 utf-8：中文 Excel 导出的 csv 多为 ANSI(GBK)，按 utf-8 解
    # 直接 UnicodeDecodeError → 金标准整表读不到（训练侧表现为"没有标签"）。
    text = read_text_any_encoding(path)
    return [(os.path.splitext(os.path.basename(path))[0],
             [[str(c).strip() for c in row]
              for row in csv.reader(text.splitlines(True))])]


def _sheet_rows(path: str) -> list[list[list[str]]]:
    """只要行、不要工作表名（多数调用方用不到名字）。"""
    return [rows for _, rows in _sheet_frames(path)]


def _detect_header(rows: list[list[str]], max_scan: int = 20,
                   level: str = "unknown") -> int | None:
    """在前若干行里找**真正的表头行**：必须含该级别的行键列，字段线索越多越优先。

    只看第一行是这个数据最常见的失效点（标题行占了第一行）；
    而"必须含行键列"这条同时挡住了把说明行、数据行误判成表头。

    ``level`` 决定"行键列"是什么：检查级别看检查号，**序列级别看序列号，
    ROI 级别看 ROI 名**。少了这个参数，序列级别 / ROI 级别的 sheet 会因为
    "没有检查号列"而判成找不到表头 → 整张表 0 行（序列级字段全丢、还不报错）。
    """
    key_cols = LEVEL_KEY_COLUMNS.get(level)
    best: tuple[int, int, int] | None = None            # (字段线索, 非空列数, 行号)
    for idx, row in enumerate(rows[:max_scan]):
        cells = [str(c).strip() for c in row]
        if not any(cells):
            continue                                               # 空行
        if key_cols is not None:
            if _find_col_in_list(cells, key_cols) is None:
                continue
        elif _find_id_column(row) is None:
            continue
        # 线索比对同样大小写/全角无关（`FIELD_HINTS` 本身已是小写+中文，只需折叠单元格）；
        # 否则一张全大写表头的表会"线索 0 命中"，评分退化成只比非空列数、表头行选错。
        folded = [_fold(c) for c in cells]
        hits = sum(1 for f in folded
                   if any(k in f for k in FIELD_HINTS))
        # 非空列数当**第二判据**：标题行常是"一格有字、其余全空"，而真表头是满行。
        # 少了它，标题行 'ROI级别' 会被前缀匹配当成 roi 列、压过真表头 → 表头行混进数据、
        # 还凭空多出一个用表头文字当检查号的假病例。
        score = (hits, sum(1 for c in cells if c))
        if best is None or score > best[:2]:
            best = (score[0], score[1], idx)
    return best[2] if best else None


def _id_key(value) -> str:
    """把"检查号"归一化成可比较的键：只留字母数字，去前导零，大小写无关。

    这样 ``C0E1F8F2-53BA_45BE``、``c0e1f8f253ba45be``、`` 00123 `` 都能对上；
    全角（中文输入法）也一并折成半角：``１２３４５６７`` → ``1234567``，
    否则全角数字会被下面那条"只留 ASCII 字母数字"的正则整段抹掉、键变空串而被丢掉。

    另修一个 Excel 专属坑：数字型检查号读出来常带小数尾巴（``1234567.0``），
    若直接去掉非字母数字会拼成 ``12345670`` —— 与目录名 ``1234567``
    差一位、永远不相等。所以先把结尾的 ``.0`` 摘掉再归一化。
    本函数**幂等**：对已归一化的键再调用结果不变。
    """
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if "." in text:
        text = re.sub(r"\.0+$", "", text)                           # 1234567.0 → 1234567
    text = re.sub(r"[^0-9a-z]+", "", text.casefold())
    return text.lstrip("0") or text


#: 公开别名：检查号归一化是"探针/数据集/脚本"三方共用的口径，
#: 各写一份必然出现"一处归一化、一处原样"的静默错配（本文件已因此踩过一次）。
id_key = _id_key


def _best_id_column_by_values(rows: list[list[str]], known_keys: set[str],
                              min_hit: float = 0.3, max_scan: int = 300
                              ) -> tuple[int, int] | None:
    """**按取值**找出检查号列：返回 ``(列号, 第一处命中的行号)``。

    这是"不猜列名"的做法：磁盘上已经有哪些检查号是确定的，
    拿它去逐列比对即可 —— 列名写 ``编号``/``AccessionNumber``/``患者ID``
    甚至乱码都不影响。比维护关键词表可靠得多（关键词表每遇到一种新命名就失效一次）。

    ⚠️ ``known_keys`` 允许是**原始目录名**（调用方常直接把 ``os.listdir`` 结果传进来）：
    这里统一做一次 :func:`_id_key` 归一化再比。此前只有表内取值归一化、
    调用方若是原样传入，遇到 ``C0E1F8F2-53BA-45BE`` 这种带大写/连字符的哈希
    检查号就会**一条都对不上**，于是静默退化成"按列名找"→ 整表 0 行。
    ``_id_key`` 幂等，已归一化的调用方重复传也安全。
    """
    if not rows or not known_keys:
        return None
    known_keys = {_id_key(k) for k in known_keys}
    width = max(len(r) for r in rows[:max_scan]) if rows[:max_scan] else 0
    best: tuple[float, int, int] | None = None            # (命中率, 命中数, 列号)
    for col in range(width):
        hit = total = first_hit = 0
        for idx, row in enumerate(rows[:max_scan]):
            value = str(row[col]).strip() if col < len(row) else ""
            if not value:
                continue
            total += 1
            if _id_key(value) in known_keys:
                hit += 1
                if not first_hit:
                    first_hit = idx
        if total >= 3 and hit / total >= min_hit:
            score = (hit / total, hit, col)
            if best is None or score > best:
                best = score
    return (best[2], next(i for i, r in enumerate(rows[:max_scan])
                          if best[2] < len(r) and _id_key(r[best[2]]) in known_keys)) \
        if best else None


def _header_row_above(rows: list[list[str]], data_row: int) -> int | None:
    """数据行往上找**最后一个不像数据**的行当表头（跳过空行）。

    官方表的表头行取值不会是检查号，因此"往上第一个不含检查号样式的行"就是表头。
    """
    for idx in range(data_row - 1, -1, -1):
        row = rows[idx]
        if not any(str(c).strip() for c in row):
            continue                                          # 空行：跳过继续往上
        return idx
    return None


def dump_table(path: str, max_rows: int = 8, max_cols: int = 12,
               cell_width: int = 22) -> str:
    """把表**整个摊开**成可读文本（工作表名、行列数、前若干行）。

    用途很直接：当解析失败时不要让人猜表长什么样 —— 直接打印出来看。
    """
    lines: list[str] = [f"文件：{path}"]
    sheets = _sheet_frames(path)
    lines.append(f"工作表数：{len(sheets)}")
    for s_idx, (name, rows) in enumerate(sheets):
        width = max((len(r) for r in rows), default=0)
        level = _sheet_level(name)
        lines.append(f"\n--- sheet{s_idx} {name!r}（级别={level}）: "
                     f"{len(rows)} 行 × {width} 列 ---")
        for r_idx, row in enumerate(rows[:max_rows]):
            cells = [str(c)[:cell_width].ljust(cell_width)
                     for c in row[:max_cols]]
            more = " …" if len(row) > max_cols else ""
            lines.append(f"  行{r_idx:>3}: " + " | ".join(cells) + more)
        if len(rows) > max_rows:
            lines.append(f"  ……（还有 {len(rows) - max_rows} 行）")
    return "\n".join(lines)


def _diagnose_id_columns(sheets: list[tuple[str, list[list[str]]]], known_keys: set[str],
                         top: int = 3, max_scan: int = 300) -> str:
    """解析失败时给出**可执行的**原因：哪一列最像行键、命中多少行。

    "0 行"有两种成因，处理方式相反：
    ① 检查号列存在，但取值与磁盘目录名不是同一套编号（要按值映射或换表）；
    ② 表里根本没有检查号（这表不是字段金标准）。
    只看 `label_field_counts: {}` 分不清，只能反复猜 —— 所以把命中率打出来。

    ⚠️ 只拿**磁盘检查号**去比对每个 sheet：序列级别 / ROI 级别的 sheet
    列的是序列号 / ROI 名，命中率天然为 0，那不代表表有问题。
    """
    if not known_keys:
        return ("  ⚠️ 取不到磁盘上的检查号（数据根下没有病例目录），"
                "只能按列名识别 —— 请先把数据根指到含检查号目录的那一层。")
    lines: list[str] = []
    for s_idx, (sheet_name, rows) in enumerate(sheets):
        tag = f"sheet{s_idx} {sheet_name!r}"
        if not rows:
            continue
        width = max(len(r) for r in rows[:max_scan])
        scored: list[tuple[float, int, int, int]] = []                 # (命中率,命中,非空,列)
        for col in range(width):
            hit = total = 0
            for row in rows[:max_scan]:
                value = str(row[col]).strip() if col < len(row) else ""
                if not value:
                    continue
                total += 1
                if _id_key(value) in known_keys:
                    hit += 1
            if total >= 3 and hit:
                scored.append((hit / total, hit, total, col))
        scored.sort(key=lambda s: (-s[0], -s[1]))
        if not scored:
            head = [str(c)[:18] for c in rows[min(1, len(rows) - 1)][:8]]
            lines.append(f"  {tag}: 没有哪一列的取值能对上磁盘检查号"
                         f"（前几列表头={head}）→ 该 sheet 多半不是订单级别、或不是这张表")
            continue
        for ratio, hit, total, col in scored[:top]:
            name = ""
            for r in rows[:20]:                                        # 该列的表头文字
                if col < len(r) and str(r[col]).strip() and _id_key(r[col]) not in known_keys:
                    name = str(r[col]).strip()[:18]
                    break
            lines.append(f"  {tag}: 第 {col} 列最像检查号（表头 {name!r}）"
                         f"命中 {hit}/{total} 行（{ratio:.0%}）")
        lines.append(f"  → 上面命中率若明显低于 30%，说明该表编号与目录名不是同一套；"
                     f"把 dump 出来的前几行发出来即可确定映射关系")
    return "\n".join(lines)


def _row_to_record(header: list[str], row: list[str]) -> dict[str, Any]:
    """一行 + 表头 → ``{列名: 值}``（表头认不出时用 ``colN`` 占位）。"""
    record: dict[str, Any] = {
        col: (str(row[i]).strip() if i < len(row) else "")
        for i, col in enumerate(header) if col
    }
    if not record:                                                # 表头认不出：用列号占位
        record = {f"col{i}": (str(row[i]).strip() if i < len(row) else "")
                  for i in range(len(row))}
    return record


def _key_forms(value: str) -> set[str]:
    """登记/查找用的键形式：原样、去前导零、以及两者的 casefold，再加归一化键。

    目录名可能是 ``C0E1F8F2-53BA-45BE``，表里是 ``c0e1f8f253ba45be``（或反之），
    只登记原始形式会一条都查不到。

    ⚠️ 归一化后是**空串**的键一律丢掉：``_id_key("检查号")`` 这种"整串都是非字母数字"
    的取值会归一化成 ``""``，留着就会把不同来源的无意义文字**合并成同一个病例**。
    """
    text = str(value).strip()
    stripped = text.lstrip("0") or text
    return {f for f in (text, stripped, text.casefold(), stripped.casefold(),
                        _id_key(text)) if f}


def _case_record(out: dict[str, dict], case_id: str, create: bool = True) -> dict | None:
    """取（必要时新建）某检查号的病例记录，并把**它的所有键形式都指到同一个对象**。

    序列级 / ROI 级的行要挂到病例上，而病例记录可能还没被创建
    （表里只有序列级 / ROI 级 sheet），也可能已由检查级别 sheet 建好 ——
    两条路径必须落到**同一个 dict**，否则后挂的子行会凭空消失。

    ``create=False`` 时只认**已存在**的病例（找不到返回 ``None``）：用于
    "检查号列是靠列名猜出来的"这种不够可靠的场景，避免把 ``SeriesId`` 之类的
    取值当成检查号、凭空造出一批假病例。
    """
    forms = _key_forms(case_id)
    record = next((out[k] for k in forms if k in out), None)
    if record is None:
        if not create:
            return None
        record = {}
    for key in forms:
        out[key] = record
    return record


def _fill_gaps(base: dict, record: dict) -> None:
    """把 ``record`` 里**非空、且 base 还没有**的列填进 base（先到的值优先）。

    多个 sheet 都可能带病例级字段时用它合并：后一张表只补空，
    不会用空值 / 粗粒度取值把前面（权威）的表覆盖掉。
    """
    for col, value in record.items():
        if isinstance(value, list):                                # 子行列表：合并而非覆盖
            base.setdefault(col, []).extend(value)
            continue
        if value == "" or base.get(col, "") != "":
            continue
        base[col] = value


#: 允许从序列级 / ROI 级子行"提"到病例级的**标签列**（三组：各取一个；见 :func:`_promote_case_labels`）。
#:
#: 只认这三组，是因为**只有丢它们才会静默缩小评测分母**（``TumorProbability`` / ``WHO_Grade``
#: 直接从这三组来）。其余字段本来就该留在各自级别上，见 :func:`_promote_case_labels`。
_CASE_LABEL_COLUMN_GROUPS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("病理结果", "pathology", "病理"), ()),
    (("WHO分级", "WHO_grade", "WHO grade", "分级"), ("病理", "pathology")),
    (("glioma_with_label", "胶质瘤"), ()),
)


def _promote_case_labels(out: dict[str, dict]) -> None:
    """把序列级 / ROI 级子行里**取值一致**的**标签列**补进病例级（只补，不覆盖已有值）。

    为什么需要：病例**只出现在**序列级 / ROI 级 sheet 里时（``检查级别`` sheet 没这一行），
    病例记录是**空的** —— 而官方表同样把 ``Study->CLINICAL->病理结果`` 放在
    **ROI 级别 sheet 的 AQ 列**。整条丢掉的表现是"这一例没有金标准"：
    评测分母悄悄变小、还不报错。

    为什么**只提标签列**、而不是把整行兜进去（试过，是错的）：
        序列级 / ROI 级是"**一例多行**"，除标签外的列本来就**逐行不同**。实测复现数据里
        同一病例的两条序列行 ``Signal_T2WI`` 分别是 ``High`` / ``Low`` —— 整行兜底会
        **任取一行**当金标准，表现是"读到了值、而且是错的"，比读不到更难查。
        所以这里：

    * 只认**标签列**（病理结果 / WHO 分级 / Glioma）；
    * 该列在本病例**所有**子行里的非空取值必须完全一致，不一致就整体不采纳并告警；
    * ``检查级别`` sheet 已给非空值的（同末段列名）一律不动 —— 它才是权威（见 README §2.2）。
    """
    done: set[int] = set()
    for rec in out.values():
        if id(rec) in done:                                        # 同一记录有多个别名键
            continue
        done.add(id(rec))
        rows = [r for key in LEVEL_NESTED_KEY.values() for r in (rec.get(key) or [])]
        if not rows:
            continue
        for keywords, exclude in _CASE_LABEL_COLUMN_GROUPS:
            values: set[str] = set()
            col_name = ""
            for row in rows:
                col = _find_col(row, list(keywords),
                                list(exclude) if exclude else None)
                if not col:
                    continue
                col_name = col_name or col
                text = str(row.get(col) if row.get(col) is not None else "").strip()
                if text:
                    values.add(text)
            if not values:
                continue
            if len(values) > 1:                                    # 子行自相矛盾：不猜
                print(f"[labels][告警] 子行里 {col_name!r} 取值不一致 "
                      f"{sorted(values)}，不提升为病例级（避免任取一行当金标准）",
                      flush=True)
                continue
            leaf = _column_leaf(col_name).casefold()
            same_leaf = [k for k in rec
                         if not _is_nested_key(k)
                         and _column_leaf(k).casefold() == leaf]
            if any(str(rec.get(k) or "").strip() for k in same_leaf):
                continue                                           # 检查级别已给：权威
            value = next(iter(values))
            if same_leaf:
                rec[same_leaf[0]] = value                          # 填掉同字段列的空格
            else:
                rec[col_name] = value


def _sheet_plan(rows: list[list[str]], level: str, known_ids: set[str] | None,
                ) -> tuple[list[str], int, int, str] | None:
    """定出这张表的 ``(表头, 数据起始行, 行键列, 实际级别)``；认不出来返回 ``None``。

    分两套走，因为各级别 sheet 的**行键不同**：

    * ``case``：行键是**检查号** —— 先按取值比对磁盘目录名
      （不看列名，最可靠），不行再退回按列名认；
    * ``series`` / ``roi``：行键是**序列号 / ROI 名** —— 只能按列名认。
      这两级 sheet 里没有检查号行键，若还按老逻辑"必须含检查号列"，
      整张表会判成"找不到表头"→ 0 行。

    返回的级别可能与入参 ``level`` 不同：工作表名认不出来（``unknown``）时，
    这里会**按行键列名回退判级**（见下方 ``_FALLBACK_LEVELS``）。
    调用方必须用返回的这个级别去决定"这行怎么挂" —— 工作表名只是线索，
    表头列才是事实。
    """
    if level == "unknown":
        # 工作表名认不出级别（``Sheet1`` / ``序列信息`` / ``病灶`` / 空名…）时，
        # 按"这张表有哪种行键列"判级，顺序**由具体到宽泛**：
        # ROI 级最具体（有 ROI 名）→ 序列级（有序列号）→ 检查级（只有检查号）。
        #
        # 顺序反了的代价：序列级 / ROI 级 sheet 里**同样有检查号列**（要把子行挂到
        # 病例上），若先按检查号判成 case，同病例的多行会互相覆盖 ——
        # 字段看着有值、实际只留最后一行，且不报任何错。
        for cand in _FALLBACK_LEVELS:
            found = _sheet_plan(rows, cand, known_ids)
            if found is not None:
                return found[0], found[1], found[2], cand
        return None

    if level == "case":
        if known_ids:                                              # ① 按取值找检查号列
            found = _best_id_column_by_values(rows, known_ids)
            if found is not None:
                id_col, first_data = found
                head = _header_row_above(rows, first_data)
                if head is not None:
                    return ([str(c).strip() for c in rows[head]], first_data, id_col,
                            "case")
        head = _detect_header(rows, level="case")                   # ② 按列名找
        if head is None:
            return None
        header = [str(c).strip() for c in rows[head]]
        id_name = _find_id_column(header)
        if not id_name:
            return None
        return header, head + 1, header.index(id_name), "case"

    key_cols = LEVEL_KEY_COLUMNS[level]
    head = _detect_header(rows, level=level)
    if head is None:
        return None
    header = [str(c).strip() for c in rows[head]]
    key_col = _find_col_in_list(header, key_cols)
    if key_col is None:
        return None
    return header, head + 1, key_col, level


#: 工作表名判不出级别时的回退判级顺序（具体 → 宽泛，理由见 :func:`_sheet_plan`）。
_FALLBACK_LEVELS: tuple[str, ...] = ("roi", "series", "case")


def _find_case_column(rows: list[list[str]], header: list[str], key_col: int,
                      known_ids: set[str] | None) -> tuple[int | None, bool]:
    """在**序列级 / ROI 级**表里找"检查号列" → ``(列号, 是否可信)``。

    找不到返回 ``(None, False)``，那批子行就只能是"挂不上病例"。

    为什么两套、还带回一个"可信"标志：

    * **按取值**（可信）：拿磁盘上的检查号去比对，不看列名 —— 命中了就是真检查号列，
      用它建病例记录是安全的；
    * **按列名**（不可信）：离线看表（没有 ``known_ids``）时的兜底，用的是
      :data:`_CASE_COLUMN_STRICT` 这套**更严**的名字。既然只是猜的，就
      **只挂到已存在的病例上**（``create=False``）—— 猜错时最多丢几行，
      而不会凭 ``SeriesId`` 造出一批字段全空的假病例。

    无论哪套都必须排除 ``key_col``：序列级表的 ``序列号`` 满足宽松关键词里的
    ``id`` / ``编号``（包含匹配），一不小心就把行键当检查号。
    """
    if known_ids:
        found = _best_id_column_by_values(rows, known_ids)
        if found is not None and found[0] != key_col:
            return found[0], True
    col = _find_col_in_list(header, _CASE_COLUMN_STRICT)
    if col is not None and col != key_col:
        return col, False
    return None, False


def read_structured_table(path: str, known_ids: set[str] | None = None) -> dict[str, dict]:
    """读结构化金标准表（csv 或 xlsx）→ ``{检查号: {列名: 值}}``。

    同一文件里可能**按级别分成多张工作表**（``脑胶质瘤标注结果-训练集.xlsx``
    就是 ``检查级别`` / ``序列级别`` / ``ROI级别`` 三张），四个坑各自有对策：

    1. **表头行不固定**：排版是 ``标题行/索引信息 → 空行/说明 → 真正的表头``，
       每个 sheet 还不一样（第 1 行、第 2 行、第 3 行都可能是索引信息），
       而 ``pandas.read_excel`` 默认把第一行当表头 —— 列名成了"标题/Unnamed"、
       检查号列认不出来，整表解析出 **0 行**。这里改为**扫描前若干行**找表头
       （见 :func:`_detect_header`：必须含该级别的行键列，字段线索最多者胜），
       不假设它在第几行。
    2. **多工作表 + 级别判定**：默认只读第一个 sheet；这里遍历全部 sheet。
       级别先按**工作表名**判（见 :func:`_sheet_level`），名字认不出来
       （``Sheet1`` / ``序列信息`` / 空名）再按**表头列**回退判级
       （ROI 名 → 序列号 → 检查号，见 :func:`_sheet_plan` 的 ``_FALLBACK_LEVELS``）。
       只看工作表名是不够的：认不出时一张序列级 sheet 会被当成检查级别 ——
       同病例的多行互相覆盖，字段看着有值、实际只留最后一行，且不报错。
    3. **行键不同**：检查级别一例一行（键=检查号）、序列级别一例多行
       （键=检查号+序列号）、ROI 级别更多行（键=+ROI 名）。全按检查号合并会让
       同病例的后续行**互相覆盖** —— 字段看着有值，实际来自最后一行（序列级），
       病例级字段全丢，且不报任何错误。所以序列级 / ROI 级的行**原样挂**在病例
       记录的 ``__series_rows__`` / ``__roi_rows__`` 下，不参与字段映射。
    4. **检查号大小写**：目录名是小写哈希、表里可能是大写，
       因此大小写折叠后的键也一并登记。

    一行都没解析出来时会打印告警（含表头预览 + 整表摊开），
    而不是只给上层返回一个空字典。
    """
    out: dict[str, dict] = {}
    sheets = _sheet_frames(path)
    # 先给每张工作表"定级别 + 定表头"，再按**实际级别**排序：
    # 检查级别先合并（病例级字段以它为准），序列 / ROI 级只做嵌套保留。
    #
    # 为什么排序也用实际级别：工作表名可能认不出来（``Sheet1`` / ``序列信息`` /
    # 甚至空名），此时级别由表头列决定（见 :func:`_sheet_plan` 的回退判级）。
    # 若排序仍按表名，认不出名字的**检查级别** sheet 会被排到序列级之后，
    # 它的病例级字段只能"补空"（``_fill_gaps``）——权威取值被前一张表压住。
    prepared: list[tuple[int, str, list[list[str]], str, tuple | None]] = []
    for idx, (name, rows) in enumerate(sheets):
        named = _sheet_level(name)
        prepared.append((idx, name, rows, named,
                         _sheet_plan(rows, named, known_ids) if rows else None))
    ordered = sorted(prepared,
                     key=lambda p: (_LEVEL_ORDER[p[4][3] if p[4] else p[3]], p[0]))
    stats: list[str] = []
    for _, sheet_name, rows, named_level, plan in ordered:
        if not rows:
            continue
        label = sheet_name or f"sheet{len(stats)}"
        if plan is None:
            extra = ("（表名认不出级别，已按 ROI→序列→检查 顺序试过行键列）"
                     if named_level == "unknown" else "")
            stats.append(f"{label}={named_level}/表头未识别{extra}")
            continue
        header, data_start, key_col, level = plan
        # 表名与表头列不一致时以表头列为准，并在统计里标出来（否则会让人以为走错了分支）
        shown = level if level == named_level else f"{level}（表名判为 {named_level}）"
        nested_key = LEVEL_NESTED_KEY.get(level)
        # 序列级 / ROI 级：还得分清"哪一列是检查号"（好把子行挂到病例上）。
        # 检查号列不可信时只挂已有病例（create=False），见 _find_case_column。
        case_col: int | None = None
        case_trusted = False
        if nested_key is not None:
            case_col, case_trusted = _find_case_column(rows, header, key_col, known_ids)

        n_rows = n_dropped = 0
        for row in rows[data_start:]:
            if not any(str(c).strip() for c in row):
                continue                                          # 跳过空行
            key_value = str(row[key_col]).strip() if key_col < len(row) else ""
            if not key_value:
                continue
            record = _row_to_record(header, row)
            if nested_key is None:
                forms = _key_forms(key_value)
                if not forms:                                     # 取值归一化后空：无意义行
                    n_dropped += 1
                    continue
                for key in forms:
                    if key not in out:
                        out[key] = record
                    else:
                        _fill_gaps(out[key], record)
            else:
                attach = (str(row[case_col]).strip()
                          if case_col is not None and case_col < len(row) else "")
                target = (_case_record(out, attach, create=case_trusted)
                          if attach else None)
                if target is None:                                # 挂不上病例：不进结果
                    n_dropped += 1
                    continue
                target.setdefault(nested_key, []).append(record)
            n_rows += 1
        stats.append(f"{label}={shown}/{n_rows} 行"
                     + (f"（{n_dropped} 行挂不到病例）" if n_dropped else ""))

    _promote_case_labels(out)

    if stats:
        print(f"[labels] {os.path.basename(path)} 分级解析：" + "，".join(stats)
              + f" → {len({id(v) for v in out.values()})} 例", flush=True)

    if not out and path not in _WARNED_TABLES:
        _WARNED_TABLES.add(path)
        diag = _diagnose_id_columns(sheets, {_id_key(k) for k in (known_ids or ())})
        print(f"[labels][告警] {os.path.basename(path)} 未解析出任何行。\n"
              f"  已尝试：按工作表名分级 → 按取值比对检查号列 → 按列名识别行键 → 跳过标题行。\n"
              + (diag + "\n" if diag else "")
              + f"  下面把表整个摊开，直接看它长什么样：\n{dump_table(path)}", flush=True)
    return out


def _is_nested_key(key: Any) -> bool:
    """是不是"序列级 / ROI 级子行"挂载用的保留键（``__series_rows__`` 之类）。

    它们**不是真实列**，字段映射必须跳过：否则子行列表会被当成字段取值，
    或反过来被 ``_find_col`` 的包含匹配命中（``__series_rows__`` 里就含 "series"）。
    """
    return str(key).startswith("__")


def _find_col(row: dict, keywords: list[str], exclude: list[str] | None = None) -> str | None:
    """列名匹配：**先按分层列名末段、再按整串**；每轮内部 精确 → 前缀 → 包含。

    ``exclude`` 用于排除干扰列（如 ``*_pattern``）。早期版本直接"包含匹配"会让
    ``t1wi_c_enhan`` 命中 ``tumor_t1wi_c_enhan_pattern``，Enhancement 永远拿不到金标准。

    末段优先：官方表的列名是字段路径（``Study->CLINICAL->病理结果``），末段才是字段名；
    整串匹配会让父段里的关键词抢列（见 :func:`_column_leaf`）。

    列名与关键词都过 :func:`_fold`（大小写 / 全角 / 空格无关）：同一张表里
    ``TUMOR_T1WI_C_ENHAN`` 与 ``tumor_t1wi_c_enhan`` 是同一列，
    漏掉就会"字段没读到"，而字段缺失是**静默**的（下游当金标准不存在）。
    """
    ex = [_fold(e) for e in (exclude or [])]
    items = [(str(k).strip(), k) for k in row if not _is_nested_key(k)]
    full: dict[str, Any] = {_fold(text): key for text, key in items}
    leaf: dict[str, Any] = {}
    for text, key in items:
        leaf.setdefault(_fold(_column_leaf(text)), key)
    for table in (leaf, full):                                     # 末段 → 整串
        for kw in keywords:                                        # 1) 精确
            folded = _fold(kw)
            if folded in table:
                return table[folded]
        for kw in keywords:                                        # 2) 前缀
            folded = _fold(kw)
            for low, key in table.items():
                if low.startswith(folded) and not any(e in low for e in ex):
                    return key
        for kw in keywords:                                        # 3) 包含
            folded = _fold(kw)
            for low, key in table.items():
                if folded in low and not any(e in low for e in ex):
                    return key
    return None


#: 官方 ``5_characteristics.xlsx`` 的列名 → 我们的规范字段
#: （值域见天坛参考实现 ``src/tasks/characteristics/schema.py``）
OFFICIAL_FIELD_COLUMNS = {
    "Glioma": "TumorProbability",            # No / Yes
    "WHO_grade": "WHO_Grade",                # 1/2/3/4
    "Enhancement": "Enhancement",            # false / true
    "EnhancementPattern": "EnhancementPattern",
    "Necrosis": "Necrosis",
    "CysticChange": "CysticChange",
    "Hemorrhage": "Hemorrhage",
    "Calcification": "Calcification",
    "Margin": "Margin",                      # Unclear / Clear
    "Lobulation": "Lobulation",
    "Morphology": "Morphology",              # Regular / Irregular
    "Signal_T2WI": "Signal_T2WI",            # Low / Iso / High
    "Signal_FLAIR": "Signal_FLAIR",
    "Location": "Location",                  # 多标签，`|` 分隔
}

#: 官方列里属于**二分类**的（存成 0/1，与 team 侧 FIELD_ENUMS 一致）
OFFICIAL_BINARY_COLUMNS = {"Glioma", "Enhancement", "Necrosis", "CysticChange",
                           "Hemorrhage", "Calcification", "Margin", "Lobulation"}


def _official_columns_to_fields(row: dict) -> dict[str, Any]:
    """按**官方英文列名**直接映射（权威路径，不猜中文关键词）。

    官方 ``5_characteristics.xlsx`` 的 14 列是
    ``Glioma / WHO_grade / Enhancement / EnhancementPattern / Necrosis /
    CysticChange / Hemorrhage / Calcification / Margin / Lobulation /
    Morphology / Signal_T2WI / Signal_FLAIR / Location``，
    与模拟集的中文列名是两套东西。之前只写中文关键词，官方表自然一条也映射不出来。

    列名与取值比较都过 :func:`_fold`：列名大小写/全角变体（``GLIOMA``、``Ｇｌｉｏｍａ``）
    与取值变体（``YES``/``yes``、``ＮＡ／ＵＮＫ``）都算命中。
    """
    lower = {_fold(k): k for k in row if not _is_nested_key(k)}
    out: dict[str, Any] = {}
    for column, field in OFFICIAL_FIELD_COLUMNS.items():
        key = lower.get(_fold(column))
        if key is None:
            continue
        raw = row.get(key)
        text = "" if raw is None else str(raw).strip()
        if text == "" or _fold(text) in ("nan", "none", "na/unk"):
            continue
        low = _fold(text)
        if column in OFFICIAL_BINARY_COLUMNS:
            # 兼容三种写法：No/Yes、false/true、0/1；Margin 的 Clear=1
            if low in ("yes", "true", "1", "clear"):
                out[field] = 1
            elif low in ("no", "false", "0", "unclear"):
                out[field] = 0
            continue
        if column == "WHO_grade":
            out[field] = text.replace(".0", "")                   # Excel 常读成 3.0
            continue
        out[field] = text                                          # 枚举/多标签：原样保留
    return out


def _grade_from_text(text: Any) -> str | None:
    """从任意写法里抠 WHO 级别 → ``"1".."4"``（认不出返回 ``None``）。

    同一件事在表里有四种写法：``4`` / ``4级`` / ``Ⅳ`` / ``IV``。只做精确匹配
    （:data:`GRADE_MAP` 的 ``脑胶质瘤N级``）时，``胶质瘤WHO 4级`` 这类写法整条丢掉 ——
    字段缺失会被静默当成"这一例没有金标准"，指标分母悄悄变小，看起来一切正常。
    """
    t = str(text if text is not None else "").strip()
    if not t:
        return None
    m = re.search(r"([1-4])\s*级", t)                              # `4级` / `WHO 4 级`
    if m:
        return m.group(1)
    m = re.search(r"([ⅠⅡⅢⅣ])", t)                                 # 全角 `Ⅳ级`
    if m:
        return _CIRCLED_GRADE[m.group(1)]
    t2 = t.replace(".0", "").strip()                               # Excel 常读成 `4.0`
    if t2 in _ROMAN_GRADE.values():
        return t2
    m = re.search(r"\b(iv|iii|ii|i)\b", t, re.IGNORECASE)          # 罗马数字 `IV`
    return _ROMAN_GRADE[m.group(1).lower()] if m else None


# --------------------------------------------------------------------------- #
# 病例级「跳过原因」：`STUDY->CLINICAL->备注`（《格式说明》赛道4 · 检查级别 sheet）
# --------------------------------------------------------------------------- #
#: 「备注」列的候选列名。官方列名是分层写法 ``STUDY->CLINICAL->备注``，
#: 末段 ``备注`` 靠"末段优先"匹配即可命中（见 :func:`_column_leaf`）。
SKIP_REASON_KEYWORDS: tuple[str, ...] = ("备注", "skip_reason", "remark", "comment", "note")

#: 格式说明列出的跳过原因（第 7 类取值是 ``无`` = 未跳过，见 :data:`_NO_SKIP_VALUES`）。
SKIP_REASONS: tuple[str, ...] = ("重点审核", "图像质量问题跳过", "序列缺失跳过",
                                 "构建失败跳过", "报告缺失跳过", "阴性数据跳过")

#: 表示"未跳过"的取值（含表格里常见的空 / nan 写法）。
_NO_SKIP_VALUES = frozenset({"", "无", "none", "nan", "null", "na", "na/unk"})


def skip_reason_of(row: dict) -> str:
    """取一行的「备注」（缺失 / 为 ``无`` / 为空 → 返回 ``""``）。

    这个字段值得单独接进来的原因：格式说明把它定义为**该检查是否被跳过**的唯一
    标记（7 类取值），而工程里原先一处都没读 —— 线上按它剔除病例时，本地训练与
    评测的分母和线上不一致，而且不报任何错。
    """
    col = _find_col(row, list(SKIP_REASON_KEYWORDS))
    if not col:
        return ""
    text = str(row.get(col) if row.get(col) is not None else "").strip()
    return "" if _fold(text) in _NO_SKIP_VALUES else text


def is_hard_skip(reason: str) -> bool:
    """该跳过原因是否意味着**影像不可用**（序列缺失 / 构建失败 / 图像质量）。

    这三类病例在数据里往往缺序列，训练侧 ``pick_series`` 挑不出模态。
    其余三类（重点审核 / 报告缺失 / 阴性数据）影像**是好的**，只是标注流程上被
    官方跳过 —— `阴性数据` 还得当检测负样本用，不能跟着一起丢。
    """
    text = str(reason or "")
    return any(k in text for k in ("序列缺失", "构建失败", "图像质量"))


def _attach_skip_reason(out: dict[str, Any], row: dict) -> None:
    """把「备注」挂进病例记录（键 ``SkipReason``；``无``/空**不挂** → 训练时自动 mask）。"""
    reason = skip_reason_of(row)
    if reason:
        out["SkipReason"] = reason


def structured_from_row(row: dict) -> dict:
    """金标准一行 → 规范字段（缺失字段不出现在结果里 → 训练时自动 mask 掉）。

    **官方英文列名优先**：命中 ``5_characteristics.xlsx`` 的列就直接返回，
    避免再过一遍中文关键词（两套命名混在一起只会互相干扰）。
    """
    official = _official_columns_to_fields(row)
    if official:
        _attach_skip_reason(official, row)
        return official

    out: dict[str, Any] = {}
    _attach_skip_reason(out, row)

    patho = row.get(_find_col(row, ["病理结果", "pathology", "病理"]) or "", "")
    if patho:
        if patho in GRADE_MAP:
            out["WHO_Grade"] = GRADE_MAP[patho]
            out["TumorProbability"] = 1
        elif "胶质瘤" in patho:
            # 更松的写法：`胶质瘤WHO 4级` / `胶质瘤Ⅳ级` / 只写 `胶质瘤`（级别在另一列）。
            # 先认病名（含"胶质瘤"就是阳性），级别能从文字里抠出来就顺手用上。
            out["TumorProbability"] = 1
            grade = _grade_from_text(patho)
            if grade:
                out["WHO_Grade"] = grade
        elif patho in NON_GLIOMA or any(k in patho for k in ("转移", "脓肿", "梗死", "无")):
            out["TumorProbability"] = 0

    if "WHO_Grade" not in out:
        # 级别单独占一列的表（`WHO分级` / `分级`，取值 1~4 或 I~IV）
        grade_col = _find_col(row, ["WHO分级", "WHO_grade", "WHO grade", "分级"],
                              exclude=["病理", "pathology"])
        grade = _grade_from_text(row.get(grade_col)) if grade_col else None
        if grade:
            out["WHO_Grade"] = grade
            out.setdefault("TumorProbability", 1)

    gl = row.get(_find_col(row, ["glioma_with_label", "胶质瘤"]) or "", "")
    if gl in ("是", "否"):
        out.setdefault("TumorProbability", 1 if gl == "是" else 0)

    def _yn(field: str, kws: list[str], exclude: list[str] | None = None) -> None:
        col = _find_col(row, kws, exclude)
        if col:
            v = to_enum(row.get(col), YESNO_MAP)
            if v is not None:
                out[field] = int(v)

    _yn("Enhancement", ["tumor_t1wi_c_enhan", "t1wi_c_enhan", "强化"], exclude=["pattern", "形态"])
    _yn("Necrosis", ["tumor_feature_necrosis", "necrosis", "坏死"])
    _yn("CysticChange", ["tumor_feature_change", "cysts", "cystic", "囊变"])
    _yn("Hemorrhage", ["tumor_feature_hemorrhage", "hemorrhage", "出血"])
    _yn("Calcification", ["tumor_feature_calcification", "calcification", "钙化"])
    _yn("Lobulation", ["lesion_mor_feature_lobulation", "lobulation", "分叶"])
    _yn("Margin", ["lesion_morph_feature_boundary", "boundary", "边界"])

    for field, kws, mp, ex in (
            ("Morphology", ["lesion_morphology", "病灶形态"], MORPH_MAP, ["feature"]),
            ("Signal_T2WI", ["tumor_t2wi_signal_intensity", "t2wi_signal", "t2信号"], SIGNAL_MAP, []),
            ("Signal_FLAIR", ["tumor_t2_flair_sign_intensity", "flair_sign", "flair信号"], SIGNAL_MAP, []),
            ("EnhancementPattern", ["tumor_t1wi_c_enhan_pattern", "enhan_pattern", "强化形态"],
             ENHAN_PATTERN_MAP, []),
            ("Location", ["location_of_lesion", "病灶位置"], LOCATION_MAP, [])):
        col = _find_col(row, kws, ex)
        if col:
            v = to_enum(row.get(col), mp)
            if v is not None:
                out[field] = v
    return out


def _norm_key(value: Any) -> str:
    """归一化查表键（**全角→半角** + 去空白 + 大小写无关）。

    多一步 NFKC 是给"中文输入法全角"准备的：``Ｔ１ＣＥ`` → ``T1CE``、
    全角检查号 ``１２３`` → ``123``。读表（:func:`read_series_types`）与查表
    （:func:`lookup_series_type`）都走本函数，两边口径天然一致 ——
    键归一化只要有一处不一致，就是"表里有、却一条都查不到"的静默错配。
    """
    s = unicodedata.normalize("NFKC", str(value if value is not None else ""))
    return re.sub(r"\s+", "", s).casefold()


#: 公开别名：序列类型表与其它模块共用同一套键归一化规则，
#: 各写一份迟早会出现"一处去空白、一处不去"的静默错配。
norm_key = _norm_key


def build_uid_index(series_types: dict | None) -> dict[str, str]:
    """``{(检查号, 序列号): 类型}`` → ``{序列号: 类型}``（UID 单键回退索引）。

    为什么需要它：类型表的**检查号列**与磁盘上的病例目录名并非总能对上
    （平台匿名化口径不同、前导零、目录名是哈希而表里是原始检查号），
    而 **SeriesUid 与影像同源**，是两边唯一必然一致的键。精确键查不到时按 UID
    单键回退，能把整批"看起来没模态"的病例救回来。

    在探针里**每次运行只构建一次**（表可能上万行，别放进每病例的循环里）。
    """
    out: dict[str, str] = {}
    for key, value in (series_types or {}).items():
        if not value:
            continue
        uid = key[1] if isinstance(key, (tuple, list)) and len(key) > 1 else key
        out.setdefault(_norm_key(uid), str(value))
    return out


def lookup_series_type(series_types: dict | None, accession: str = "",
                       uid_candidates: tuple | list = (),
                       uid_index: dict | None = None) -> str:
    """两级查表：``(检查号, 序列号)`` 精确键 → ``序列号`` 单键回退。

    ``uid_candidates`` 按可靠性降序给（如 ``(序列目录名, 文件名主干)``）。
    未传 ``uid_index`` 时本函数自行构建（单次调用用；循环里请在外面建好传进来）。
    """
    if not series_types:
        return ""
    if isinstance(uid_candidates, str):
        # 裸字符串会被当成"候选的字符序列"逐个查（``"2.25.1001"`` → ``"2"``、``"."``…）
        # → 精确键与 UID 回退两轮全落空、**静默返回空表**：调用方以为"表里没有这一路"，
        # 实际只是参数形态不对。这里收成单元素候选（表键本身是 UID，形态唯一）。
        uid_candidates = (uid_candidates,)
    for uid in uid_candidates:
        value = series_types.get((_norm_key(accession), _norm_key(uid)))
        if value:
            return str(value)
    index = uid_index if uid_index is not None else build_uid_index(series_types)
    for uid in uid_candidates:
        value = index.get(_norm_key(uid))
        if value:
            return str(value)
    return ""


def describe_modality_sources(root: str | os.PathLike | None = None) -> str:
    """一句话自检"模态来源现在什么状态"（专供报错文案，省掉一轮来回排查）。

    形如 ``数据信息表 SeriesType.xlsx=<路径或"未找到">；体素判别模型 <路径>=存在/缺失``。
    两者都不可用时，任何模态相关报错都会附带它 —— 用户立刻能分清是"表没接上"
    还是"体素模型没装"，不必猜。

    ``root`` 传**病例目录**也行：候选目录含数据根/父/祖父，正好覆盖到
    ``training/annotation/``（``SeriesType.xlsx`` 与病例目录同层）。
    """
    found = [(name, find_named_table(name, root)) for name in SERIES_TYPE_FILENAMES]
    table_desc = "；".join(f"数据信息表 {name}={path}" for name, path in found if path)
    if not table_desc:
        table_desc = (f"数据信息表 {SERIES_TYPE_TABLE}=未找到（已搜 "
                      "数据根/父/祖父 + 像标注容器的子目录，以及显式指定的 "
                      "$GLIOMA_LABELS_DIR；它就在数据里、与病例目录同层）")
    try:
        from .modality_model import DEFAULT_MODEL_PATH
        model_desc = (f"体素判别模型 {DEFAULT_MODEL_PATH}="
                      f"{'存在' if Path(str(DEFAULT_MODEL_PATH)).is_file() else '缺失'}")
    except Exception:                                     # 依赖不全也不能让报错本身再抛
        model_desc = "体素判别模型=不可用（依赖缺失）"
    return f"{table_desc}；{model_desc}"


#: sidecar JSON 里可能承载序列类型的键（按优先级）
SIDECAR_DESC_KEYS = ("SeriesType", "series_type", "SeriesDescription",
                     "ProtocolName", "SequenceName", "modality", "Modality")

#: **明确的**掩膜线索。判断"这条序列是掩膜"时必须命中其中之一：
#: 类型表里的普通影像名可能带 ``增强``/``et`` 这类词（如 "T1增强"），
#: 而它们同时也是 ``mask_role_for`` 的 core 关键词——不先卡一道，
#: 会把**增强影像**误判成掩膜，于是输入通道里少一个模态、多一个标签。
STRICT_MASK_KW = ("mask", "seg", "label", "roi", "掩码", "标注",
                  "瘤体", "水肿", "异常", "核心", "病灶", "肿瘤区")


def has_strict_mask_hint(text: str) -> bool:
    """该文本是否含**明确**的掩膜线索（大小写无关；中文不受影响）。"""
    low = (text or "").lower()
    return any(k in low for k in STRICT_MASK_KW)

#: 已告警过"发现类型表但缺依赖"（避免每例刷屏）
_WARNED_SERIES_TYPE_DEP = False


#: 类型表的列名候选（顺序即优先级；用**包含**匹配，故短词靠后）。
#:
#: ⚠️ 取值看**列位置**不看关键词顺序：:func:`read_series_types` 里是"逐列"扫，
#: 第一个命中任一关键词的列即当选 —— 同表同时有 ``SeriesType`` 与描述列时，
#: 位置靠前的那个生效（实测表只有 ``AccessionNumber/SeriesUid/SeriesType`` 三列，
#: 不存在歧义；这里记一笔是给"以后多了个描述列"的情形留线索）。
_SERIES_TYPE_ALIASES: dict[str, tuple[str, ...]] = {
    "acc": ("accessionnumber", "accession", "检查号", "检查编号", "病例号"),
    "uid": ("seriesinstanceuid", "seriesuid", "序列号", "序列uid"),
    # 「序列描述」这一列是**格式说明里唯一还没被认的列名**，三种写法都要认：
    #   * 官方口径 ``DetailDescription`` —— "序列描述；对于序列扫描期相的描述，
    #     一般包含层厚和期相内容"；
    #   * 数据集**实测拼写** ``SeriesDescription``（与 DICOM 标签 (0008,103E) 同名；
    #     ROI 级别 sheet 的 Y 列）；
    #   * 分层写法 ``Study->IMAGE->序列描述``（末段 ``序列描述`` 靠"包含"已能命中，
    #     上面两种英文拼写则一条都命不中 → 整表读成 0 条）。
    # 它承载的正是那 5 类模态取值（``T1`` / ``T2-FLAIR`` / ``T1CE（增强）`` / ``其他``），
    # 认不出的表现是"表找到了、却是空的"，最容易被误判成数据损坏或路径写错。
    # 注意**不要**收录裸 ``description``：它会命中 ``StudyDescription`` 这类与模态
    # 无关的描述列，把整表的类型读成自由文本（模态全空，且不报错）。
    # 末位 ``serisdescription``（少一个 ``e``）是**历史笔误**留下的兜底：实测表头并没有
    # 这个拼法，留着是照 :data:`ID_COLUMN_KEYWORDS` 的惯例"拼写变体只增不减"。
    "typ": ("seriestype", "type", "序列类型", "模态", "序列描述",
            "detaildescription", "seriesdescription", "serisdescription"),
}

#: "像模态取值"的前缀（用于**按取值**找类型列，见 :func:`_sniff_series_type_columns`）。
#: 配上长度上限后，检查号 / 序列号这类长串一律不会命中。
_MODALITY_VALUE_TOKENS = (
    "t1", "t1c", "t1ce", "t1wi", "t1w", "t2", "t2w", "t2wi", "t2flair", "flair",
    "其他", "其它", "other", "none", "无", "增强", "平扫", "adc", "dwi",
)
#: 模态取值的长度上限：``T1CE（增强）`` 也就 8 个字符，长串必不是模态。
_MODALITY_VALUE_MAXLEN = 16


def _looks_like_modality_value(value) -> bool:
    """该单元格"看着像模态取值"吗（专供列名认不出时的取值嗅探）。"""
    text = re.sub(r"[\s\-_/]+", "",
                  unicodedata.normalize("NFKC", str(value if value is not None else ""))).casefold()
    if not text or len(text) > _MODALITY_VALUE_MAXLEN:
        return False
    return any(text == tok or text.startswith(tok) for tok in _MODALITY_VALUE_TOKENS)


def _looks_like_series_type_header(acc, uid, typ) -> bool:
    """这三个取值是不是"类型表的表头行"。

    多张工作表拼成一个 ``rows`` 后，**每张表都带一遍表头**；不跳过就会出现
    ``(accessionnumber, seriesuid) → SeriesType`` 这种拿表头文字当数据的脏条目。
    """
    def _hit(value, keys: tuple[str, ...]) -> bool:
        text = _norm_key(value)
        return bool(text) and any(k in text for k in keys)

    return (_hit(acc, _SERIES_TYPE_ALIASES["acc"])
            and _hit(uid, _SERIES_TYPE_ALIASES["uid"])
            and _hit(typ, _SERIES_TYPE_ALIASES["typ"]))


def _sniff_series_type_columns(rows: list[list], max_scan: int = 300
                               ) -> tuple[int, int, int] | None:
    """**不看列名**，按取值找出 ``(检查号列, 序列号列, 类型列)``；认不出返回 ``None``。

    为什么需要它：列名是唯一会被"改版"的东西 —— 前两列写成 ``序号/影像编号``、
    加了索引列、或者干脆是 ``A/B/C``，按列名匹配就一条也读不到，
    而**取值**不会变（检查号、DICOM UID、5 类模态取值）。

    判据（全在取值上，不需要任何外部信息）：

    * **类型列**：该列非空取值里"像模态取值"的比例最高且 ≥ 0.5；
    * **序列号列**：剩下两列里，取值含 ``.``（DICOM UID）比例更高 /
      平均更长的那个；
    * **检查号列**：另一个。

    找不到（例如整表只有两列、或该列取值是自由文本）就返回 ``None``，
    绝不在没有把握时硬凑 —— 凑错会把整表挂到错误的键上，比读不到更难查。
    """
    body = [r for r in rows[:max_scan] if any(str(c).strip() for c in r)]
    if len(body) < 3:
        return None
    width = max(len(r) for r in body)
    if width < 3:
        return None
    cols = [[str(r[c]).strip() for r in body if c < len(r) and str(r[c]).strip()]
            for c in range(width)]
    ratio = [(sum(1 for v in vals if _looks_like_modality_value(v)) / len(vals)
              if vals else 0.0) for vals in cols]
    typ_col = max(range(width), key=lambda c: (ratio[c], len(cols[c])))
    if ratio[typ_col] < 0.5:
        return None
    rest = [c for c in range(width) if c != typ_col and cols[c]]
    if len(rest) < 2:
        return None

    def dotted(c: int) -> float:
        return sum(1 for v in cols[c] if "." in v) / len(cols[c])

    def mean_len(c: int) -> float:
        return sum(len(v) for v in cols[c]) / len(cols[c])

    rest.sort(key=lambda c: (dotted(c), mean_len(c)), reverse=True)
    uid_col, acc_col = rest[0], rest[1]
    return acc_col, uid_col, typ_col


def read_series_types(root: str | os.PathLike,
                      labels_dir: str | os.PathLike | None = None
                      ) -> dict[tuple[str, str], str]:
    """读序列类型：``(检查号, 序列号) → 序列类型``（模态的唯一可靠来源）。

    ⚠️ 数据里的序列目录名是 DICOM UID / 哈希，靠"按名字猜关键词"一个都命中不了：
    探针会把整批序列归到 ``other``，训练侧直接报 ``无任何可用序列`` ——
    而病例数、目录结构看起来完全正常，极易被误判成数据损坏或路径写错。

    **只读数据集自带的 ``SeriesType.xlsx``** —— 与病例目录**同层**
    （训练集 ``<阶段>/annotation/``、验证集 ``<阶段>/original/``）；
    ``training`` / ``verification`` 的数据里都有，``evaluation_*`` 评测期
    **随测试数据一起下发**。列 ``AccessionNumber / SeriesUid / SeriesType``，
    取值 5 类：``T1`` / ``T1CE（增强）`` / ``T2-Flair`` / ``T2WI`` / ``其他``。
    它不只在数据根那一层 —— 数据根指成**某一病例目录**或**填高一层**时也要能找到，
    所以统一按候选目录搜（数据根/父/祖父 + 像标注容器的子目录，含 ``original/``）。

    读取上**不假设任何排版**：表头行是哪一行由"能否凑齐三列名"扫出来
    （实测第 1 行，带标题/索引行的版本同样能认）；列名一条都不命中时
    还会按**取值**嗅探三列（见 :func:`_sniff_series_type_columns`），
    所以列名改版、前两列是索引列都读得到。

    **工作区那份 ``工作区兼容表`` 一律不读**：它不是本赛道数据集的内容，
    且取值更粗（只有 ``T1CE``/``T2``/``FLAIR``），一旦参与合并就会把 ``T2WI`` /
    ``T2-Flair`` 静默压平、把 ``其他`` 变成假 ``FLAIR`` ——
    表现是"模态看着都认出来了、通道里却是错的对比度"，比直接报错难查得多。

    缺文件不是错误（返回空表，由上层报错时附自检）；同一文件内同键冲突取值
    **直接失败**（规范 §21）。
    """
    global _WARNED_SERIES_TYPE_DEP

    out: dict[tuple[str, str], str] = {}

    # ★ 数据优先：这张表随数据下发、与检查号目录同层。先在数据目录里直接找，
    #   找不到才退回**显式**指定的 $GLIOMA_LABELS_DIR（不再隐式搜 <工程>/labels
    #   与 $WORKSPACE：那会串表 —— 工作区里若残留另一份数据的 SeriesType.xlsx，
    #   它会先被命中 → 拿训练集的键查验证集，一条都对不上，见
    #   :func:`find_series_type_table_in_data`）。
    path = (find_series_type_table_in_data(root)
            or find_named_table(SERIES_TYPE_TABLE, root, labels_dir))
    if not path:
        return out                                                # 表没接上：交给上层自检

    rows: list[list] = []
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet in wb.worksheets:
                rows.extend(list(sheet.iter_rows(values_only=True)))
        finally:
            wb.close()
    except ImportError:
        try:
            import pandas as pd
            df = pd.read_excel(path, dtype=str, header=None)
            rows = df.fillna("").values.tolist()
        except Exception as exc:                                  # noqa: BLE001
            if not _WARNED_SERIES_TYPE_DEP:
                _WARNED_SERIES_TYPE_DEP = True
                print(f"[probe][告警] 发现 {path} 但既没有 openpyxl 也没有 pandas，"
                      f"序列类型读不到 → UID 命名的序列会全部归到 other。"
                      f"请 pip install openpyxl（{exc}）", flush=True)
            return out

    aliases = _SERIES_TYPE_ALIASES
    # ① 先按**列名**找表头行：哪一行能凑齐三列就用哪一行，不假设第几行
    #    （实测表头就在第 1 行；标题/索引行占位的版本也照样能认出来）。
    idx: dict[str, int] = {}
    data_start = 0
    for i, row in enumerate(rows):
        header = [_norm_key(c) for c in row]
        if not any(header):
            continue
        found: dict[str, int] = {}
        for want, keys in aliases.items():
            for j, h in enumerate(header):
                if any(k in h for k in keys):
                    found[want] = j
                    break
        if set(found) == {"acc", "uid", "typ"}:
            idx, data_start = found, i + 1
            break
    sniffed = False
    if not idx:
        # ② 列名一条都没命中（改版 / 前两列是索引 / 英文缩写）→ 按**取值**嗅探三列。
        #    这是唯一不依赖列名的手段：检查号、DICOM UID、5 类模态取值本身就有形态，
        #    而"列名"是唯一会被改版改掉的东西（含空格、全角括号、加后缀…）。
        sniff = _sniff_series_type_columns(rows)
        if sniff is None:
            if not _WARNED_SERIES_TYPE_DEP:
                _WARNED_SERIES_TYPE_DEP = True
                print(f"[probe][告警] {os.path.basename(path)} 里既没找到 "
                      f"AccessionNumber/SeriesUid/SeriesType 三列、也没能按取值嗅探出它们"
                      f"（{path}）→ 序列类型读不到。请把表头行原样贴出来。", flush=True)
            return out
        acc_col, uid_col, typ_col = sniff
        idx = {"acc": acc_col, "uid": uid_col, "typ": typ_col}
        data_start = 0
        sniffed = True
        print(f"[labels][告警] {os.path.basename(path)} 的列名未识别 → 已按取值定位："
              f"检查号=第 {acc_col + 1} 列、序列号=第 {uid_col + 1} 列、"
              f"类型=第 {typ_col + 1} 列（读到 {len(rows)} 行）。"
              f"若取值明显不对，把表的前几行贴出来。", flush=True)

    seen_here: dict[tuple[str, str], str] = {}                     # 只用于检测本文件内的冲突
    for row in rows[data_start:]:
        try:
            acc, uid, typ = row[idx["acc"]], row[idx["uid"]], row[idx["typ"]]
        except IndexError:
            continue
        if acc in (None, "") or uid in (None, "") or typ in (None, ""):
            continue
        if sniffed:
            # 嗅探模式下靠取值过滤：表头行、说明行的"类型"取值不像模态
            if not _looks_like_modality_value(typ):
                continue
        elif _looks_like_series_type_header(acc, uid, typ):
            continue                                              # 多 sheet 拼接出的重复表头
        key = (_norm_key(acc), _norm_key(uid))
        value = str(typ).strip()
        if not value:
            continue
        if key in seen_here and seen_here[key] != value:
            raise ValueError(
                f"SeriesType.xlsx 冲突：检查号={acc!r} 序列={uid!r} "
                f"同时映射到 {seen_here[key]!r} 与 {value!r}（{path}）")
        seen_here[key] = value
        out[key] = value
    if seen_here:
        print(f"[labels] 已读序列类型表 {os.path.basename(path)}：{len(seen_here)} 条"
              f"（{path}）", flush=True)
    return out


#: 类型表里**明确表示"不是目标模态"**的取值。
#:
#: 平台下发的 ``SeriesType.xlsx`` 给**每一路**序列都标了类型，不属于
#: T1 / T2-Flair / T1CE（增强）的写成 ``其他``。这是**权威结论**，不能再当成
#: "没认出来"丢给体素模型猜：模型只认识 t1c/t2/flair/t1 四类，把一路 DWI/ADC
#: 判成 ``t2``（置信度往往还不低）会往通道里灌错对比度 —— 比留一个空通道更有害；
#: 顺带还白跑一次模型、也把 ``unknown_series_total`` 这个诊断指标带偏。
EXPLICIT_OTHER_VALUES = frozenset({
    "其他", "其它", "其他序列", "非目标", "other", "others", "none", "na", "n/a", "无",
})


def is_explicit_other(text: Any) -> bool:
    """该序列类型是否**明确写着"其他"**（而不是"没写/没认出来"）。

    只认**全等**，不做包含匹配：``其他肿瘤或病变`` 是病灶/病理取值（见
    :data:`NON_GLIOMA`），不是"序列类型=其他"，混在一起会把真正的影像丢掉。
    """
    return _norm_key(text) in EXPLICIT_OTHER_VALUES


def nifti_stem(filename: str) -> str:
    """``x.nii.gz`` / ``x.nii`` → ``x``。"""
    low = filename.lower()
    for ext in (".nii.gz", ".nii"):
        if low.endswith(ext):
            return filename[: -len(ext)]
    return filename


#: 官方掩膜文件名的后缀（《公共数据集格式说明》赛道四原文）：
#: ``<检查号>/<序列UID>_<RoiName>_<RoiNumber>_mask.nii.gz``。
#:
#: ``RoiName`` / ``RoiNumber`` 取自 ``脑胶质瘤标注结果-训练集.xlsx`` 的
#: ``ROI级别`` sheet 的 **AC / AD 列**（``瘤体`` / ``水肿`` / ``肿瘤瘤体`` /
#: ``全肿瘤`` / ``异常信号`` … 与 ``1`` / ``2`` / ``3`` / ``4`` …）。
#: 掩膜与影像**同目录**、只靠文件名区分。
MASK_FILE_SUFFIX = "_mask"


def is_official_mask_name(filename: str) -> bool:
    """该文件名是否为官方掩膜命名（``…_mask.nii.gz``）。

    **必须**按后缀判定，不能靠 ROI 名关键词：``RoiName`` 的取值是开放的
    （"等等"），关键词表之外的取值会让掩膜掉进"认不出模态的序列"里 ——
    评测期被体素模型猜成 ``t1c``/``t2`` 塞进输入通道，**标签当输入**且不报错。
    有这条判据后，认不出角色的掩膜会被明确告警并跳过，而不是混进影像。

    大小写/全角无关（``_MASK`` / 全角扩展名都认）。
    """
    return nifti_stem(filename).casefold().endswith(MASK_FILE_SUFFIX)


#: 大赛检查号的形态：**32 位十六进制**（实测 ``0050d79429cf4d86907dc8c4a34cbf04``）。
#:
#: 这是"这一层就是大赛数据的病例目录"的**唯一判据**，也是"只读大赛数据"的执行点：
#: 平台上 ``<数据根>/<检查号>/…``，检查号必然长这样。任何不满足的顶层目录
#: （别的数据集、随手一个堆着 nii 的目录）都在**读取之前**被挡掉 ——
#: 读了会污染训练与指标，报出来会把排查方向带偏，两者都不允许（2026-09-28）。
OFFICIAL_ACCESSION_RE = re.compile(r"^[0-9a-f]{32}$")


def is_official_accession(name: str) -> bool:
    """该目录名是大赛检查号吗（32 位十六进制；大小写不敏感）。"""
    return bool(OFFICIAL_ACCESSION_RE.match(str(name or "").strip().casefold()))


def series_uid_candidates(stem: str) -> tuple[str, ...]:
    """文件名主干 → 可能的序列 UID 候选（**按可靠度降序**）。

    官方布局是**扁平**的：影像 ``<序列UID>.nii.gz`` 与掩膜
    ``<序列UID>_<RoiName>_<RoiNumber>_mask.nii.gz`` 都直接躺在
    ``<检查号>/`` 下。影像的主干就是裸 UID；掩膜的主干多带了
    ``_RoiName_RoiNumber_mask``，**拿去查 ``SeriesType.xlsx`` 必然落空**。

    DICOM UID 只含数字与点、**不含下划线**，所以第一个 ``_`` 之前就是序列 UID：
    这里返回「整串 + 逐级剥短的各个前缀」，查询按序命中即停
    （整串排最前，保证"影像主干恰好等于 UID"这一最常见情形优先命中）。

    漏认这条的表现（**不报错**）：掩膜模态为 ``None`` → ``瘤体`` 被判成
    **core**（规范里它是 FLAIR/T2 上的 peri）→ 掩膜进错任务空间，且几何
    来自另一条序列 —— 只有指标会悄悄偏低。
    """
    s = unicodedata.normalize("NFKC", str(stem or "")).strip()
    if not s:
        return ()
    out = [s]
    if is_official_mask_name(s):
        parts = s.split("_")
        out.extend("_".join(parts[:i]) for i in range(len(parts) - 1, 0, -1))
    return tuple(dict.fromkeys(x for x in out if x))


def sidecar_desc(path: str | os.PathLike) -> str | None:
    """读同名 JSON sidecar 的序列描述（不存在或解析失败返回 None）。

    JSON 的键是**区分大小写**的精确匹配，而 sidecar 可能是平台导出的
    （``SeriesDescription``）、也可能是别的工具/人工写的（``seriesdescription``、
    ``SERIESTYPE``）。所以先按 :func:`_fold` 建一次键映射再查 ——
    漏认的表现是"旁车明明有描述、却认不出模态"，一路降级且不报错。
    """
    import json

    p = os.path.abspath(str(path))
    stem = nifti_stem(os.path.basename(p))
    sidecar = os.path.join(os.path.dirname(p), stem + ".json")
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:                                             # noqa: BLE001
        return None
    if not isinstance(data, dict):
        return None
    folded = {_fold(k): v for k, v in data.items()}
    for key in SIDECAR_DESC_KEYS:
        value = folded.get(_fold(key))
        if value not in (None, ""):
            return str(value)
    return None


# --------------------------------------------------------------------------- #
# "这一例能不能凑出输入通道"：训练中途崩的根因判据
# --------------------------------------------------------------------------- #
#: 输入通道**实际支持的模态**：与 ``configs/preprocess.yaml`` 的 ``channels``
#: 及各自的 ``fallback`` 完全一致（``t1c``←t1,t2；``flair``←t2；``t2``；``t1``）。
#:
#: ``guess_modality`` 还会返回 ``dwi`` / ``adc`` / ``swi``，它们**不在**这里 ——
#: 那三路序列再多也进不了输入通道。改动 ``channels`` 时这里必须同步。
INPUT_CHANNEL_MODALITIES = frozenset({"t1", "t1c", "t2", "flair"})


def has_input_modality(images: dict | None, unknown_series: list | None = None,
                       channels: list | None = None) -> bool:
    """该病例能否凑出**至少一个**输入通道（纯结构判断，不读体素、不调判别模型）。

    必须单独有这条判据：``images`` 非空 ≠ 有可用通道，而两处"筛查"用的都是前者
    （``probe.scan_real`` 的 ``if not images`` 与 ``GliomaDataset`` 的自检），于是
    下面这类病例会一路活到取样那一刻：

    * 数据信息表里**明写 ``其他``** 的病例 —— ``images`` 是 ``{"other": {...}}``
      （非空！），而且被**刻意**排除在 ``unknown_series`` 之外（"其他"是权威排除、
      不是"没认出来"，见 :func:`is_explicit_other`）；
    * 只有 ``dwi`` / ``adc`` / ``swi`` 的病例（认得出模态，但都不是输入通道）。

    两类都会让 ``dataset.pick_series`` 一个通道都挑不出。**注意口径**：全放开之后
    这不再是失败 —— ``build_case_volume`` 会借该例任意一路影像的几何、把 4 个通道
    置零，让这一例照常参与训练（一例数据不丢）。本判据因此退化成**诊断**：
    训练侧据此**报数**（"有多少例输入侧是全空的"），不据此剔除。

    ``unknown_series`` 非空时返回 True：认不出模态的序列还能靠体素判别模型
    （``dataset.classify_unknown``）救回来 —— 只有"表里明写其他"的病例
    （unknown 必为空）在这里必然是 False。
    """
    imgs = images or {}
    chans = channels if channels is not None else [
        {"name": m, "fallback": []} for m in sorted(INPUT_CHANNEL_MODALITIES)]
    for ch in chans:
        for name in [ch.get("name")] + list(ch.get("fallback") or []):
            if name in imgs:
                return True
    return bool(unknown_series)


# --------------------------------------------------------------------------- #
# 序列描述（模态旁证）：`SeriesType.xlsx` 拿不到时的第二条路
# --------------------------------------------------------------------------- #
#: 「序列描述」列的候选列名。与 :data:`_SERIES_TYPE_ALIASES` 的 ``typ`` 同源：
#: 那一列**本身就是模态取值**（``T1`` / ``T2-Flair`` / ``T1CE（增强）`` / ``其他``），
#: 所以当类型表整个缺失时，标注表里的这一列可以顶上。
#:
#: 三种写法都是实测过的：官方口径 ``DetailDescription``（格式说明原文）、
#: 数据集实测拼写 ``SeriesDescription``（ROI 级别 sheet 的 Y 列）、
#: 分层写法 ``Study->IMAGE->序列描述``（末段靠"包含"命中）。
#: **不要**收录裸 ``description``：那会命中 ``StudyDescription`` 这类与模态无关的列。
SERIES_DESC_KEYWORDS: tuple[str, ...] = (
    "seriesdescription", "serisdescription", "detaildescription",
    "seriestype", "序列描述", "序列类型", "模态",
)

#: 序列 UID 的形态：DICOM UID 只含**数字与点**（``2.25.9002``）。
#: 这条是"UID 列被认成描述列"的兜底判据 —— :func:`_find_col` /
#: :func:`_find_col_in_list` 的"前缀 / 包含"轮里，``series`` 会命中
#: ``SeriesDescription`` **自己**，于是 uid_col == desc_col、取值互相顶替；
#: 加了形态校验后最坏也只是"这一列没用上"。
_SERIES_UID_RE = re.compile(r"^\d[\d.]*$")


def _is_series_uid(value: Any) -> bool:
    """该取值像**序列 UID** 吗（数字 + 点；纯数字也认）。"""
    text = unicodedata.normalize("NFKC", str(value if value is not None else "")).strip()
    return len(text) >= 3 and bool(_SERIES_UID_RE.match(text))


def _is_modality_desc(value: Any) -> bool:
    """该取值像**模态**吗（``T1`` / ``T2-Flair`` / ``其他`` …）。

    必须卡这道：列名认错时（如把 ``StudyDescription`` 当描述列）整列都是自由文本，
    照单全收会把散文塞进"模态"位；过滤后最坏也只是"这一列没用上"。
    """
    text = str(value if value is not None else "").strip()
    return bool(text) and (guess_modality(text) is not None
                           or _looks_like_modality_value(text))


def desc_index_from_records(records) -> dict[str, str]:
    """从 :func:`read_structured_table` 的结果里抽 ``{序列UID: 序列描述}``。

    为什么不重读一遍表：标注表的三个 sheet 已经被 ``read_structured_table``
    解析成病例记录了，序列级 / ROI 级的行**原样**挂在
    :data:`LEVEL_NESTED_KEY` 的两个键下（见那里的说明）。再读一次既慢，
    又要重新处理"表头行不固定"那一整套坑 —— 复用已解析的行最省事也最一致。

    ``SeriesType.xlsx`` 到手之前（或干脆没有它）这可能是**唯一**能判模态的线索：
    行键就是 ``序列UID``，与磁盘上的文件名主干同源，所以能直接对上。
    返回的键走 :func:`_norm_key`（与 ``lookup_series_type`` 同一套归一化），
    两侧口径不一致就是"表里有、却一条都查不到"的静默错配。
    """
    out: dict[str, str] = {}
    seen: set[int] = set()
    for rec in (records or []):
        if not isinstance(rec, dict) or id(rec) in seen:
            continue
        seen.add(id(rec))
        for key_name in LEVEL_NESTED_KEY.values():
            for row in (rec.get(key_name) or []):
                if not isinstance(row, dict):
                    continue
                uid_col = _find_col(row, list(LEVEL_KEY_COLUMNS["series"]))
                desc_col = _find_col(row, list(SERIES_DESC_KEYWORDS))
                if not uid_col or not desc_col or uid_col == desc_col:
                    continue
                uid, desc = row.get(uid_col), row.get(desc_col)
                if _is_series_uid(uid) and _is_modality_desc(desc):
                    out.setdefault(_norm_key(uid), str(desc).strip())
    return out


def read_series_desc_index(path: str | os.PathLike) -> dict[str, str]:
    """**直接读文件**抽 ``{序列UID: 序列描述}``（离线用；在线走
    :func:`desc_index_from_records` 以免重复解析）。

    专为"``SeriesType.xlsx`` 拿不到、但标注表的 ROI 级别 sheet 有 ``SeriesDescription``"
    这一情形准备：那一列承载的正是 5 类模态取值，是模态的第二条可靠来源。
    表头行不固定（第 1~3 行都可能是索引信息），所以仍然逐行扫"哪一行能同时凑出
    序列号列与描述列"，不假设位置。
    """
    out: dict[str, str] = {}
    for _name, rows in _sheet_frames(str(path)):
        idx: tuple[int, int] | None = None
        start = 0
        for i, row in enumerate(rows):
            header = [str(c).strip() for c in row]
            if not any(header):
                continue
            uid_col = _find_col_in_list(header, LEVEL_KEY_COLUMNS["series"])
            desc_col = _find_col_in_list(header, list(SERIES_DESC_KEYWORDS))
            # 两列**必须不同**：列名认错时它们会落到同一列上（见 :data:`_SERIES_UID_RE`）。
            if uid_col is not None and desc_col is not None and uid_col != desc_col:
                idx, start = (uid_col, desc_col), i + 1
                break
        if idx is None:
            continue
        uid_col, desc_col = idx
        for row in rows[start:]:
            if max(uid_col, desc_col) >= len(row):
                continue
            uid, desc = row[uid_col], row[desc_col]
            if _is_series_uid(uid) and _is_modality_desc(desc):
                out.setdefault(_norm_key(uid), str(desc).strip())
    return out


#: 非"字段金标准表"的文件名关键词（见 :func:`find_structured_tables`）。
#: 序列类型表的命名（只认数据集自带的 ``SeriesType``；工作区里别处的表一律不读）
#: 都在其中：它们的列是 ``AccessionNumber/SeriesUid/SeriesType|SeriesLabel``，
#: 一行字段都映射不出来，混进来只会让"解析出 N 行 / 有表但没解析出"这两句诊断互相矛盾。
#: （后者虽已不读，但仍要在**找表**时排除 —— 否则诊断计数还是会被它带偏。）
_NON_LABEL_TABLE_KW = ("seriestype", "series_type",          # 模态表（数据集自带）
                       "serieslabel", "series_label",        # 模态表（工作区那份，已不读）
                       "masklabel", "mask_label",            # 掩膜名表（4_masklabel.xlsx）
                       "gold", "duplicate", "folds")


def find_structured_tables(root: str, max_parents: int = 2) -> list[str]:
    """在数据根（及其**上级 1~2 层**）找结构化金标准表（csv/xlsx）。

    为什么要向上看：官方数据的层级通常是 ``<数据集根>/<某层>/<检查号>/``，
    而字段金标准表常常放在**检查号那一层的上一级**。只扫数据根会出现
    "表明明存在、却一条都没读进来"，报告里只显示 ``{}``，
    分不清是"没表"还是"数据根定位偏了一层"。

    排除另有专用读取器的表（见 :data:`_NON_LABEL_TABLE_KW`）：序列类型表
    （``SeriesType.xlsx`` / ``工作区兼容表``）、掩膜名表（``4_masklabel.xlsx``）、
    重复影像金标准（``gold*.csv`` / ``2_duplicate.xlsx``）—— 它们都不是字段金标准，
    混进来会污染"解析出 N 行"这个计数，把诊断信息带偏。

    唯一**保留**的官方表是 ``5_characteristics.xlsx``（14 个英文列的字段金标准）。
    """
    roots: list[str] = [os.path.abspath(root)]
    parent = roots[0]
    for _ in range(max(0, int(max_parents))):
        parent = os.path.dirname(parent)
        if not parent or parent == os.sep or not os.path.isdir(parent):
            break
        roots.append(parent)

    hits: set[str] = set()
    for idx, base in enumerate(roots):
        if idx == 0:                                              # 数据根：整棵树
            for dirpath, _dirs, files in os.walk(base):
                hits.update(os.path.join(dirpath, fn) for fn in files)
        else:                                                     # 上级：只看本层
            try:
                hits.update(os.path.join(base, fn) for fn in os.listdir(base))
            except OSError:
                continue

    out: list[str] = []
    for path in sorted(hits):
        fn = os.path.basename(path)
        if not fn.endswith((".csv", ".xlsx", ".xls")) or fn.startswith("~$"):
            continue
        if any(k in fn.lower() for k in _NON_LABEL_TABLE_KW):
            continue
        out.append(path)
    return out
