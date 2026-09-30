"""内部枚举 ↔ 比赛字符串的**唯一**映射点（规范 §9.5、§10）。

规范要求："Goal4 的 ``labels.py`` 只维护内部领域标签，正式比赛字符串集中放在
``output/competition_mapping.py``"。这样组委会一旦调整枚举拼写，只改本文件即可，
不必触碰模型代码——这是把"协议不确定性"隔离在适配层的关键。

**本文件里所有字符串都必须与组委会最终 schema 逐字一致**，因此：

- 映射表集中在此，且**只允许单点维护**；
- 反向映射由本模块自动生成并做**双向一致性校验**（避免手工维护两份表后悄悄不一致）；
- 未收录的取值**显式失败**而不是静默透传——静默透传会让非法枚举进到 JSON，
  被 schema 拒绝时已经很难定位是哪一例、哪一字段。
"""
from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------- #
# 位置（规范示例为字符串，不是 {predicted, probabilities} 结构）
# --------------------------------------------------------------------------- #
LOCATION_TO_COMPETITION: dict[str, str] = {
    "RIGHT_FRONTAL": "RightFrontal",
    "LEFT_FRONTAL": "LeftFrontal",
    "RIGHT_TEMPORAL": "RightTemporal",
    "LEFT_TEMPORAL": "LeftTemporal",
    "RIGHT_PARIETAL": "RightParietal",
    "LEFT_PARIETAL": "LeftParietal",
    "RIGHT_OCCIPITAL": "RightOccipital",
    "LEFT_OCCIPITAL": "LeftOccipital",
    "RIGHT_CEREBELLUM": "RightCerebellum",
    "LEFT_CEREBELLUM": "LeftCerebellum",
    "RIGHT_BASAL_GANGLIA": "RightBasalGanglia",
    "LEFT_BASAL_GANGLIA": "LeftBasalGanglia",
    "BRAINSTEM": "Brainstem",
    "OTHER": "Other",
    "NA": "NA",
}

# --------------------------------------------------------------------------- #
# 形态（规范示例只有两类；内部保留 NA 兜底）
# --------------------------------------------------------------------------- #
MORPHOLOGY_TO_COMPETITION: dict[str, str] = {
    "REGULAR": "Regular",
    "IRREGULAR": "Irregular",
    "NA": "NA",
}

# --------------------------------------------------------------------------- #
# 强化形态（8 类，含 "None"）
# --------------------------------------------------------------------------- #
ENHANCEMENT_PATTERN_TO_COMPETITION: dict[str, str] = {
    "NONE": "None",
    "RING": "Ring",
    "RIM_ENHANCING": "RimEnhancing",
    "NODULAR": "Nodular",
    "GROUND_GLASS": "GroundGlass",
    "GYRIFORM": "Gyriform",
    "MULTIFOCAL": "Multifocal",
    "OTHER": "Other",
}

# --------------------------------------------------------------------------- #
# 信号强度（T2WI / FLAIR 共用）
# --------------------------------------------------------------------------- #
SIGNAL_TO_COMPETITION: dict[str, str] = {
    "LOW": "Low",
    "ISO": "Iso",
    "HIGH": "High",
}

#: WHO 分级：规范示例里 predicted 是**数字**，probabilities 的键是字符串
WHO_GRADE_TO_COMPETITION: dict[str, str] = {
    "GRADE_1": "1",
    "GRADE_2": "2",
    "GRADE_3": "3",
    "GRADE_4": "4",
}

#: 全部字段的映射表（供反向校验与 Aggregator 统一遍历）
ALL_MAPPINGS: dict[str, dict[str, str]] = {
    "Location": LOCATION_TO_COMPETITION,
    "Morphology": MORPHOLOGY_TO_COMPETITION,
    "EnhancementPattern": ENHANCEMENT_PATTERN_TO_COMPETITION,
    "Signal_T2WI": SIGNAL_TO_COMPETITION,
    "Signal_FLAIR": SIGNAL_TO_COMPETITION,
    "WHO_Grade": WHO_GRADE_TO_COMPETITION,
}


class UnknownEnumError(ValueError):
    """内部枚举未在映射表中登记。"""


def to_competition(field: str, internal: Any) -> str:
    """内部枚举 → 比赛字符串。

    Raises:
        UnknownEnumError: 字段未登记或取值未登记（**显式失败**，不静默透传）。
    """
    table = ALL_MAPPINGS.get(field)
    if table is None:
        raise UnknownEnumError(f"字段 {field!r} 未登记映射（已知：{sorted(ALL_MAPPINGS)}）")
    key = str(internal).strip().upper()
    if key not in table:
        raise UnknownEnumError(f"{field}: 内部取值 {internal!r} 未登记（已知：{sorted(table)}）")
    return table[key]


def to_internal(field: str, competition: str) -> str:
    """比赛字符串 → 内部枚举（评估/回读时用）。"""
    table = ALL_MAPPINGS.get(field)
    if table is None:
        raise UnknownEnumError(f"字段 {field!r} 未登记映射")
    for k, v in table.items():
        if v == competition:
            return k
    raise UnknownEnumError(f"{field}: 比赛取值 {competition!r} 无对应内部枚举")


def check_bijection() -> dict[str, list[str]]:
    """校验各映射表**双向唯一**，返回有问题的字段。

    若两个内部枚举映射到同一个字符串（或反之），反向映射就会歧义——
    这类问题在正向写出时完全看不出来，只在读回或统计时暴露。
    """
    problems: dict[str, list[str]] = {}
    for field, table in ALL_MAPPINGS.items():
        vals = list(table.values())
        dup = sorted({v for v in vals if vals.count(v) > 1})
        if dup:
            problems[field] = dup
    return problems


def coverage_report() -> dict[str, dict[str, int]]:
    """各字段的映射规模，便于向组委会核对枚举全集。"""
    return {f: {"internal": len(t), "competition": len(set(t.values()))}
            for f, t in ALL_MAPPINGS.items()}
