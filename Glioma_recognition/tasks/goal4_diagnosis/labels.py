"""Goal4 的**内部**领域标签（规范 §9.5）。

规范明确：本文件只维护内部枚举，**比赛字符串一律不写在这里**
（它们在 ``output/competition_mapping.py`` 单点维护）。

这样做的实际收益：组委会若调整枚举拼写（这类改动在赛前很常见），
只需改映射表一处；若把字符串直接写进模型侧代码，就要在十几个字段间
逐个排查，且极易漏改——而漏改的后果是 schema 校验失败、整例作废。
"""
from __future__ import annotations

from enum import Enum, auto


class InternalLocation(Enum):
    """病灶位置（15 类，含 Other / NA 兜底）。"""

    RIGHT_FRONTAL = auto()
    LEFT_FRONTAL = auto()
    RIGHT_TEMPORAL = auto()
    LEFT_TEMPORAL = auto()
    RIGHT_PARIETAL = auto()
    LEFT_PARIETAL = auto()
    RIGHT_OCCIPITAL = auto()
    LEFT_OCCIPITAL = auto()
    RIGHT_CEREBELLUM = auto()
    LEFT_CEREBELLUM = auto()
    RIGHT_BASAL_GANGLIA = auto()
    LEFT_BASAL_GANGLIA = auto()
    BRAINSTEM = auto()
    OTHER = auto()
    NA = auto()


class InternalMorphology(Enum):
    REGULAR = auto()
    IRREGULAR = auto()
    NA = auto()


class InternalEnhancementPattern(Enum):
    NONE = auto()
    RING = auto()
    RIM_ENHANCING = auto()
    NODULAR = auto()
    GROUND_GLASS = auto()
    GYRIFORM = auto()
    MULTIFOCAL = auto()
    OTHER = auto()


class InternalSignal(Enum):
    LOW = auto()
    ISO = auto()
    HIGH = auto()


class InternalWhoGrade(Enum):
    GRADE_1 = auto()
    GRADE_2 = auto()
    GRADE_3 = auto()
    GRADE_4 = auto()


#: 分类头定义：``(字段名, 内部枚举类)``。训练与推理都以此为准，
#: 保证"头的顺序"与"标签含义"在任何地方都不会错位。
FIELD_ENUMS: tuple[tuple[str, type[Enum] | None], ...] = (
    ("TumorProbability", None),                                   # 二分类，1 个 logit
    ("Location", InternalLocation),
    ("Morphology", InternalMorphology),
    ("WHO_Grade", InternalWhoGrade),
    ("Enhancement", None),
    ("EnhancementPattern", InternalEnhancementPattern),
    ("Necrosis", None),
    ("CysticChange", None),
    ("Hemorrhage", None),
    ("Calcification", None),
    ("Margin", None),
    ("Lobulation", None),
    ("Signal_T2WI", InternalSignal),
    ("Signal_FLAIR", InternalSignal),
)


def cls_spec() -> list[tuple[str, int]]:
    """分类头规格 ``[(key, n_classes)]``；二分类为 1。"""
    return [(name, 1 if enum_cls is None else len(enum_cls))
            for name, enum_cls in FIELD_ENUMS]


def enum_names(field: str) -> list[str]:
    """某字段的内部枚举成员名（顺序即类别索引顺序）。"""
    for name, enum_cls in FIELD_ENUMS:
        if name == field:
            return [] if enum_cls is None else [m.name for m in enum_cls]
    return []
