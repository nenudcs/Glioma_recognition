"""配置与路径工具。

设计要点：
- 平台路径（WORKSPACE）与数据根（DATASET_ROOT）支持环境变量覆盖，便于在
  云桌面 / 训推平台容器 / 本地三处运行同一份代码；
- 训练日志目录按《赛事开发规范》要求自动切到 ``{WORKSPACE}/logs``（否则可能被标记为可疑对象）。
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Any

import yaml

# --------------------------------------------------------------------------- #
# 控制台编码兜底：**只防止"打印就崩"**，不改任何业务行为
# --------------------------------------------------------------------------- #
# 诊断文案里带 ✗ / ⚠️ / ✓ 这类字符，在 **GBK 控制台**（Windows 中文环境）上
# ``print`` 会抛 ``UnicodeEncodeError``。危险的不是"少打一行"，而是它**打在了
# 关键路径上**：
#   · ``assert_data_source`` 的"数据源标识不一致（**仅告警**）"会变成硬失败；
#   · ``16_finalize.sh`` 的验证集探针把任何异常吞成"未接入"→ **验证集配好了却
#     静默回退折内口径**（实测踩过：清单在、掩膜在，评估却一路走 OOF）；
#   · 探针的 ``[probe][告警]`` 系列同理。
# 因此导入本模块时把两个标准流的错误策略置为 ``replace``：非 UTF-8 终端上那些
# 字符退化成 ``?``，其余内容照常输出。UTF-8 环境下行为完全不变。
def _make_std_streams_unicode_safe() -> None:
    for name in ("stdout", "stderr"):
        try:
            getattr(sys, name).reconfigure(errors="replace")      # Python 3.7+
        except Exception:                                        # noqa: BLE001
            pass                                                 # 被重定向/无此 API 时忽略


_make_std_streams_unicode_safe()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")

#: 平台私有存储（《训推平台使用指南》）；容器内被自动挂载
WORKSPACE = os.environ.get("WORKSPACE", "/2026aicompetition/workspace")
#: 《赛事开发规范》强制的训练日志目录
PLATFORM_LOGS_DIR = os.path.join(WORKSPACE, "logs")


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name: str) -> Any:
    """加载 configs/<name>（name 可带 .yaml 后缀）。"""
    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"
    return load_yaml(os.path.join(CONFIG_DIR, name))


def resolve(path: str) -> str:
    """相对路径按项目根解析；绝对路径原样返回。"""
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


#: 官方数据挂载前缀（《训推平台使用指南》）
OFFICIAL_PREFIXES = ("/2026aicompetition/datasets", "/2026aicompetition/public_")


def dataset_root() -> str:
    """当前使用的数据根（``DATASET_ROOT`` 优先，其次 paths.yaml 的 raw.track4）。"""
    if os.environ.get("DATASET_ROOT"):
        return os.environ["DATASET_ROOT"]
    try:
        return str((load_config("paths.yaml").get("raw") or {}).get("track4", ""))
    except Exception:                                             # noqa: BLE001
        return ""


def val_root() -> str:
    """官方验证集数据根（``VAL_ROOT`` 优先，其次 paths.yaml 的 ``raw.val``）；未配置返回 ""。

    与 :func:`dataset_root` 对称：路径解析统一走这里，避免各处直接读 env/yaml
    造成"有人在 env 里覆盖了、有人没读到"这种半路切换数据源的隐蔽问题。
    """
    if os.environ.get("VAL_ROOT"):
        return os.environ["VAL_ROOT"]
    try:
        return str((load_config("paths.yaml").get("raw") or {}).get("val") or "")
    except Exception:                                             # noqa: BLE001
        return ""


def is_official_data(root: str | None = None) -> bool:
    """数据是否来自大赛官方挂载目录。"""
    r = os.path.abspath(root or dataset_root() or "")
    return any(r.startswith(p) for p in OFFICIAL_PREFIXES)


def data_source_tag(root: str | None = None, phase: str = "train") -> str:
    """按《赛事开发规范》生成 ``data_source`` 日志标识。

    **重要**：早期实现把该字段硬编码为 ``official/train_v1``；若用本地/公开数据
    训练也记成 official，属于**日志不实**。现在按数据根自动区分：

    - 官方挂载路径 → ``official/<phase>_v1``
    - 其他路径     → ``local/<目录名>/<phase>``
    """
    r = os.path.abspath(root or dataset_root() or "")
    if is_official_data(r):
        return f"official/{phase}_v1"
    name = os.path.basename(r.rstrip("/")) or "unknown"
    return f"local/{name}/{phase}"


def assert_data_source(man: dict, phase: str = "train", strict: bool = True) -> str:
    """校验清单（manifest）的数据源与本机当前数据源是否同类。

    **合规闸门**：赛事规则要求"比赛数据只能在大赛专属环境中使用"，
    且日志必须可追溯数据来源。若拿本地/公开数据（本地/公开数据）生成的清单
    去训练官方数据（或反之），结果与日志都会不可追溯 —— 因此默认**拒绝执行**。

    返回当前数据源标识；不一致时抛 ``RuntimeError``（``strict=False`` 时仅告警）。
    """
    man_tag = str((man or {}).get("data_source") or "")
    cur_tag = data_source_tag(phase=phase)
    if not man_tag:
        print("[guard] ⚠️ 清单缺少 data_source 字段（可能是旧清单）；"
              "建议重新执行 bash scripts/01_probe.sh")
        return cur_tag
    if man_tag.split("/")[0] != cur_tag.split("/")[0]:
        msg = (f"数据源不一致：清单={man_tag}（{man.get('data_root')}）  "
               f"当前={cur_tag}（{dataset_root()}）。\n"
               "  · 若要用**官方数据**训练：确认 DATASET_ROOT 指向大赛挂载目录，然后\n"
               "    bash scripts/01_probe.sh && bash scripts/02_build_dataset.sh\n"
               "  · 若只是**本地验证**：请使用独立的 CACHE_DIR / CKPT_DIR，不要与正式产物混用")
        if strict:
            raise RuntimeError(msg)
        print(f"[guard] ⚠️ {msg}")
    elif man_tag != cur_tag:
        print(f"[guard] ⚠️ 数据源标识变化：清单={man_tag} 当前={cur_tag}（将继续，请确认）")
    return cur_tag


def external_val_manifest(verify: bool = True) -> tuple[dict | None, str]:
    """官方验证集清单（**可选链路**）：不可用时返回 ``(None, 清单路径)``。

    这是"最终指标以官方验证集为准"的唯一入口：训练侧不看它；
    评估侧（04/14/15/16）在它可用时切到 external 分支，
    否则原样回退折内 val（OOF / 留一折集成）。

    判定"可用"的条件刻意保守：文件在、能解析、有病例（三条缺一即回退）。
    数据源标识不一致只**告警**不拒绝 —— 用本地复现的验证集跑通流程是允许的，
    但日志里必须留下痕迹（合规要求可追溯）。
    """
    path = resolve(str(load_paths().get("manifest_val") or "data/manifest_val.json"))
    if not os.path.isfile(path):
        return None, path
    try:
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
    except Exception as e:                                        # noqa: BLE001
        print(f"[guard] ⚠️ 验证集清单解析失败（{path}）：{e} → 回退折内 val")
        return None, path
    if not (man.get("cases") or []):
        print(f"[guard] ⚠️ 验证集清单没有病例（{path}）→ 回退折内 val")
        return None, path
    if verify:
        assert_data_source(man, phase="val", strict=False)
    return man, path


def external_val_cases(require_label: bool = True) -> tuple[list[dict], str]:
    """官方验证集病例（**全量训练**模式的验证集来源）。

    与 :func:`external_val_manifest` 共用同一个可用性判定入口，额外**只保留带掩膜的**
    病例。为什么不是 ``masks or labels``：全量模式的 val 只用来算 Dice 选 best
    （``checkpoint_metric: val_dice_peri``），结构化标签在 val 里**不参与任何计算**；
    而没有掩膜的病例会让 ``make_targets`` 产出**全零 target** —— Dice 要么恒 0，
    要么在"预测也为空"时按 ``den == 0`` 记成 **1.0 的假满分**，把 best 直接选歪
    （官方验证集实测没有字段金标准表，若它同时也没有掩膜，这种情况会全量命中）。
    被剔掉几例由调用方打印，不静默。
    """
    man, path = external_val_manifest()
    if not man:
        return [], path
    cases = list(man.get("cases") or [])
    if require_label:
        cases = [c for c in cases if c.get("masks")]
    return cases, path


def fold_ckpts(paths: dict | None = None) -> list[str]:
    """已存在的折权重（``checkpoints/g4_fold*/best.pth``，按折号排序）。

    用途：**官方验证集**评估的默认集成成员 —— 验证集与训练集无交集，
    无需"留一折"排除任何一折（排除反而白少用一个模型）。
    """
    import glob
    p = paths or load_paths()
    pattern = os.path.join(resolve(p.get("checkpoints_dir", "checkpoints")),
                           "g4_fold*", "best.pth")

    def _num(fp: str) -> int:
        m = re.search(r"g4_fold(\d+)", fp)
        return int(m.group(1)) if m else 1 << 30

    return sorted(glob.glob(pattern), key=_num)


def load_paths() -> dict:
    """路径配置 + 环境变量覆盖 + 平台合规兜底。"""
    p = load_config("paths.yaml")

    ws = os.environ.get("WORKSPACE") or p.get("workspace") or WORKSPACE
    p["workspace"] = ws

    # 数据根：DATASET_ROOT 优先（云桌面里换路径最省事）
    if os.environ.get("DATASET_ROOT"):
        p.setdefault("raw", {})["track4"] = os.environ["DATASET_ROOT"]
    # 验证集数据根：VAL_ROOT 优先（同 DATASET_ROOT 的道理）
    if os.environ.get("VAL_ROOT"):
        p.setdefault("raw", {})["val"] = os.environ["VAL_ROOT"]

    # 训练日志：规范要求写在 {workspace}/logs；平台目录存在时强制切换
    if os.environ.get("LOGS_DIR"):
        p["logs_dir"] = os.environ["LOGS_DIR"]
    elif os.path.isdir(os.path.join(ws, "logs")) or os.path.isdir(ws):
        p["logs_dir"] = os.path.join(ws, "logs")

    # 答案目录：默认 {workspace}/answer（不可写时由 writer 回退到项目内 answer/）
    if os.environ.get("ANSWER_ROOT"):
        p["answer_root"] = os.environ["ANSWER_ROOT"]
    elif os.path.isdir(os.path.join(ws, "answer")):
        p["answer_root"] = os.path.join(ws, "answer")

    for k, env in (("uif_root", "UIF_DIR"), ("manifest", "MANIFEST"),
                   ("manifest_val", "VAL_MANIFEST"),
                   ("folds", "FOLDS"), ("checkpoints_dir", "CKPT_DIR"),
                   ("preprocessed_root", "PREPROCESSED_ROOT"),
                   ("preprocess_cache", "CACHE_DIR")):
        if os.environ.get(env):
            p[k] = os.environ[env]
    return p
