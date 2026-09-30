"""赛道四自建模型组：``glioma_track4`` 算法实现的接入点。

本包**只做路径注入与转发**，算法实现位于 ``glioma_track4`` 工程
（训练 + 推理 + 桥接层，``integration/`` + ``src/``）。

## 自包含（默认）

算法实现已**内置**在本仓 ``vendor/glioma_track4/``（``integration/`` + ``src/`` +
``configs/``，见 ``vendor/README.md``），所以本仓 **clone 下来即可运行**，
不需要另外准备外部算法工程。查找顺序（第一个命中者生效）::

    1. $GLIOMA_TRACK4_ROOT                显式覆盖（指向独立演进的算法工程）
    2. <仓根>/vendor/glioma_track4        本仓内置副本 ← 默认
    3. <仓根>/../glioma_track4            同级目录（开发期，两工程并列）
    4. /2026aicompetition/workspace/glioma_track4   平台约定目录

为什么内置副本排在"同级目录"**前面**：内置副本与本仓的提交协议**同版本**，
而同级目录可能是任意时间点的另一份代码。默认走内置，行为才可复现。

注意内置副本**只覆盖推理链路**（``integration/`` + ``src/`` 中被它引用的模块）。
训练入口（``scripts/``、``glioma_track4`` 的顶层脚本）不在本仓 ——
需要训练时请用独立的算法工程，并通过 $GLIOMA_TRACK4_ROOT 指过去。

## 接口

- 比赛协议、Writer/Validator/Aggregator 仍由本仓库单点维护；
- 算法产物（checkpoint）按规范放在
  ``/2026aicompetition/workspace/checkpoint/<goal>/``。

接入方式（见 `configs/competition.env.example`）::

    COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline

> 另一条不依赖算法工程的路：``tasks.real_pipeline`` 直接注册本仓
> ``tasks/goalX`` 插件（自包含，见其模块 docstring）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_track4_on_path() -> Path | None:
    """把算法实现所在目录加入 ``sys.path``；返回其根目录（找不到返回 None）。

    内置副本排在外部工程之前，理由见模块 docstring。
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[2]                                   # tasks/glioma/ → 仓根
    candidates = [
        os.environ.get("GLIOMA_TRACK4_ROOT"),
        repo_root / "vendor" / "glioma_track4",                   # 本仓内置（自包含）
        here.parents[3] / "glioma_track4",                        # 同级目录（开发期）
        Path("/2026aicompetition/workspace/glioma_track4"),       # 平台约定目录
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "integration" / "factory.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    return None


TRACK4_ROOT = _ensure_track4_on_path()

__all__ = ["TRACK4_ROOT"]
