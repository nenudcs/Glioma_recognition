"""{'mod': 'goal1_authenticity', 'title': 'Goal1 影像真实性识别'} 的训练入口（[研发] 文件，是否合入由 Goal 负责人决定）。

本工程是一个多任务共享骨干，**一次训练同时优化所有头**；
因此这里只是统一训练引擎的薄封装，真正的实现位于
``tasks._common.training``（引擎与损失）与算法工程的 ``glioma_track4``（数据管线）。

用法::

    python -m tasks.goal1_authenticity.train --fold 0 --tag goal1_authenticity_fold0
    GLIOMA_TRACK4_ROOT=/path/to/glioma_track4 python -m tasks.goal1_authenticity.train --fold 0
"""
from __future__ import annotations

from tasks._common.training.cli import run_goal

GOAL = "goal1_authenticity"

if __name__ == "__main__":
    raise SystemExit(run_goal(GOAL))
