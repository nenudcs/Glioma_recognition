"""共享训练工程（多任务单骨干）。

对外入口：
    from tasks._common.training.engine import TrainConfig, train
    from tasks._common.training.losses import LossWeights, compute_losses
"""
from __future__ import annotations

from tasks._common.training.engine import TrainConfig, train, write_history
from tasks._common.training.losses import LossWeights, compute_losses, summarize

__all__ = ["TrainConfig", "LossWeights", "compute_losses", "summarize", "train",
           "write_history"]
