"""Goal5 分割与可解释性 的离线评估入口（[研发] 文件）。

评估口径与训练时的验证**完全一致**（复用同一个 ``build_val_fn``），
避免"训练时报一个数、评估时报另一个数"。

两种评估的定位：

    python -m tasks.goal5_segmentation.evaluate --fold 0    # 轻量、patch 级，快速自查
    bash glioma_track4/scripts/16_finalize.sh   # 权威：全图 / 留一折集成 / OOF

面向比赛指标的数字请以 ``16_finalize.sh`` 为准。
"""
from __future__ import annotations

from tasks._common.training.cli import run_eval

GOAL = "goal5_segmentation"


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：评估本 Goal 在指定折验证集上的指标。"""
    return run_eval(GOAL, argv)


if __name__ == "__main__":
    raise SystemExit(main())
