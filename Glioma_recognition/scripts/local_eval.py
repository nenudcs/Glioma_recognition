from __future__ import annotations

import argparse
import sys
import uuid
from dataclasses import replace
from pathlib import Path

# 以脚本方式运行时 ``sys.path[0]`` 是 ``scripts/``，拿不到仓根的 ``core`` 包
# （README 里那条 ``python scripts/local_eval.py ...`` 因此必然 ModuleNotFoundError）。
# 与 ``scripts/validate_output.py`` 用同一写法把仓根补进搜索路径。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import Settings                       # noqa: E402
from core.runner import EvaluationJob, EvaluationRunner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the competition pipeline locally")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-id", default="local-evaluation")
    args = parser.parse_args()

    output_root = args.output.resolve()
    # 以 ``Settings.from_env()`` 为基底，再覆盖本地路径。
    #
    # 早期实现是直接 ``Settings(...)`` 构造，于是 ``COMPETITION_PIPELINE_FACTORY``
    # 与 ``COMPETITION_CHECKPOINT_ROOT`` **被静默忽略** —— 本地预演只能跑 Dummy
    # 基线，却看起来"跑通了"，与容器里的真实插件行为完全不是一条路径。
    # 现在本地与容器共享同一套环境变量解析，只是目录改由 CLI 指定。
    settings = replace(
        Settings.from_env(),
        workspace=output_root.parent,
        answer_root=output_root,
        log_root=output_root.parent / "logs",
        callback_url=None,
    )
    runner = EvaluationRunner(settings)
    result = runner.run(
        EvaluationJob(
            request_id=f"local-{uuid.uuid4()}",
            evaluation_id=str(args.evaluation_id),
            dataset_path=args.dataset.resolve(),
        ),
        send_callback=False,
    )
    print(result)


if __name__ == "__main__":
    main()

