"""目标二（Goal2：重复影像识别）插件包。

按《赛道四·自建模型组：系统架构与协作规范》§5 的成熟目录结构建立：

```text
tasks/goal2_duplicate/
├── config.py        # [运行] 检索/判定/稀疏性参数（中心可来自规范路径标定文件）
├── checkpoint.py    # [运行] calibration.json 规范路径（复用 goal2_stitched 的解析）
├── preprocess.py    # [运行] 中间层 slab → 粗/精两级描述子
├── model.py         # [运行] 相似度模型（平移搜索的归一化互相关）
├── retrieval.py     # [运行] 全库索引、候选检索、精排、Top-K
├── postprocess.py   # [运行] 相似度 → PairProb、成对结果规整（去重/自配对/Top-200）
├── inference.py     # [运行] 从 Study 抽描述子 + 逐例比对（Goal2 检查级探针）
├── task.py          # [运行] DatasetTask：reset/update/finalize → DuplicateResult
├── dataset.py       # [研发] 金标准解析、负类采样、图像对级指标
└── evaluate.py      # [研发] 金标准标定与自检 CLI
```

产出比赛文件 ``duplicate_pairs.jsonl``（``StudyUID`` / ``StudyUID_dup`` / ``PairProb``）。
注册入口见 ``tasks/real_pipeline.py``：重复检测的**逐例探针**由 ``Goal2StitchedTask``
携带（同一个任务链位置），**成对汇总**由 ``Goal2DuplicateRecorder`` 在数据集级完成。
"""

from __future__ import annotations

__all__ = [
    "checkpoint",
    "config",
    "dataset",
    "evaluate",
    "inference",
    "model",
    "postprocess",
    "preprocess",
    "retrieval",
    "task",
]
