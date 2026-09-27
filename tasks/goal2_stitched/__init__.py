"""目标二（Goal2：拼接影像识别）插件包。

按《赛道四·自建模型组：系统架构与协作规范》§5 的成熟目录结构建立：

```text
tasks/goal2_stitched/
├── config.py        # [运行] 判定阈值/口径/概率映射（阈值可来自规范路径标定文件）
├── checkpoint.py    # [运行] checkpoint 根目录与 calibration.json 规范路径
├── preprocess.py    # [运行] 确定性预处理（层方向、亮度尺度）
├── model.py         # [运行] 规则模型：体数据 → 逐层拼接分数
├── postprocess.py   # [运行] 序列级 → 检查级聚合、分数 → 概率
├── inference.py     # [运行] 纯推理：一个 Study → 拼接分数/概率
├── task.py          # [运行] StudyTask 适配器（比赛唯一入口）
├── gating.py        # [运行] 下游任务闸门（命中拼接/重复后不再向下执行）
├── dataset.py       # [研发] 赛方数据目录扫描（标定用）
└── evaluate.py      # [研发] 阈值标定与自检 CLI
```

产出比赛字段 ``IsStitchedProb``（检查级、连续概率，见 ``StitchedResult``）。
注册入口见 ``tasks/real_pipeline.py``（一次只替换一个插件，其余保持 Dummy）。
"""

from __future__ import annotations

__all__ = [
    "checkpoint",
    "config",
    "dataset",
    "evaluate",
    "gating",
    "inference",
    "model",
    "postprocess",
    "preprocess",
    "task",
]
