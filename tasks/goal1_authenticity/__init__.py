"""目标一（Goal1：非人体/真实性判断）插件包。

按《赛道四·自建模型组：系统架构与协作规范》§5 的成熟目录结构建立：

```text
tasks/goal1_authenticity/
├── config.py               # [运行] 权重相对路径/设备/切片几何/聚合方式
├── preprocess.py           # [运行] 确定性预处理（体数据 → K 张 2.5D 切片）
├── model.py                # [运行] ConvNeXt-Tiny + FFT 频域分支
├── postprocess.py          # [运行] 序列级 → 检查级聚合
├── inference.py            # [运行] 权重加载（规范路径）与纯推理
├── task.py                 # [运行] StudyTask 适配器（比赛唯一入口）
├── checkpoint.py           # [运行] checkpoint 路径约定（§5.2/§20.1）
├── migrate_checkpoint.py   # 运维：把历史训练产物迁移到规范路径
├── dataset.py              # [研发] 赛方数据扫描/切分/切片数据集
├── augmentations.py        # [研发] 数据增强
├── losses.py               # [研发] pos_weight + BCEWithLogits
├── train.py                # [研发] 从零训练入口
└── evaluate.py             # [研发] 指标与离线批量打分
```

注册入口见 ``tasks/real_pipeline.py``（一次只替换一个插件，其余保持 Dummy）。
"""

from __future__ import annotations

__all__ = ["checkpoint", "config", "dataset", "evaluate", "inference", "migrate_checkpoint",
           "model", "postprocess", "preprocess", "task", "train"]
