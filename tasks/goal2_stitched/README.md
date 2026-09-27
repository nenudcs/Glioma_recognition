# Goal2 stitched（目标二：拼接影像识别）

按《赛道四·自建模型组：系统架构与协作规范》§5 建立的插件包，产出比赛字段
`IsStitchedProb`（检查级、连续概率）。**规则式检测器，无权重文件**，因此不需要
checkpoint，只需要在赛方数据上标定一个判定阈值。

> 增量包应用：`tasks/real_pipeline.py` 由所有真实插件共用，**Goal2 包必须最后解压**
> （它同时包含 Goal1+Goal2 的接线）。顺序搞错会静默绑到 Dummy，症状与修法见
> `APPLY_NOTES.md`。

## 输入约定

- 输入是 `PipelineContext.study`（`data/loader.py` 已按检查号分组、按序列流式读入）；
  本插件只处理**当前检查**，不遍历数据集、不读比赛目录、不写结果文件；
- 序列选择：当前实现使用该检查的**全部序列**，逐序列打分后取最大值（一层拼接即可判定）；
- 层方向 = 最短轴（`np.argmin(shape)`），与数据 Loader 的 3-D 约定一致；
- 序列类型（`SeriesType.xlsx` → `Series.modality`）写入诊断信息 `goal2.series_types`。

## 算法

```text
d_i = mean(|I_i - (I_{i-1} + I_{i+1}) / 2|) / mean|I|     # 无量纲层间突变
score = max_i d_i                                        # 检查级分数（默认取最大）
IsStitchedProb = sigmoid(scale * (score / threshold - 1))# 分数=阈值时 0.5
```

- 默认丢弃层方向两端各 10%（`GOAL2_STITCHED_BAND`）：颈部、空气进出视野在 FOV 边缘
  本来就有大的层间变化，计入会误判；
- 可换口径：`GOAL2_STITCHED_METRIC=curvature|adjacent|local`、
  `GOAL2_STITCHED_STAT=max|p99|mean`，标定命令会同时给出 AP / 部分 AP / ROC-AUC /
  Recall@10%FPR 供比较。

## 配置

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `GOAL2_STITCHED` | `1` | 是否启用拼接检测 |
| `GOAL2_STITCHED_THRESHOLD` | `0.16` | 判定阈值，**必须用赛方标定结果替换** |
| `GOAL2_STITCHED_METRIC` | `curvature` | 单层差异口径 |
| `GOAL2_STITCHED_STAT` | `max` | 层分数汇总方式 |
| `GOAL2_STITCHED_BAND` | `0.1` | 两端丢弃比例 |
| `GOAL2_STITCHED_SCALE` | `8.0` | 概率映射斜率 |
| `GOAL2_STITCHED_GATE` | `1` | 命中拼接后是否对下游任务上闸 |

`GOAL2_GATED_FIELDS`（默认 `goal3,goal4,goal5`）控制闸门拦哪些下游任务，
置空表示完全不上闸；详见 `gating.py` 与本包 README 的「闸门」小节。

配置取值优先级：显式构造参数 → `GOAL2_STITCHED_*` 环境变量 →
**规范路径** `<checkpoint_root>/goal2_stitched/calibration.json` → 内置默认值。
`checkpoint_root` 的解析顺序与 Goal1 的权重路径完全一致（显式参数 →
`core.config` → `GOAL2_CHECKPOINT_ROOT`/`CHECKPOINT_ROOT` → `COMPETITION_WORKSPACE/checkpoint`
→ 默认 `/2026aicompetition/workspace/checkpoint`），见 `checkpoint.py`。

## 标定（上线前必做）

```bash
cd <仓库根目录>

# 1) 目录体检：确认能找到 Composition 正类与正常影像负类
python -m tasks.goal2_stitched.dataset --data-root /2026aicompetition/datasets/training

# 2) 按目标假阳性率标定阈值（正类 = annotation/Composition，负类 = 正常影像）
python -m tasks.goal2_stitched.evaluate \
  --data-root /2026aicompetition/datasets/training \
  --out-dir /2026aicompetition/workspace/Glioma_task12_runs \
  --target-fpr 0.05 --workers 4

# 3) 标定结果会自动写到规范路径（服务启动时直接读取，不必再 export 环境变量）：
#    /2026aicompetition/workspace/checkpoint/goal2_stitched/calibration.json
#    想临时覆盖时再 export GOAL2_STITCHED_THRESHOLD=<阈值>
```

产物 `stitched_calibration.json` 里同时给阈值、阈值处的召回/误报率、正负分数中位数与
排序指标。若 AP 很低，换 `--metric local` 或 `--metric adjacent` 再标一次比较。

### 在任意测试集上离线打分

```bash
# 逐序列输出分数与 IsStitchedProb（抽查「正常影像会不会被判成拼接」）
python -m tasks.goal2_stitched.evaluate \
  --dataset /data/testset --out-dir /tmp/goal2_scores --workers 8
# → /tmp/goal2_scores/stitched_scores.jsonl
```

## 闸门

命中拼接（或重复）时，`Goal2StitchedTask` 写 `context.diagnostics["goal2_gate"]`；
`tasks/real_pipeline.py` 用 `GatedStudyTask` 包装下游任务，看到闸门就返回中性结果、
**不执行模型**，被跳过的任务记在 `context.diagnostics["goal2_skipped_tasks"]`。

## 失败语义

| 情况 | 行为 |
| --- | --- |
| 单条序列打不了分 | 记 `context.warnings`，继续其它序列（可降级） |
| 全部序列都打不了分 | 分数 0.0（= 无证据）+ warning，不抛异常（不破坏输出格式） |
| 拼接检测关闭 | 概率 0.0，不写闸门 |

## 资源与性能

- 纯 numpy 计算，无 GPU、无权重加载；
- 单检查耗时：本机 CPU 上 10~40 ms/检查（24×64×64 合成数据实测 13 ms）；
  真实数据（约 160 层 × 256²）在数十毫秒量级，取决于读盘（比赛链路读的是内存里的 `Series.image`）；
- 内存：只处理当前检查，不缓存其它病例。

## 已知限制

- 阈值与影像内容强相关（脑提取后的临床 T1 约 0.005~0.01，带颈部/全头视野或体模
  可到 0.4~1.5），因此默认阈值只是占位，**必须按上面的流程标定**；
- 「跨部位套用」的 2D 拼接若落在非层方向，`curvature` 口径可能不敏感，需换 `local` 口径；
- 闸门只影响「是否继续跑下游任务」，误判为拼接会让下游字段变成中性值；
  若某项任务对中性结果敏感，从 `GOAL2_GATED_FIELDS` 里去掉对应字段。

## 研发侧（不进比赛运行链路）

```bash
python -m tasks.goal2_stitched.dataset --data-root /2026aicompetition/datasets/training
python -m tasks.goal2_stitched.evaluate --data-root <数据根> --out-dir <产物目录> --target-fpr 0.05
python -m tasks.goal2_stitched.evaluate --dataset <测试集> --out-dir <产物目录>
# 没有测试集时，从训练集切一个小样本（测试集布局）再打分：
python -m tasks.goal2_stitched.dataset --data-root <训练集> --make-subset 8 --target-dir /tmp/subset
```

契约测试见 `tests/contracts/test_goal2_contract.py`；一键验证见 `verify.sh`
（五阶段：环境 → 配置 → 契约测试 → 标定/离线打分 → 服务级 Mock Competition，
支持 `--data-root`、`--dataset`、`--make-subset N`、`--skip-service`、`--checkpoint-root`）。

全部上线命令见 **`运行命令汇总.txt`**（与任务一的《运行命令汇总》同格式：准备 →
代码更新与版本校验 → 拼接/重复标定 → 离线自检 → 起服务 → 环境变量速查 → 产物位置 →
常见报错处置）。
