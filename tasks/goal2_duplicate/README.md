# Goal2 duplicate（目标二：重复影像识别）

按《赛道四·自建模型组：系统架构与协作规范》§5 建立的插件包，产出比赛文件
`duplicate_pairs.jsonl`（`StudyUID` / `StudyUID_dup` / `PairProb`）。
赛方定义：重复影像 = 「在原图基础上**轻微调整**且与原临床无差异的复制影像」；
评分口径是**图像对级 AUC-PR**，每例最多提交 Top-200 个候选对。

> 增量包应用：本包与 Goal1 共用 `tasks/real_pipeline.py`，**必须最后解压**
> （它同时包含 Goal1+Goal2 的接线）。顺序搞错会静默绑到 Dummy，症状与修法见
> `../goal2_stitched/APPLY_NOTES.md`。

## 输入约定

- 输入是 `PipelineContext.study`（当前检查）与 `Series.source_path`（用于反推数据集根目录）；
- **先扫全库**：处理第一个检查时遍历整个测试集，每个体数据只抽**中间 K 层**建索引，
  因此不需要额外标注；`GOAL2_DUPLICATE_SCAN=0` 时退化为「只与已见过的检查增量比对」；
- 金标准（训练数据）在 `annotation/duplicate/` 下，每行一对检查号 `src_img, desc_img`，
  顺序无关；未出现在提交文件里的 pair 在评测时按 `PairProb = 0` 处理。

## 算法

三级处理，逐级提高精度、降低代价：

1. **精确指纹**：中间层像素 SHA1 相同 → `PairProb = 1.0`（完全一致的拷贝不会漏）；
2. **粗检索**：中间 K 层 → 鲁棒裁剪（0.5/99.5 百分位）→ 盒式池化到 12×12 → 零均值单位范数，
   用余弦相似度在全库取每例 Top 候选（矩阵乘法，亚二次）；
3. **精排**：候选对在 64×64 上做**带 ±1 平移搜索的归一化互相关**
   （强度图与梯度图取较大者，层序正反都比），对加噪、亮度/偏置场、轻微重采样、
   单体素平移这类「轻微调整」不敏感；
4. **概率**：`PairProb = sigmoid((similarity - center) / scale)`，单调有界；
   概率达到 `GOAL2_DUPLICATE_GATE_PROB` 才判定为重复并对下游上闸。

## 配置

| 环境变量 | 默认 | 说明 |
| --- | --- | --- |
| `GOAL2_DUPLICATE` | `1` | 是否启用重复检测 |
| `GOAL2_DUPLICATE_MODE` | `near` | `near` 近似重复检索 / `exact` 只认逐像素一致 |
| `GOAL2_DUPLICATE_SCAN` | `1` | `1` 先扫全库 / `0` 增量比对 |
| `GOAL2_DUPLICATE_SCAN_MAX_VOLUMES` | `0` | 全库扫描条数上限（0 = 不限），超大测试集可用来限时 |
| `GOAL2_DUPLICATE_SCAN_WORKERS` | `4` | 并行读盘线程数（只影响速度） |
| `GOAL2_DUPLICATE_SIM_CENTER` | `0.90` | 相似度→概率中心（**必须用金标准标定**） |
| `GOAL2_DUPLICATE_SIM_SCALE` | `0.02` | 概率映射斜率 |
| `GOAL2_DUPLICATE_MIN_SIM` | `0.50` | 精排相似度下限（低于不上报） |
| `GOAL2_DUPLICATE_COARSE_FLOOR` | `0.80` | 粗检索余弦下限（低于不精排） |
| `GOAL2_DUPLICATE_TOP_K` | `20` | 每个检查最多上报的候选对数 |
| `GOAL2_DUPLICATE_GATE_PROB` | `0.5` | 达到该概率才判定为重复并上闸 |
| `GOAL2_DUPLICATE_GATE` | `1` | 命中重复后是否对下游任务上闸 |
| `GOAL2_DUPLICATE_REPORT_MIN_PROB` | `0.0` | 低于该概率不写进提交文件（只影响稀疏度） |
| `GOAL2_DUPLICATE_MAX_PAIRS` | `200` | 每例最多提交的候选对数（赛方稀疏性约束） |
| `GOAL2_DUPLICATE_SLICES` / `_COARSE_GRID` / `_FINE_GRID` / `_SHIFT` | `3/12/64/1` | 描述子几何（一般不改） |

配置取值优先级：显式构造参数 → `GOAL2_DUPLICATE_*` 环境变量 →
**规范路径** `<checkpoint_root>/goal2_duplicate/calibration.json` → 内置默认值。
`checkpoint_root` 的解析顺序与 Goal1 的权重路径一致（见 `checkpoint.py`，
复用 `tasks/goal2_stitched/checkpoint.py` 的实现）。

## 标定（上线前必做）

```bash
cd <仓库根目录>

# 1) 金标准体检：确认能找到 annotation/duplicate 下的金标准文件
python -m tasks.goal2_duplicate.dataset --data-root /2026aicompetition/datasets/training

# 2) 图像对级标定（正类 = 金标准对；负类 = 随机采样对 + 检索命中的难负类）
python -m tasks.goal2_duplicate.evaluate \
  --data-root /2026aicompetition/datasets/training \
  --out-dir /2026aicompetition/workspace/Glioma_task12_runs \
  --target-fpr 0.10 --workers 4

# 3) 标定结果会自动写到规范路径（服务启动时直接读取，不必再 export 环境变量）：
#    /2026aicompetition/workspace/checkpoint/goal2_duplicate/calibration.json
#    想临时覆盖时再 export GOAL2_DUPLICATE_SIM_CENTER=<相似度中心>
```

`duplicate_evaluation.json` 给出：金标准召回、图像对级 AP / ROC-AUC / Recall@10%FPR /
Precision@15%Recall、正负相似度分布，以及「只用随机负类」的对照指标（后者会高估，
以含难负类的那套为准）。

### 在真实测试集上跑检索（无金标准）

```bash
# 输出与提交同格式的候选对，抽查「有多少对、最高概率多少、有没有明显误配」
python -m tasks.goal2_duplicate.evaluate \
  --dataset /data/testset --out-dir /tmp/goal2_pairs --workers 8
# → /tmp/goal2_pairs/duplicate_pairs_offline.jsonl（StudyUID / StudyUID_dup / PairProb）
```

## 稀疏性、排序与幂等

- 同一无序对只保留最高 `PairProb`；`(A, B)` 与 `(B, A)` 视为同一对；
- 每例检查最多参与 200 对，按概率降序截取（插件侧 `postprocess.normalize_pairs`
  与管线 Writer 都会做一遍，双保险）；
- 输出顺序确定（概率降序 → 检查号），重复运行同一次 evaluation 结果可复现。

## 失败语义

| 情况 | 行为 |
| --- | --- |
| 全库扫描失败（权限/坏文件） | 记 warning，退回增量比对，不终止 evaluation |
| 扫描时个别文件读不出 | 跳过并计数，写入 `goal2.duplicate.scan.errors` |
| 当前检查抽描述子失败 | 记 warning，该检查按「未命中」处理 |
| 没有任何重复 | `DuplicateResult(pairs=())`，由 Writer 写一行 `PairProb = 0` 的占位（格式要求至少一行） |

## 资源与性能

- 描述子：约 0.5 KB（粗）+ 12 KB（精，float16）/卷；单对精排约 2 ms；
- 全库扫描是主要成本（每个体数据只读中间 3 层，`.nii.gz` 需解压），
  实测合成数据约 30~50 ms/卷（单线程）；用 `GOAL2_DUPLICATE_SCAN_WORKERS` 并行读盘；
- 内存：只保留描述子，不长期持有影像或 GPU tensor。

## 已知限制

- `GOAL2_DUPLICATE_SIM_CENTER` 默认 0.90 只是占位，**必须标定**；
- 覆盖「轻微调整」：加噪、亮度/对比度、偏置场、轻微重采样、单体素平移、层序反转；
  假设副本与原件**视野与几何基本一致**。若标定显示召回不足，按顺序调
  `_COARSE_GRID`/`_FINE_GRID`（更细）→ `_COARSE_FLOOR`（放宽）→ `_TOP_K`（更多候选）
  后重新标定；
- 标定用的负类是**采样**得到的，正式指标以赛方全量 pair 口径为准。

## 研发侧（不进比赛运行链路）

```bash
python -m tasks.goal2_duplicate.dataset --data-root /2026aicompetition/datasets/training
python -m tasks.goal2_duplicate.evaluate --data-root <数据根> --out-dir <产物目录> --target-fpr 0.10
python -m tasks.goal2_duplicate.evaluate --dataset <测试集> --out-dir <产物目录>
# 没有测试集时，从训练集切一个小样本（测试集布局）再跑：
python -m tasks.goal2_stitched.dataset --data-root <训练集> --make-subset 8 --target-dir /tmp/subset
```

契约测试见 `tests/contracts/test_goal2_contract.py`；一键验证见 `verify.sh`
（五阶段：环境 → 配置 → 契约测试 → 图像对级标定/测试集检索 → 服务级 Mock Competition，
支持 `--data-root`、`--dataset`、`--make-subset N`、`--skip-service`、`--checkpoint-root`）。

全部上线命令（准备 → 代码更新与版本校验 → 拼接/重复标定 → 离线自检 → 起服务 →
环境变量速查 → 产物位置 → 常见报错处置）见 `../goal2_stitched/运行命令汇总.txt`。
