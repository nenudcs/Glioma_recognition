# Goal1 authenticity（目标一：非人体/伪造影像识别）

按《赛道四·自建模型组：系统架构与协作规范》§5 建立的插件包，产出比赛字段
`IsNotHumanBodyProb`（检查级、连续概率）。

## 输入约定

- 输入是 `PipelineContext.study`（`data/loader.py` 已按检查号分组、按序列流式读入）；
  本插件只处理**当前检查**，不遍历数据集、不读比赛目录；
- 序列选择：当前实现使用该检查的**全部序列**，逐序列打分后按 `GOAL1_SERIES_AGGREGATION`
  聚合（默认 `max`：只要一条序列可疑就标记该检查）；
- 序列类型 `<SeriesType.xlsx>` 由 Loader 解析后写入 `Series.modality`，会记录到诊断信息里。

## 权重与配置

| 配置 | 说明 |
| --- | --- |
| 规范路径 | `<checkpoint_root>/goal1_authenticity/model.pt`（`checkpoint_root` 默认 `/2026aicompetition/workspace/checkpoint`） |
| 相对路径 | `Goal1Config.checkpoint_relative_path`，默认 `goal1_authenticity/model.pt`（规范要求各 Goal 只声明相对路径） |
| 临时覆盖 | `GOAL1_CHECKPOINT=/abs/path/model.pt`（迁移前后自测用） |
| 根目录解析顺序 | 显式参数 → `core.config.Settings.checkpoint_root`（若已支持）→ `GOAL1_CHECKPOINT_ROOT`/`CHECKPOINT_ROOT` → `COMPETITION_WORKSPACE/checkpoint` → 默认值 |
| 运行参数 | `GOAL1_DEVICE`、`GOAL1_BATCH_SLICES`、`GOAL1_SLICES_PER_CASE`、`GOAL1_IMAGE_SIZE`、`GOAL1_SERIES_AGGREGATION`、`GOAL1_STRICT` |

权重由 `migrate_checkpoint.py` 从训练产物迁移过来（附带 `model_meta.json`：SHA256、
backbone、`image_size`、`slices_per_case`、`val_ap`、epoch、数据来源）：

```bash
python -m tasks.goal1_authenticity.migrate_checkpoint --dry-run
python -m tasks.goal1_authenticity.migrate_checkpoint \
  --source /2026aicompetition/workspace/task1_runs/seed42/best.pt --also-last
```

## 失败语义（规范 §8.2 / §9.1）

| 情况 | 行为 |
| --- | --- |
| 单条序列读不出/打不了分 | 记 `context.warnings`，继续其它序列（可降级） |
| 整个 Study 没有任何可打分序列 | 抛 `MissingSeriesError` → 终止该 evaluation（不可降级） |
| 权重未加载成功 | `predict` 抛 `ModelInferenceError`；`GOAL1_STRICT=1` 时在启动阶段就失败 |

## 资源与性能

- 推理：1 个 ConvNeXt-Tiny（约 28M 参数）+ 频域分支，`batch_slices=32` 时单序列 16 切片；
- 权重加载一次（`load_model()`），单例耗时约 1~2 s（GPU）/ 5~10 s（CPU）；
- 单检查耗时：实测 CPU 上 0.2~2.7 s/检查（含 I/O），GPU 更快；
- 显存：单序列 16×3×224² 输入，批次 32 切片时约 1 GB 级别；显存紧张时调小 `GOAL1_BATCH_SLICES`。

## 已知限制

- 训练数据正类（`annotation/fake/`）规模有限时，模型对隐藏测试集的泛化仍需多种子集成验证；
- 单权重形态（`model.pt`）暂不支持多权重集成，如需集成可在 `inference.py` 内扩展；
- `core/config.py` 尚未提供 checkpoint 根目录（规范 §26 差距），当前用环境变量兜底；
- 训练侧不做 `SeriesType.xlsx` 解析（推理侧由 Loader 负责）。

## 研发侧（不进比赛运行链路）

```bash
# 数据体检
python -m tasks.goal1_authenticity.dataset --data-root /2026aicompetition/datasets/training
# 从零训练
python -m tasks.goal1_authenticity.train --epochs 40 --batch-size 4 --slices-per-case 16 --image-size 224
# 离线打分与指标
python -m tasks.goal1_authenticity.evaluate --manifest <manifest.jsonl> --split val --out-dir <dir>
```

契约测试见 `tests/contracts/test_goal1_contract.py`。

## 上线验证（五阶段）

插件自带一键验证脚本（只读代码 + 迁移权重 + 跑测试 + 起服务 mock，不改管线）：

```bash
bash tasks/goal1_authenticity/verify.sh                       # 环境 / 测试 / 端到端 mock
bash tasks/goal1_authenticity/verify.sh \
  --source /2026aicompetition/workspace/task1_runs/seed42/best.pt --also-last
bash tasks/goal1_authenticity/verify.sh --make-subset 8        # 额外做离线打分（从训练集自动切 8 例）
bash tasks/goal1_authenticity/verify.sh --dataset /data/testset # 用真实测试集做离线打分
```

脚本内部依次做：① 依赖与注册入口检查；② 权重迁移到规范路径（含 `model_meta.json`）；
③ `python -m unittest discover -s tests -t .`；④ 可选离线打分；⑤
`python -m scripts.mock_competition`（起服务 + `/call` + 回调 + 输出校验，期望 5 个 PASS）。

参数说明：`--dataset <目录>` 必须真实存在（脚本会先校验并提示）；`--make-subset N` 会从
`--subset-from`（默认 `$GOAL1_DATA_ROOT`，即 `/2026aicompetition/datasets/training`）里的
`annotation/{fake,正常病例}` 复制 N 例到临时目录，按比赛布局 `<检查号>/<序列号>/<序列号>.nii(.gz)`
摆好后再打分——**不需要真实测试集也能完成第 4 步**。
