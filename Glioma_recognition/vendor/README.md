# vendor/ —— 内置的算法实现副本（自包含）

本目录让 **本仓 clone 下来即可运行**，不需要另外准备独立的算法工程。

## 内容

| 路径 | 说明 |
|---|---|
| `glioma_track4/integration/` | 与团队串联工程的**桥接层**（`factory` / `tasks` / `common` / `export_ckpt`） |
| `glioma_track4/src/` | 桥接层引用的**推理链路**（`inference/` `data/` `models/` `utils/` 等） |
| `glioma_track4/configs/` | `load_config()` 读取的配置（`preprocess.yaml` `labels.yaml` `paths.yaml` `train.yaml` 等） |
| `glioma_track4/data/modality_model.json` | 体素判别模型的**外部系数**（1.8 KB）。`start.sh` 会自动挂载它；缺失时 `data/voxel_modality.py` 退回**内嵌系数**（实测一致率 ≈27%，低于随机），所以内置一份 |

来源：独立的算法工程 `glioma_track4`（训练 + 推理 + 桥接）。

> `start.sh` 与 `scripts/{audit_goal5,pre_submit_check}.py` 的**默认路径**都优先指向本目录，
> 找不到才回退平台上的算法工程路径 —— 所以干净 clone 下这些脚本也能直接跑。

## 为什么内置

`tasks/glioma/pipeline.py` 是赛道四自建模型组的接入点，其实现原本位于同级的
独立算法工程。为让提交仓库**自包含**（评审 / 队友 clone 后无需额外准备），
把**推理链路**复制一份到这里。

查找顺序在 `tasks/glioma/__init__.py::_ensure_track4_on_path` 中定义，内置副本
**排在外部工程之前** —— 内置副本与本仓的提交协议同版本，默认走它行为才可复现；
需要独立演进的算法工程时，用 `GLIOMA_TRACK4_ROOT` 显式覆盖即可。

## 边界（重要）

- 内置副本**只覆盖推理链路**。训练入口（`glioma_track4/scripts/`、顶层脚本）
  **不在本仓** —— 需要训练时用独立算法工程，并 `export GLIOMA_TRACK4_ROOT=<算法工程根>`。
- 副本里的 `src/utils/config.py` 以**自身位置**推导 `PROJECT_ROOT`，因此
  `PROJECT_ROOT` = `vendor/glioma_track4/`、`CONFIG_DIR` = `vendor/glioma_track4/configs/`
  —— 这正是选这个目录层级的原因：**副本代码一行都不用改**。
- 副本内的 `from src.xxx` / `from integration.xxx` 是绝对导入，靠
  `_ensure_track4_on_path()` 把 `vendor/glioma_track4/` 注入 `sys.path` 来解析。
  因此**不要**把 `src/` 或 `integration/` 单独移走。

## 同步方式

内置副本是**快照**，不会自动跟随算法工程更新。算法侧改了推理链路后，按下表同步：

```bash
# 设成算法工程根（按实际路径改）
TRACK4=/path/to/glioma_track4
DST=$(git rev-parse --show-toplevel)/vendor/glioma_track4   # 或本目录的绝对路径

rm -rf "$DST/src" "$DST/integration" "$DST/configs" "$DST/data"
cp -r "$TRACK4/src" "$TRACK4/integration" "$DST/"
mkdir -p "$DST/configs" "$DST/data"
cp "$TRACK4/configs/"*.yaml "$DST/configs/"
cp "$TRACK4/data/modality_model.json" "$DST/data/"
find "$DST" -name __pycache__ -type d -exec rm -rf {} +
```

同步后务必跑一遍自检（见下），确认没有引入新的外部依赖。

## 自检

```bash
# 1) 副本能否被解析到（应打印 vendor/glioma_track4 的绝对路径，而非外部工程）
python3 -c "from tasks.glioma import TRACK4_ROOT; print(TRACK4_ROOT)"

# 2) 桥接层能否导入
python3 -c "import integration.factory; print('ok')"

# 3) 真实插件工厂能否构建（需要 torch / 权重；无权重时应在加载权重处报错，而非 ImportError）
COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline \
  python3 -c "from core.registry import build_pipeline; build_pipeline()"
```

第 1 步若打印出**仓库外**的路径，说明 `$GLIOMA_TRACK4_ROOT` 被设置了 ——
那是刻意的覆盖行为，不是故障。
