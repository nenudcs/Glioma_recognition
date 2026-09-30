# 赛道四自建模型组推理系统

本工程实现《赛道四_自建模型组_系统架构与协作规范》的 P0 基线：可以在比赛容器中启动 HTTP 服务，异步处理整个测试集，生成并校验比赛目录，然后调用平台回调。

> **位置**：本目录是团队仓库里的**自包含子工程**（`Glioma_recognition/`），与仓库根的
> Dummy 基线并存。克隆整个团队仓库后，进入本目录即可直接使用；也可把本目录单独
> 拷出使用（无任何仓库外依赖）。

**本工程自包含**：算法实现内置在 `vendor/glioma_track4/`（见 [`vendor/README.md`](vendor/README.md)），
装好依赖即可运行，**不需要**另外准备独立的算法工程。

真实插件位于 `tasks/goalX/`，两种接入方式（都用 `COMPETITION_PIPELINE_FACTORY` 指定）：

| 工厂 | 说明 |
|---|---|
| `tasks.real_pipeline:build_pipeline` | 直接注册本工程 `tasks/goalX` 插件，**不经过**算法工程 —— `start.sh` 的默认值 |
| `tasks.glioma.pipeline:build_pipeline` | 走内置的算法实现（`vendor/glioma_track4/` 的桥接层）。需要算法侧的权重与训练口径时才用 |

`GLIOMA_GOALS` 控制启用哪些真实插件，未启用的用 Dummy 补位（规范 §15.1「每次只替换一个插件」）。

## 已实现

- `GET /health` 与异步 `POST /call`，监听 `8000` 端口；
- `evaluation_id` 支持字符串和数字输入；
- 按病例流式加载 NIfTI 数据；
- `Series`、`Study`、`CompetitionDataset`、`PipelineContext`；
- `StudyTask`、`DatasetTask` 和 Goal1～Goal5 Result；
- **真实 Goal 插件**（`tasks/goal1_authenticity` / `goal2_stitched` / `goal3_tumor` /
  `goal5_segmentation` / `goal4_diagnosis` / `goal2_duplicate`），共享一个多任务骨干，
  一次前向产出六路结果；
- Dummy 插件（`tasks/dummy/`）用于未启用插件的补位；
- **模态识别三层兜底**：关键词 → 官方 `SeriesType.xlsx` → **体素统计判别**
  （`data/voxel_modality.py`，不依赖表里的键与磁盘一致）；
- `prediction.json`、`duplicate_pairs.jsonl`、两个二值 NIfTI 掩膜；
- JSON、JSONL、概率、Top-200、shape、affine 和二值掩膜校验；
- 临时目录写入、校验通过后发布；**逐例容错**（单例失败只丢该例，不作废整批）；
- JSON Lines 推理日志；
- 有限次数 callback 重试；
- 本地运行、Mock Competition、Dockerfile 和自动化测试。

## 目录

```text
app/             HTTP 接口和 callback
core/            配置、插件加载和 EvaluationRunner
data/            NIfTI Loader、体素模态判别与统一数据结构
tasks/           Task/Result 契约
  ├── goalX/     真实插件（真实性 / 拼接 / 肿瘤 / 分割 / 结构化 / 重复）
  ├── dummy/     未启用插件的零分补位
  ├── real_pipeline.py    ← 直接注册本工程 goalX 的工厂（默认）
  └── glioma/     ← 走内置算法实现的桥接工厂
vendor/          内置的算法实现（自包含用；见 vendor/README.md）
pipeline/        PipelineContext、编排与聚合
output/          Writer、schema 常量与 Validator
observability/   比赛 JSONL 日志
configs/         插件与平台配置示例
scripts/         本地评测、Mock Competition、提交前自检
tests/           契约与端到端测试
```

## 比赛容器运行

要求 Python 3.10 或更高版本。进入项目目录后安装依赖：

```bash
python -m pip install -r requirements.txt
```

配置平台回调地址：

```bash
export COMPETITION_WORKSPACE=/2026aicompetition/workspace
export COMPETITION_CALLBACK_URL='平台页面提供的完整回调地址'
```

启动长期前台进程：

```bash
chmod +x start.sh
./start.sh
```

检查服务：

```bash
curl http://127.0.0.1:8000/health
```

预期结果：

```json
{"status":"active"}
```

调用示例：

```bash
curl -X POST http://127.0.0.1:8000/call \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id":"2f4f3f2a-6320-44c2-bd49-de29bd64dc09",
    "team_id":"2054451790241292288",
    "track_code":"2eaf6e36583b4d06af0f4582220956d0",
    "input":{
      "evaluation_id":"454899866564564",
      "dataset_path":"/data/testset"
    }
  }'
```

`/call` 只进行轻量校验和入队，立即返回 HTTP 200。后台完成后输出到：

```text
/2026aicompetition/workspace/answer/{evaluation_id}/
```

日志写入：

```text
/2026aicompetition/workspace/logs/inference.jsonl
```

## 使用 Docker

```bash
docker build -t track4-inference .
docker run --rm -p 8000:8000 \
  -e COMPETITION_CALLBACK_URL='http://host.docker.internal:9000/callback' \
  -v /host/workspace:/2026aicompetition/workspace \
  -v /host/testset:/data/testset:ro \
  track4-inference
```

真实 GPU 模型可在构建时替换基础镜像：

```bash
docker build \
  --build-arg BASE_IMAGE='<比赛允许的 CUDA/PyTorch 基础镜像>' \
  -t track4-inference .
```

不要在容器启动时联网下载依赖或权重；正式镜像应提前包含全部依赖和模型文件。

## 本地 Pipeline 模式

本地模式复用与比赛服务完全相同的 Loader、Pipeline、Writer 和 Validator：

```bash
python scripts/local_eval.py \
  --dataset ./test_data \
  --output ./answer \
  --evaluation-id local-001
```

## 测试

契约和端到端测试：

```bash
python -m unittest discover -s tests -v
```

完整模拟平台请求、5 秒响应限制、后台推理、输出和 callback：

```bash
python scripts/mock_competition.py
```

成功时应看到：

```text
PASS Health
PASS Call response time
PASS Background execution
PASS Output and NIfTI validation
PASS Callback
```

## 提交前自检

**一条命令、退出码即结论**（`0` = GO，`1` = NO-GO）：

```bash
python3 scripts/pre_submit_check.py --limit 20     # 冒烟 20 例，约 1 分钟
python3 scripts/pre_submit_check.py --full         # 完整跑（慢，提交前最后一遍）
```

关卡：`G1 环境` → `G2 权重`（**`seg` 头是否真的加载上**）→ `G3 预处理`
（本仓两份预处理与训练侧逐项对齐）→ `G4 冒烟`（空掩膜率 / `pmax`）→
`G5 端到端`（`--full`）→ `G6 内存`（常驻模型总数）→ `G7 后处理`（真实体积下的峰值）。

设计上**任何一关抛异常都不中断**，只记 FAIL 继续跑 —— 一次看到全部问题，
而不是修一个跑一次。

按需使用的单项诊断脚本：

| 脚本 | 用途 |
|---|---|
| `scripts/audit_goal5.py` | 权重真伪 + 预处理一致性（只读，不跑推理） |
| `scripts/probe_goal5.py` | **不重启服务**，直接跑几例看 Goal5 诊断（空掩膜成因） |
| `scripts/mem_audit.py` | 量化常驻内存的放大环节并推算峰值 |
| `scripts/voxel_consistency.py` | 体素判别的准确率 + 概率饱和度 |

启动方式提醒：**用 `./start.sh`，不要直接 `python -m uvicorn`** ——
`start.sh` 会设好 `COMPETITION_PIPELINE_FACTORY`、容错开关、体素模型路径，
并打印权重解析结果。

## 接入真实模型

### 1. 实现 Task

检查级任务继承 `StudyTask`：

```python
from tasks.base import StudyTask
from tasks.results import Goal3Result


class RealGoal3Task(StudyTask[Goal3Result]):
    name = "goal3"

    def load_model(self) -> None:
        self.model = ...

    def predict(self, context):
        probability = self.model(...)
        return Goal3Result(tumor_probability=float(probability))
```

重复影像任务继承 `DatasetTask[DuplicateResult]`，通过 `reset()`、逐病例
`update(study, context)` 和 `finalize()` 完成一次 evaluation；`update()` 只应保留
embedding、哈希等轻量特征。Task 不能直接读取比赛输出目录、写
`prediction.json` 或调用 callback。

### 2. 构建真实 Pipeline

**本仓已有** `tasks/real_pipeline.py`，直接注册 `tasks/goalX` 的真实插件：

```python
from pipeline.inference import InferencePipeline, StudyTaskBinding

DEFAULT_GOALS = "goal1,goal2_stitched,goal3,goal5,goal4,goal2_duplicate"

_BUILDERS = {
    "goal1": ("tasks.goal1_authenticity.task", "Goal1Task"),
    "goal2_stitched": ("tasks.goal2_stitched.task", "StitchedTask"),
    # …goal3 / goal5 / goal4 / goal2_duplicate
}


def build_pipeline() -> InferencePipeline:
    enabled = {g.strip() for g in os.environ.get("GLIOMA_GOALS", DEFAULT_GOALS).split(",") if g.strip()}
    # 启用的用真实插件，未启用的用 tasks/dummy 补位
    ...
```

配置：

```bash
export COMPETITION_PIPELINE_FACTORY='tasks.real_pipeline:build_pipeline'
```

服务启动时会调用工厂并执行每个 Task 的 `load_model()`。启用几个插件由 `GLIOMA_GOALS`
决定，**未启用的自动用 Dummy 补位** —— 这就是规范 §15.1「每次只替换一个插件并运行
完整回归」的直接支持。启动日志会打印 `真实插件 N/6` 以及补位的 Dummy 名单，
避免"以为全开了、其实只有列出来的几个是真的"。

新增或替换一个插件时，只需在 `tasks/` 下实现 Task 并在 `_BUILDERS` 注册，
不修改 API、Writer 或 Validator。

## 环境变量

平台协议相关：

- `COMPETITION_WORKSPACE`：默认 `/2026aicompetition/workspace`；
- `COMPETITION_ANSWER_ROOT`：可选，默认 `${COMPETITION_WORKSPACE}/answer`；
- `COMPETITION_LOG_ROOT`：可选，默认 `${COMPETITION_WORKSPACE}/logs`；
- `COMPETITION_CALLBACK_URL`：平台提供的完整 callback URL；
- `COMPETITION_CALLBACK_TIMEOUT`：单次 callback 超时，默认 10 秒；
- `COMPETITION_CALLBACK_ATTEMPTS`：callback 尝试次数，默认 3；
- `COMPETITION_MAX_WORKERS`：后台队列 worker 数，默认 1；共享 Pipeline 的推理会串行执行；
- `COMPETITION_PIPELINE_FACTORY`：真实插件工厂，格式为 `module:function`。
  **不设它 = 静默跑 Dummy 基线**（服务照常起、`/health` 通，但答案是占位），
  `start.sh` 已默认设为 `tasks.real_pipeline:build_pipeline`。

插件与算法实现：

- `COMPETITION_CHECKPOINT_ROOT`：权重根，默认 `${COMPETITION_WORKSPACE}/checkpoint`；
  各 Goal 权重在其子目录（如 `goal5_segmentation/`）下取 `core.pt`，
  没有该文件时取该目录全部 `*.pt` 做多折集成；
- `GLIOMA_GOALS`：启用哪些真实插件（逗号分隔），未列出的用 Dummy 补位；
- `GLIOMA_TRACK4_ROOT`：**覆盖**内置算法实现的路径（默认用 `vendor/glioma_track4/`）。
  仅当你要用独立演进的算法工程时才设；
- `GLIOMA_DEVICE`：推理设备，默认 `cuda`；
- `GLIOMA_CKPT`：显式指定权重路径（逗号分隔，多折集成），优先级高于
  `COMPETITION_CHECKPOINT_ROOT`；
- `GLIOMA_MODALITY_MODEL`：体素判别模型 JSON。`start.sh` 已默认指向
  `vendor/glioma_track4/data/modality_model.json`，缺失时退回**内嵌系数**
  （实测一致率 ≈27%，低于三分类随机水平）。

评测健壮性（**默认值已按实测结论设好，一般不用改**）：

- `GLIOMA_LOADER_TOLERANT`：逐例容错，默认 **1（开）**。评测**不可重跑**，
  关掉它则 1 例脏数据会让整批 staging 被删 → 几百例一起 0 分；
- `GLIOMA_VOXEL_GUESS_EXCLUDED`：是否允许体素判别**覆盖**官方表里的
  `其他`/`正常`/`平扫`，默认 **0（不覆盖）**。实测判别器对这类 OOD 输入的
  置信度会饱和、整体一致率低于随机，把定位像当 T1CE 填进通道比留空更有害；
- `GLIOMA_PROGRESS`：逐例打印进度行（默认开）。一次几百例跑很久，
  没有它「进程卡死」与「某例在跑分钟级滑窗」在日志上无法区分；
- `GLIOMA_MAX_ENSEMBLE`：多折集成的**份数上限**，默认 8。超出即报错并列出
  全部文件 —— 防止权重目录里混进训练快照后被静默加载成几十个常驻模型（容器 OOM）。

示例见 `configs/competition.env.example`。

## 平台数据目录（`/2026aicompetition/datasets`）

容器内该挂载点下是**五个平行阶段目录**，各自的用途不同：

```text
/2026aicompetition/datasets/
├── training/            ← 官方训练集（含 annotation/ 与各检查号目录）
├── evaluation_first/    ← 第一轮评测输入
├── evaluation_second/   ← 第二轮评测输入
├── evaluation_finals/   ← 决赛评测输入
└── verification/        ← 验证集
```

**数据根必须精确到其中一个阶段目录**，不能停在 `datasets/`：

```bash
# 训练（算法工程 / 训练工程）
export DATASET_ROOT=/2026aicompetition/datasets/training

# 推理：--dataset / dataset_path 指向具体评测阶段
python scripts/local_eval.py \
  --dataset /2026aicompetition/datasets/evaluation_first \
  --output /2026aicompetition/workspace/answer/local-001 \
  --evaluation-id local-001
```

误传父目录会被**当场拦截**（父目录下只有一个阶段目录时自动下钻并打印告警），
不会把 `evaluation_first` 这类阶段名当成检查号后静默跑出一份对不上的答案。

Loader 对官方数据形态的容错规则：

- 顶层 `annotation/`（`fake` / `Composition` / `duplicate`）等**非病例目录一律跳过**；
- 与影像同目录的掩膜按**中英文关键词**识别并过滤（`mask/seg/label/roi` 与
  `掩码/标注/瘤体/水肿/异常/核心/病灶/肿瘤区`），与训练侧 `MASK_HINTS` 同一语义；
- 缺文件或匹配不上时沿用原有元数据，`SeriesType.xlsx` 见下方说明。

本地复现整条平台协议（`/health` → `/call` → 后台推理 → 回调 → 校验）：

```bash
COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline \
COMPETITION_CHECKPOINT_ROOT=/path/to/checkpoint \
python scripts/mock_competition.py \
  --dataset /2026aicompetition/datasets/evaluation_first \
  --workspace /2026aicompetition/workspace --timeout 3600
```

`--dataset` 省略时会现场生成一个 2 例的最小数据集；指向真实评测目录时请同时
放大 `--timeout`，否则回调等待会先于推理结束而超时。

## 当前规范解释

- 官方输入仅支持 NIfTI（.nii/.nii.gz）；
- 数据根目录可提供标准 `SeriesType.xlsx`，Loader 按 `AccessionNumber` 和 `SeriesUid` 匹配后使用 `SeriesType` 补充序列类型；缺少文件或匹配行时沿用原有元数据；
- Series UID 目录包含多个 NIfTI 时，只读取文件主名与目录名完全一致的原文件；没有唯一匹配时明确报错；
- 核心区写入其来源 T1 增强 Series UID 目录；周围区写入其 Flair/T2 Series UID 目录；
- `SegmentationMaskURI` 相对于病例目录，格式为 `./{SeriesUid}/{SeriesUid}.nii.gz`；
- 所有病例均输出 `IsNotHumanBodyProb` 和 `IsStitchedProb`；
- 二分类字段采用比赛示例中的专属 `*Probability` 名称；
- Dummy duplicate 至少为两个检查生成一个概率为 0 的合法 pair，因为规范要求 JSONL 至少一行；
- 单检查数据集无法同时满足“至少一行”和“禁止 self-pair”，因此会明确失败。

组委会确认正式 JSON Schema、枚举或 URI 后，只需更新聚合器、映射和 Validator，不需要修改模型接口。

## 已知边界

- 当前任务状态保存在单进程内存中；平台若明确要求容器重启后恢复运行，再增加持久化；
- callback 失败会有限重试并保留已经校验的结果，但没有持久化重试队列；
- Goal4 英文枚举仍以现有比赛示例为基线，正式提交前必须按评分脚本确认。
