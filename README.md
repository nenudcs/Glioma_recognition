# 赛道四自建模型组推理系统

本仓库实现《赛道四_自建模型组_系统架构与协作规范》的 P0 基线：可以在比赛容器中启动 HTTP 服务，异步处理整个测试集，生成并校验比赛目录，然后调用平台回调。

> 当前 Goal1～Goal5 均为 **Dummy 零分基线**。它用于验证比赛协议和工程链路，不是可提交得分的真实模型。真实模型通过冻结的 Task/Result 接口逐项替换。

## 已实现

- `GET /health` 与异步 `POST /call`，监听 `8000` 端口；
- `evaluation_id` 支持字符串和数字输入；
- 按病例流式加载 NIfTI 数据；
- `Series`、`Study`、`CompetitionDataset`、`PipelineContext`；
- `StudyTask`、`DatasetTask` 和 Goal1～Goal5 Result；
- Dummy Goal1～Goal5 及数据集级重复影像任务；
- `prediction.json`、`duplicate_pairs.jsonl`、两个二值 NIfTI 掩膜；
- JSON、JSONL、概率、Top-200、shape、affine 和二值掩膜校验；
- 临时目录写入、校验通过后发布；
- JSON Lines 推理日志；
- 有限次数 callback 重试；
- 本地运行、Mock Competition、Dockerfile 和自动化测试。

## 目录

```text
app/             HTTP 接口和 callback
core/            配置、插件加载和 EvaluationRunner
data/            NIfTI Loader 与统一数据结构
tasks/           Task/Result 契约和 Dummy 插件
pipeline/        PipelineContext、编排与聚合
output/          Writer、schema 常量与 Validator
observability/   比赛 JSONL 日志
scripts/         本地评测和 Mock Competition
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

## 接入真实模型

### 训练数据、mask 和真实后端

训练索引以 `SeriesType.xlsx` 的 `(AccessionNumber, SeriesUid)` 为全集。先在
`/2026aicompetition/datasets` 下运行：

```bash
python scripts/read_training_annotations.py \
  --series-type /2026aicompetition/datasets/training/annotation/SeriesType.xlsx \
  --annotation /2026aicompetition/datasets/training/annotation/脑胶质瘤标注结果-训练集.xlsx \
  --out-dir ./readout
python scripts/scan_training_files.py \
  --index ./readout/sample_index.csv \
  --data-root /2026aicompetition/datasets/training/annotation \
  --out-dir ./file_check
python scripts/summarize_label_overlap.py \
  --index ./readout/sample_index.csv \
  --out-dir ./label_overlap
python scripts/summarize_study_modalities.py \
  --index ./file_check/file_index.csv \
  --out ./study_modalities.json
```

`file_check/file_index.csv` 只记录路径，不把 NIfTI 像素全部读入内存。正式分类训练按
检查聚合四个固定通道；缺失模态通道用零填充，并在样本中保留
`modality_present`。`study_modalities.json` 用于先确认真实的模态组合分布。
`training.dataset.BrainNiftiDataset` 在 `__getitem__` 中懒加载一个序列；分割任务
按 `--mask-name t2_auto` 使用“水肿 → 全肿瘤 → 瘤体”，按 `--mask-name
t1ce_auto` 使用“肿瘤瘤体 → 瘤体 → 全肿瘤”。

训练 MedicalNet 的最小入口为：

```bash
python -m training.train_medicalnet \
  --file-index ./file_check/file_index.csv \
  --labels-csv ./readout/series_merged.csv \
  --label-column check__glioma_with_label__std \
  --in-channels 1 --epochs 1 \
  --output ./checkpoint/goal3_medicalnet.pt
```

训练后的比赛服务可以通过 `tasks.real_medical_pipeline:build_pipeline` 接入。Goal3/4
的 `medicalnet` 后端和 Goal5 的 `nnunet` 后端都是可选的，默认仍保留 dummy/小模型，
避免没有外部权重时服务启动就失败：

分割训练使用 nnUNet v2 的标准数据集转换、规划和训练命令；本仓库的
`training.dataset.BrainNiftiDataset` 用于快速检查/抽样读取，最终 nnUNet 训练仍需要
按 nnUNet 的 `imagesTr`、`labelsTr` 命名规范导出。mask 文件选择规则与训练前检查一致，
不要把没有 mask 的 646 条样本混入分割训练。

可直接导出一个 nnUNet 数据集（默认使用 `t2_auto` mask）：

```bash
python scripts/export_nnunet_dataset.py \
  --index ./file_check/file_index.csv \
  --labels-csv ./readout/series_merged.csv \
  --out-dir ./nnunet_dataset \
  --target abnormal --link
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity
nnUNetv2_train 501 3d_fullres 0 --npz
```

```bash
export COMPETITION_PIPELINE_FACTORY=tasks.real_medical_pipeline:build_pipeline
export GOAL3_BACKEND=medicalnet
export GOAL3_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal3_medicalnet.pt
export GOAL4_BACKEND=medicalnet
export GOAL4_CHECKPOINT=/2026aicompetition/workspace/checkpoint/goal4_medicalnet.pt
export GOAL5_BACKEND=nnunet
export GOAL5_NNUNET_MODEL_FOLDER=/2026aicompetition/workspace/checkpoint/nnunet
```

MedicalNet 源码和 nnUNet 权重必须提前放入镜像或挂载目录，服务启动时不会联网下载。

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

例如创建 `tasks/real_pipeline.py`：

```python
from pipeline.inference import InferencePipeline, StudyTaskBinding
from tasks.dummy.dataset_tasks import DummyDuplicateTask
from tasks.dummy.study_tasks import (
    DummyGoal1Task,
    DummyGoal4Task,
    DummyGoal5Task,
    DummyStitchedTask,
)
from tasks.goal3.task import RealGoal3Task


def build_pipeline():
    return InferencePipeline(
        study_tasks=(
            StudyTaskBinding("goal1", DummyGoal1Task()),
            StudyTaskBinding("goal2_stitched", DummyStitchedTask()),
            StudyTaskBinding("goal3", RealGoal3Task()),
            StudyTaskBinding("goal5", DummyGoal5Task()),
            StudyTaskBinding("goal4", DummyGoal4Task()),
        ),
        duplicate_task=DummyDuplicateTask(),
    )
```

然后配置：

```bash
export COMPETITION_PIPELINE_FACTORY='tasks.real_pipeline:build_pipeline'
```

服务启动时会调用工厂并执行每个 Task 的 `load_model()`。这样可以逐项替换 Dummy，不修改 API、Writer 或 Validator。

## 环境变量

- `COMPETITION_WORKSPACE`：默认 `/2026aicompetition/workspace`；
- `COMPETITION_ANSWER_ROOT`：可选，默认 `${COMPETITION_WORKSPACE}/answer`；
- `COMPETITION_LOG_ROOT`：可选，默认 `${COMPETITION_WORKSPACE}/logs`；
- `COMPETITION_CALLBACK_URL`：平台提供的完整 callback URL；
- `COMPETITION_CALLBACK_TIMEOUT`：单次 callback 超时，默认 10 秒；
- `COMPETITION_CALLBACK_ATTEMPTS`：callback 尝试次数，默认 3；
- `COMPETITION_MAX_WORKERS`：后台队列 worker 数，默认 1；共享 Pipeline 的推理会串行执行；
- `COMPETITION_PIPELINE_FACTORY`：可选真实插件工厂，格式为 `module:function`。

示例见 `configs/competition.env.example`。

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
