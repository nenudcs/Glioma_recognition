#!/usr/bin/env sh
set -eu

# --------------------------------------------------------------------------- #
# 加载容错（正式评测默认开启）
# --------------------------------------------------------------------------- #
# ⚠️ ``data/loader._TOLERANT`` 是在**模块导入期**读取本变量的 —— 进程跑起来之后再改
#    环境变量**不生效**，所以必须在这里 export，而不是等到运行时。
#
# 打开后覆盖 4 处（**全部只在容错模式生效**；关闭时行为与上游完全一致）：
#   ① 序列目录有多个 NIfTI 且无"主名 == 目录名"的唯一原件 → 跳过该序列
#   ② 单个序列文件损坏 / 维度非法 → 跳过该序列
#   ③ SeriesType.xlsx 读不到 / 表头不认 / 同键冲突 → 降级为"无表继续"
#      （描述退回 sidecar/目录名，模态随后走官方表 / 体素判别兜底）
#   ④ core/runner 逐例容错：单例失败只丢该例，不再 rmtree 整个 staging
#
# 为什么默认开：评测**不可重跑**。默认 fail-fast 下，真数据里只要有 1 个脏目录或
# 1 次推理异常，``_run_streaming`` 就会把整批 staging 删掉 → **几百例一起 0 分**；
# 打开后最坏情况只是丢 1 例的 1 个通道。
#
# 想严格与上游 README §3 字面一致（fail-fast）：GLIOMA_LOADER_TOLERANT=0 ./start.sh
: "${GLIOMA_LOADER_TOLERANT:=1}"
export GLIOMA_LOADER_TOLERANT

# --------------------------------------------------------------------------- #
# 体素判别兜底：**不覆盖**官方表的「其他」（正式评测默认关闭）
# --------------------------------------------------------------------------- #
# 表里明写 `其他` / `正常` / `平扫` 是**权威排除**。允许体素判别覆盖它，风险不对称：
#   · 赌对的收益：少数"整例全被标成其他"的检查，通道被填上，不至于必然 0 分
#   · 赌错的代价：把定位像 / 非脑之类的异常序列当 T1CE 填进通道 ——
#                 **比留空通道更有害**（留空是训练时见过的"缺失"标记，错填是 OOD 输入）
# 实测（`scripts/voxel_consistency.py`）：判别器对这类输入的置信度会**饱和到 1.0**
# （次优 0.0），0.5 的门槛形同虚设；且整体一致率**低于三分类的随机水平**。
# 结论：正式评测**默认关闭**，尊重官方表的判定。
#
# 想打开（例如你重训后把一致率测到 >=80%）：GLIOMA_VOXEL_GUESS_EXCLUDED=1 ./start.sh
: "${GLIOMA_VOXEL_GUESS_EXCLUDED:=0}"
export GLIOMA_VOXEL_GUESS_EXCLUDED

# --------------------------------------------------------------------------- #
# 体素判别模型：**自动挂载**（优先本仓内置副本，其次重训后的外部模型）
# --------------------------------------------------------------------------- #
# 事实对照：
#   · 仓库**内嵌**系数 = 在**本地模拟集**上拟合的，与官方数据有域差 → 实测一致率 ≈27%，
#     **低于三分类随机水平（33%）**；
#   · 用官方训练集**重训**后（glioma_track4/scripts/31_train_modality_model.py）
#     自报 **5 折 CV = 0.735** —— 说明模型本来是有真实信号的。
#   · 而 data/voxel_modality.load_model() **只认 GLIOMA_MODALITY_MODEL 环境变量**：
#     漏了它就会**静默地**用内嵌系数（日志上只差一行字），重训等于白做。
# 评测不可重跑 + 环境变量最容易漏配 → 所以在 start.sh 里显式挂上，并在缺失时**大声报警**。
#
# 查找顺序：① 环境变量显式指定；② 平台约定的算法工程位置。
GLIOMA_MODALITY_MODEL="${GLIOMA_MODALITY_MODEL:-/2026aicompetition/workspace/dcs/glioma_track4/data/modality_model.json}"
if [ -f "$GLIOMA_MODALITY_MODEL" ]; then
  export GLIOMA_MODALITY_MODEL
  echo "[start.sh] 体素判别模型：$GLIOMA_MODALITY_MODEL（外部 / 重训后）"
else
  echo "[start.sh] !! 未找到 $GLIOMA_MODALITY_MODEL"
  echo "[start.sh] !! 将退回**内嵌系数**（本地模拟集训练，实测一致率约 27%，低于随机）"
  echo "[start.sh] !! 修法：在算法工程里重训后 export GLIOMA_MODALITY_MODEL=<modality_model.json 路径>"
fi

# --------------------------------------------------------------------------- #
# 真实插件工厂：**不设它 = 静默跑 Dummy 基线**（最致命的一条）
# --------------------------------------------------------------------------- #
# core/registry.py 的行为：``factory_path`` 为空时**只打印一条告警**就返回
# ``InferencePipeline()`` —— 服务照常起来、``/health`` 通、回调也正常，
# 但答案是占位内容（几乎 0 分），而日志里只有一行字，极难定位。
#
# 本仓 ``tasks/goalX`` 是完整实现（自包含，不依赖外部算法工程），
# 所以用 ``tasks.real_pipeline``；``tasks.glioma.pipeline`` 是过渡桥接（指向 glioma_track4）。
#
# ``GLIOMA_GOALS`` 控制启用哪些真实插件，未启用的用 Dummy 补位（规范 §15.1 逐项接入）。
: "${COMPETITION_PIPELINE_FACTORY:=tasks.real_pipeline:build_pipeline}"
export COMPETITION_PIPELINE_FACTORY
echo "[start.sh] 真实插件工厂：$COMPETITION_PIPELINE_FACTORY"
echo "[start.sh] 启用 Goal：${GLIOMA_GOALS:-（默认：除下面这条外全开）}"
echo "[start.sh] 权重根    ：${COMPETITION_CHECKPOINT_ROOT:-<未设>→回退 \${COMPETITION_WORKSPACE:-/2026aicompetition/workspace}/checkpoint}"

# 权重存在性：缺失时 registry 会抛 FileNotFoundError（响亮），
# 但**路径写错成另一份旧权重**是静默的 —— 所以把解析结果打印出来。
_G5DIR="${COMPETITION_CHECKPOINT_ROOT:-${COMPETITION_WORKSPACE:-/2026aicompetition/workspace}/checkpoint}/goal5_segmentation"
if [ -e "$_G5DIR/core.pt" ]; then
  echo "[start.sh] Goal5 权重：$_G5DIR/core.pt"
elif [ -d "$_G5DIR" ]; then
  echo "[start.sh] Goal5 权重：$_G5DIR/ 下无 core.pt → 取该目录下全部 *.pt 做多折集成："
  ls -1 "$_G5DIR"/*.pt 2>/dev/null || echo "[start.sh] !! 该目录下没有 *.pt"
else
  echo "[start.sh] !! Goal5 权重目录不存在：$_G5DIR"
  echo "[start.sh] !! 服务会在加载模型时抛 FileNotFoundError；请确认 COMPETITION_CHECKPOINT_ROOT"
fi

exec python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --workers 1
