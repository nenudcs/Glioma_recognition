#!/usr/bin/env bash
# 目标二（重复影像）上线验证脚本 —— 与 tasks/goal1_authenticity/verify.sh 同结构。
#
# 用法（在仓库根目录执行）：
#   bash tasks/goal2_duplicate/verify.sh                          # 环境 + 配置 + 测试 + mock
#   bash tasks/goal2_duplicate/verify.sh --data-root <训练集>      # 额外做图像对级标定（写规范路径）
#   bash tasks/goal2_duplicate/verify.sh --dataset /data/testset   # 在测试集上跑检索、输出候选对
#   bash tasks/goal2_duplicate/verify.sh --make-subset 8           # 从训练集切小样本再跑
#   bash tasks/goal2_duplicate/verify.sh --skip-service            # 跳过服务级 mock
#
# 五个阶段：
#   1) 环境与依赖   2) 注册入口与配置   3) 单元/契约测试
#   4) 图像对级标定或测试集检索（可选）  5) 服务级 Mock Competition
set -uo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${GOAL2_DATA_ROOT:-/2026aicompetition/datasets/training}"
CHECKPOINT_ROOT="${GOAL2_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT:-/2026aicompetition/workspace/checkpoint}}"
DATASET=""
OUT_DIR="${GOAL2_RUN_DIR:-}"
TARGET_FPR="0.10"
WORKERS="4"
MAX_VOLUMES="0"
MAKE_SUBSET=0
SUBSET_FROM="${GOAL2_DATA_ROOT:-/2026aicompetition/datasets/training}"
SKIP_SERVICE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; SUBSET_FROM="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --make-subset) MAKE_SUBSET="$2"; shift 2 ;;
    --subset-from) SUBSET_FROM="$2"; shift 2 ;;
    --target-fpr) TARGET_FPR="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --max-volumes) MAX_VOLUMES="$2"; shift 2 ;;
    --skip-service) SKIP_SERVICE=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "== 仓库根：$REPO_ROOT"
echo "== Python：$(command -v "$PYTHON")"
echo "== 数据根：$DATA_ROOT"
echo "== checkpoint 根：$CHECKPOINT_ROOT"

fail() { echo; echo "[FAIL] $1"; exit 1; }
pause() { echo; echo "---- $1"; }

pause "1/5 环境与依赖"
"$PYTHON" - <<'PY' || fail "依赖不齐（pip install -r requirements.txt）"
import importlib, sys

from tasks.goal2_stitched.dataset import configure_stdout

configure_stdout()          # Windows/GBK 控制台也能安全输出中文
missing = []
for name in ("numpy", "nibabel", "openpyxl", "fastapi", "uvicorn"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}")
if missing:
    print("缺失依赖 ->", "; ".join(missing))
    sys.exit(1)
print("依赖 OK（重复检测只用 numpy；nibabel 供全库扫描读盘）")
PY

pause "2/5 注册入口与配置"
GOAL2_CHECKPOINT_ROOT="$CHECKPOINT_ROOT" "$PYTHON" - <<'PY' || fail "注册入口或配置不可用"
from tasks.goal2_duplicate import checkpoint
from tasks.goal2_duplicate.config import Goal2DuplicateConfig
from tasks.goal2_duplicate.retrieval import NearDuplicateIndex
from tasks.goal2_stitched.dataset import configure_stdout

configure_stdout()
print("标定文件规范路径：", checkpoint.calibration_path())
config = Goal2DuplicateConfig.from_env()
print("重复配置：", config.describe())
if config.calibration_file is None:
    print("[warn] 规范路径下没有 calibration.json：当前用的是默认相似度中心，上服务器后必须标定")
index = NearDuplicateIndex(config.descriptor_params(), mode=config.mode, **config.index_kwargs())
print("索引可实例化：volumes =", index.volumes)

import tasks.real_pipeline as rp
pipeline = rp.build_pipeline()
chain = [(b.context_field, type(b.task).__name__) for b in pipeline.study_tasks]
print("注册入口 OK；任务链：", chain)
print("数据集级任务：", type(pipeline.duplicate_task).__name__)

names = [kind for _, kind in chain]
if "DummyStitchedTask" in names or type(pipeline.duplicate_task).__name__ == "DummyDuplicateTask":
    raise SystemExit(
        "[FAIL] tasks/real_pipeline.py 仍是旧版：goal2 绑到了 Dummy。\n"
        "       修法：用 Goal2 增量包里的 tasks/real_pipeline.py 覆盖（应用顺序必须 goal1 → goal2），\n"
        "       或直接解压合并包 goal12_upload.zip 到仓库根目录。"
    )
PY

pause "3/5 单元与契约测试（期望全绿）"
"$PYTHON" -m unittest tests.contracts.test_goal2_contract -v || fail "契约测试未通过"

pause "4/5 图像对级标定或测试集检索（可选）"
if [[ -z "$DATASET" && "$MAKE_SUBSET" -gt 0 ]]; then
  echo "从训练集切 $MAKE_SUBSET 例作为小样本（源：$SUBSET_FROM）"
  SUBSET_DIR="$(mktemp -d)/subset"
  "$PYTHON" -m tasks.goal2_stitched.dataset \
    --data-root "$SUBSET_FROM" --make-subset "$MAKE_SUBSET" --target-dir "$SUBSET_DIR" \
    || fail "切小样本失败（检查 --subset-from 是否指向含 annotation/ 的训练集根目录）"
  DATASET="$SUBSET_DIR"
fi

SCORE_ARGS=()
[[ -n "$OUT_DIR" ]] && SCORE_ARGS+=(--out-dir "$OUT_DIR")
[[ "$MAX_VOLUMES" != "0" ]] && SCORE_ARGS+=(--max-volumes "$MAX_VOLUMES")
if [[ -n "$DATASET" ]]; then
  [[ -d "$DATASET" ]] || fail "--dataset 不是目录：$DATASET"
  echo "测试集检索：$DATASET（输出与提交同格式的候选对）"
  GOAL2_CHECKPOINT_ROOT="$CHECKPOINT_ROOT" "$PYTHON" -m tasks.goal2_duplicate.evaluate \
    --dataset "$DATASET" "${SCORE_ARGS[@]}" --workers "$WORKERS" \
    || fail "测试集检索失败"
elif [[ -d "$DATA_ROOT" ]]; then
  echo "金标准体检："
  "$PYTHON" -m tasks.goal2_duplicate.dataset --data-root "$DATA_ROOT" || echo "[warn] 未找到金标准文件"
  echo "图像对级标定：$DATA_ROOT（正类 = annotation/duplicate 金标准）"
  GOAL2_CHECKPOINT_ROOT="$CHECKPOINT_ROOT" "$PYTHON" -m tasks.goal2_duplicate.evaluate \
    --data-root "$DATA_ROOT" "${SCORE_ARGS[@]}" \
    --target-fpr "$TARGET_FPR" --workers "$WORKERS" \
    || fail "标定失败"
else
  echo "[warn] 既没有 --dataset，也找不到数据根 $DATA_ROOT：跳过标定/检索"
  echo "   上服务器后必须重跑：bash $0 --data-root <训练集根目录>"
fi

pause "5/5 服务级 Mock Competition（起服务 + /call + 回调 + 输出校验）"
if [[ "$SKIP_SERVICE" == "1" ]]; then
  echo "按 --skip-service 跳过"
else
  if [[ -z "${GOAL1_CHECKPOINT:-}" ]]; then
    echo "[info] 未设置 GOAL1_CHECKPOINT：goal1 会用兜底概率（服务仍能跑通，只是目标一没有真实输出）"
  fi
  COMPETITION_PIPELINE_FACTORY="tasks.real_pipeline:build_pipeline" \
  GOAL2_CHECKPOINT_ROOT="$CHECKPOINT_ROOT" \
  GOAL1_DEVICE="${GOAL1_DEVICE:-auto}" \
  "$PYTHON" -m scripts.mock_competition --timeout 300 \
    || fail "端到端 mock 未全部 PASS"
fi

echo
echo "[ok] 重复影像验证完成：环境 / 配置 / 测试 / 服务级链路 全部通过"
echo "   正式启动："
echo "     export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline"
echo "     export GOAL2_CHECKPOINT_ROOT=$CHECKPOINT_ROOT   # 相似度中心从 \$GOAL2_CHECKPOINT_ROOT/goal2_duplicate/calibration.json 读取"
echo "     ./start.sh"
