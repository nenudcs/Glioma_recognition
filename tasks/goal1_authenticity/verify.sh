#!/usr/bin/env bash
# Goal1 authenticity 上线验证脚本（插件自带的运维脚本，不改动管线代码）。
#
# 用法（在仓库根目录执行）：
#   bash tasks/goal1_authenticity/verify.sh                       # 只做环境+测试+端到端 mock
#   bash tasks/goal1_authenticity/verify.sh --source /path/best.pt --also-last
#   bash tasks/goal1_authenticity/verify.sh --dataset /path/to/nifti_dir
#
# 五个阶段：
#   1) 环境与依赖   2) 权重迁移与规范路径   3) 单元/契约测试
#   4) 离线打分（可选，需 --dataset）        5) 服务级 Mock Competition
set -uo pipefail

PYTHON="${PYTHON:-python}"
SOURCE=""
DATASET=""
CHECKPOINT_ROOT="${GOAL1_CHECKPOINT_ROOT:-/2026aicompetition/workspace/checkpoint}"
ALSO_LAST=0
SKIP_SERVICE=0
MAKE_SUBSET=0
SUBSET_FROM="${GOAL1_DATA_ROOT:-/2026aicompetition/datasets/training}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --make-subset) MAKE_SUBSET="$2"; shift 2 ;;          # 从训练集切 N 例做离线打分
    --subset-from) SUBSET_FROM="$2"; shift 2 ;;          # 训练集根目录（含 annotation/）
    --also-last) ALSO_LAST=1; shift ;;
    --skip-service) SKIP_SERVICE=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数：$1" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
echo "== 仓库根：$REPO_ROOT"
echo "== Python：$(command -v "$PYTHON")"
echo "== checkpoint 根：$CHECKPOINT_ROOT"

fail() { echo; echo "❌ $1"; exit 1; }
pause() { echo; echo "---- $1"; }

# ------------------------------------------------------- 0) 参数与路径体检
if [[ -n "$DATASET" && ! -d "$DATASET" ]]; then
  echo "❌ --dataset 不是目录：$DATASET"
  echo "   可选做法："
  echo "   a) 换成真实存在的测试集目录（平台给的 dataset_path，例如 /data/testset）"
  echo "   b) 从赛方训练集自动切一个小样本：bash $0 --make-subset 8 --subset-from $SUBSET_FROM"
  echo "   c) 不做离线打分：直接不带 --dataset 重跑"
  exit 2
fi

# ---------------------------------------------------------------- 1) 环境
pause "1/5 环境与依赖"
"$PYTHON" - <<'PY' || fail "依赖不齐（pip install -r requirements.txt；训练还需 torch/timm/nibabel）"
import importlib, sys
missing = []
for name in ("fastapi", "uvicorn", "numpy", "nibabel", "openpyxl", "torch", "timm"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}")
if missing:
    print("缺失依赖 ->", "; ".join(missing)); sys.exit(1)
import torch
print(f"依赖 OK；torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"devices={torch.cuda.device_count()}")
PY
"$PYTHON" -c "import tasks.real_pipeline as rp; print('注册入口 OK：', rp.build_pipeline.__name__)" \
  || fail "注册入口不可导入（检查仓库根与包路径）"

# ---------------------------------------------------------------- 2) 权重
pause "2/5 权重迁移与规范路径"
MODEL_PATH="$CHECKPOINT_ROOT/goal1_authenticity/model.pt"
if [[ -n "$SOURCE" ]]; then
  MIGRATE_ARGS=(--source "$SOURCE" --checkpoint-root "$CHECKPOINT_ROOT")
  [[ "$ALSO_LAST" == "1" ]] && MIGRATE_ARGS+=(--also-last)
  "$PYTHON" -m tasks.goal1_authenticity.migrate_checkpoint "${MIGRATE_ARGS[@]}" \
    || fail "权重迁移失败（看上面 JSON 的 status/error）"
elif [[ ! -f "$MODEL_PATH" ]]; then
  echo "规范路径下暂无权重，尝试自动发现："
  "$PYTHON" -m tasks.goal1_authenticity.migrate_checkpoint \
    --checkpoint-root "$CHECKPOINT_ROOT" --dry-run \
    || fail "没找到可迁移的 best.pt；请用 --source 指定（例如 …/task1_runs/seed42/best.pt）"
  fail "请确认上面的 auto_detected 源，再重跑并加上 --source"
fi
[[ -f "$MODEL_PATH" ]] || fail "规范路径下没有 model.pt：$MODEL_PATH"
ls -lh "$CHECKPOINT_ROOT/goal1_authenticity"
[[ -f "$CHECKPOINT_ROOT/goal1_authenticity/model_meta.json" ]] && \
  "$PYTHON" -c "import json,sys;d=json.load(open('$CHECKPOINT_ROOT/goal1_authenticity/model_meta.json'));print('模型元信息：', {k:d.get(k) for k in ('backbone','image_size','slices_per_case','val_ap','source')})"

# ---------------------------------------------------------------- 3) 测试
pause "3/5 单元与契约测试（期望全绿）"
GOAL1_CHECKPOINT="$MODEL_PATH" "$PYTHON" -m unittest discover -s tests -t . -v \
  || fail "测试未通过"

# ---------------------------------------------------------------- 4) 离线打分
pause "4/5 离线打分（可选）"
if [[ -z "$DATASET" && "$MAKE_SUBSET" -gt 0 ]]; then
  echo "从训练集切 $MAKE_SUBSET 例作为小样本（源：$SUBSET_FROM）"
  DATASET="$("$PYTHON" - "$SUBSET_FROM" "$MAKE_SUBSET" <<'PY'
import shutil, sys, tempfile
from pathlib import Path

from tasks.goal1_authenticity.dataset import find_annotation_root, iter_images, special_dir

root = Path(sys.argv[1]).expanduser()
count = max(2, int(sys.argv[2]))
annotation = find_annotation_root(root)
if annotation is None:
    print(f"在 {root} 下找不到 annotation/fake（用 --subset-from 指定训练集根目录）", file=sys.stderr)
    sys.exit(2)

fake_dir = special_dir(annotation, "fake")
composition_dir = special_dir(annotation, "composition")
duplicate_dir = special_dir(annotation, "duplicate")
skips = tuple(item for item in (fake_dir, composition_dir, duplicate_dir) if item)

positives = iter_images(fake_dir) if fake_dir else []
negatives = iter_images(annotation, skip_dirs=skips)
half = max(1, count // 2)
picked = [(path, fake_dir) for path in positives[:half]]
picked += [(path, annotation) for path in negatives[: max(0, count - len(picked))]]

target_root = Path(tempfile.mkdtemp(prefix="goal1_subset_"))
copied = 0
for path, base in picked:
    relative = path.relative_to(base)
    accession = relative.parts[0] if len(relative.parts) > 1 else path.stem
    series_uid = relative.parts[-2] if len(relative.parts) >= 3 else path.stem
    suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else ".nii"
    destination = target_root / accession / series_uid / f"{series_uid}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(path, destination)
        copied += 1

print(target_root)
print(
    f"      [subset] {copied} 例 -> {target_root}"
    f"（正类 {min(half, len(positives))} / 负类 {copied - min(half, len(positives))}）",
    file=sys.stderr,
)
PY
)" || fail "切小样本失败（检查 --subset-from 是否指向含 annotation/ 的训练集根目录）"
fi

if [[ -n "$DATASET" ]]; then
  [[ -d "$DATASET" ]] || fail "--dataset 不是目录：$DATASET"
  OUT_DIR="$(mktemp -d)"
  GOAL1_CHECKPOINT="$MODEL_PATH" "$PYTHON" -m tasks.goal1_authenticity.evaluate \
    --dataset "$DATASET" --out-dir "$OUT_DIR" \
    || fail "离线打分失败"
  echo "逐例结果：$OUT_DIR/authenticity_scores.jsonl"
else
  echo "未提供 --dataset 且未用 --make-subset，跳过离线打分"
fi

# ---------------------------------------------------------------- 5) 服务级
pause "5/5 服务级 Mock Competition（起服务 + /call + 回调 + 输出校验）"
if [[ "$SKIP_SERVICE" == "1" ]]; then
  echo "按 --skip-service 跳过"
else
  COMPETITION_PIPELINE_FACTORY="tasks.real_pipeline:build_pipeline" \
  GOAL1_CHECKPOINT="$MODEL_PATH" \
  GOAL1_DEVICE="${GOAL1_DEVICE:-auto}" \
  "$PYTHON" -m scripts.mock_competition --timeout 300 \
    || fail "端到端 mock 未全部 PASS"
fi

echo
echo "✅ 验证完成：环境 / 权重规范路径 / 测试 / 服务级链路 全部通过"
echo "   正式启动："
echo "     export COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline"
echo "     export GOAL1_DEVICE=auto     # 权重默认读 $MODEL_PATH"
echo "     ./start.sh"
