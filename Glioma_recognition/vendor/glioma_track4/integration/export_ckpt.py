"""把本工程训练出的权重导出到团队约定的 checkpoint 目录。

规范 §5.2 约定：

```text
/2026aicompetition/workspace/checkpoint/
├── goal1_authenticity/model.pt
├── goal2_stitched/model.pt
├── goal2_duplicate/encoder.pt
├── goal3_tumor/model.pt
├── goal4_diagnosis/model.pt
└── goal5_segmentation/
    ├── core.pt
    └── flair.pt
```

本工程的骨干是**一个多任务网络**（一次前向同时给出分割、结构化、特殊影像与嵌入），
因此同一份权重会被导出到各 goal 目录（默认使用**硬链接**，不额外占用磁盘）；
若后续某个 Goal 换成独立模型，只需替换对应目录下的文件即可，团队侧无需改动。

用法：
    python -m integration.export_ckpt --mode link
    python -m integration.export_ckpt --src checkpoints --mode copy --folds g4_fold0,g4_fold1
    python -m integration.export_ckpt --verify
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from .common import CHECKPOINT_ROOT, PROJECT_ROOT

#: goal 目录 → 该目录下的文件名（与本工程共享同一多任务权重）
GOAL_FILES = {
    "goal1_authenticity": "model.pt",
    "goal2_stitched": "model.pt",
    "goal2_duplicate": "encoder.pt",
    "goal3_tumor": "model.pt",
    "goal4_diagnosis": "model.pt",
    "goal5_segmentation": "core.pt",
}


def discover_weights(src: Path, folds: list[str] | None = None) -> dict[str, Path]:
    """发现训练产物，**优先 ``best.pth``**（``last.pth`` 仅在缺少 best 时兜底）。

    注意：早期实现把 ``best`` 与 ``last`` 放在同一循环里，两者目录名相同（如
    ``g4_fold0``）会让 ``last.pth`` 覆盖 ``best.pth``——导出到错误权重。
    """
    def _collect(patterns: tuple[str, ...]) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for pattern in patterns:
            for path in sorted(src.glob(pattern)):
                tag = path.parent.name
                if folds and tag not in folds:
                    continue
                out.setdefault(tag, path)                         # 同 tag 先到先得
        return out

    # ⚠️ `g4_full*` 必须**一起**收：全量训练（`03_train.sh full`）的产物在
    # ``checkpoints/g4_full/``，它不含 `fold` 字样。只 glob `g4_fold*` 时，
    # "只训了全量模型"的项目里 `--folds g4_full` 会一条都找不到 →
    # 报"未找到 g4_fold*/best.pth，先训练"（磁盘上明明有训好的权重）。
    # `g4_full*` 顺带覆盖多 seed 的全量 tag（`g4_full43`）。
    found = _collect(("g4_fold*/best.pth", "g4L_fold*/best.pth", "g4_full*/best.pth"))
    if not found:
        found = _collect(("g4_fold*/last.pth", "g4L_fold*/last.pth",
                          "g4_full*/last.pth"))
    return found


def _place(source: Path, target: Path, mode: str) -> str:
    """把权重放到目标路径；返回实际采用的方式。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "link":
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            pass
    if mode == "symlink":
        target.symlink_to(source)
        return "symlink"
    shutil.copy2(source, target)
    return "copy"


def export(src_dir: Path, dst_root: Path, mode: str, folds: list[str] | None,
           goal_files: dict[str, str]) -> int:
    weights = discover_weights(src_dir, folds)
    if not weights:
        print(f"[export] ✗ 在 {src_dir} 未找到 g4_fold*/best.pth；先训练（scripts/03_train.sh 0）")
        return 1
    names = ", ".join(f"{k}={v.name}" for k, v in weights.items())
    print(f"[export] 源权重 {len(weights)} 个: {names}")
    print(f"[export] 目标根目录: {dst_root}（mode={mode}）")

    for goal, filename in goal_files.items():
        goal_dir = dst_root / goal
        if goal_dir.is_dir():
            for old in goal_dir.glob("*"):
                if old.is_file() or old.is_symlink():
                    old.unlink()
        for tag, path in weights.items():
            # 多折：goal5 按 <tag>.pt 保留全部（供集成），其余 Goal 取首个（共享骨干）
            target_name = f"{tag}.pt" if goal == "goal5_segmentation" and len(weights) > 1 \
                else filename
            how = _place(path, goal_dir / target_name, mode)
            print(f"  · {goal}/{target_name}  <-  {path.name}  ({how})")
            if len(weights) == 1 or goal == "goal5_segmentation":
                continue
            break

    manifest = {
        "source_dir": str(src_dir),
        "weights": {k: str(v) for k, v in weights.items()},
        "goals": goal_files,
        "mode": mode,
    }
    (dst_root / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[export] 完成；清单 -> {dst_root / 'export_manifest.json'}")
    return 0


def verify() -> int:
    """复核团队约定路径下的权重可被插件解析（模拟服务启动时的查找）。"""
    from .common import resolve_checkpoints

    try:
        found = resolve_checkpoints("goal5_segmentation")
    except FileNotFoundError as exc:
        print(f"[verify] ✗ {exc}")
        return 1
    print(f"[verify] goal5_segmentation 解析到 {len(found)} 个权重:")
    for path in found:
        print(f"  · {path}")
    for goal in GOAL_FILES:
        d = Path(CHECKPOINT_ROOT) / goal
        files = sorted(p.name for p in d.glob("*")) if d.is_dir() else []
        flag = "ok" if files else "缺"
        print(f"[verify] {goal:<22} {flag}  {files}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(PROJECT_ROOT / "checkpoints"))
    ap.add_argument("--dst", default=str(CHECKPOINT_ROOT))
    ap.add_argument("--mode", choices=("link", "copy", "symlink"), default="link")
    ap.add_argument("--folds", default=None, help="逗号分隔的 tag 白名单，如 g4_fold0,g4_fold1")
    ap.add_argument("--verify", action="store_true", help="只复核团队约定路径")
    a = ap.parse_args()

    if a.verify:
        return verify()
    folds = [f.strip() for f in a.folds.split(",")] if a.folds else None
    return export(Path(a.src), Path(a.dst), a.mode, folds, GOAL_FILES)


if __name__ == "__main__":
    sys.exit(main())
