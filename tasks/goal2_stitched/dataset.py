"""[研发] 赛方训练数据目录扫描（标定与自检用，**不进比赛运行链路**）。

数据布局（赛方说明）::

    <data_root>/
    ├── annotation/
    │   ├── fake/           目标一正类（假人体/非人体）
    │   ├── Composition/    目标二正类（拼接影像）
    │   └── duplicate/      目标二重复影像的阳性/阴性 + 金标准文件
    └── <正常病例>/...      目标二负类（正常影像）

本模块只做「发现」：把某个特殊目录的影像取出来当正类，把其余正常影像当负类，
供 ``evaluate.py`` 标定阈值或做自检。比赛运行链路（``task.py``）不 import 本模块。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

NIFTI_SUFFIXES = (".nii", ".nii.gz")
MASK_HINTS = ("mask", "seg", "label", "roi")
SPECIAL_KINDS = ("fake", "composition", "duplicate")
# 目录名同义词：赛方目录名一旦变化（Composition / stitched / 拼接 ...），体检能直接提示
KIND_SYNONYMS = {
    "fake": ("fake", "synth", "非人体", "假人体"),
    "composition": ("compos", "compose", "stitch", "拼接", "拼合"),
    "duplicate": ("dup", "重复", "copy", "replica"),
}


def configure_stdout() -> None:
    """让 CLI 的 JSON 输出在 Windows/GBK 控制台上也不报 UnicodeEncodeError。"""
    stream = getattr(sys, "stdout", None)
    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 配置失败时保持原样
            pass


def is_image(path: Path) -> bool:
    if not path.name.lower().endswith(NIFTI_SUFFIXES):
        return False
    name = path.name.lower()
    return not any(hint in name for hint in MASK_HINTS)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


_REPO_MARKERS = ("tasks", "pipeline", "core", "app", "output")


def looks_like_code_repo(root: Path) -> bool:
    """判断"数据根"是不是给成了代码仓库（``$DATA_ROOT`` 为空时最容易踩）。"""
    if not root.is_dir():
        return False
    names = {path.name for path in root.iterdir() if path.is_dir()}
    if "annotation" in names:
        return False
    return len(names & set(_REPO_MARKERS)) >= 3


def iter_images(root: Path, skip_dirs: Sequence[Path] = ()) -> list[Path]:
    """递归收集输入影像（跳过 ``skip_dirs`` 子树及其内容）。"""
    if root is None or not root.is_dir():
        return []
    skips = [Path(item).resolve() for item in skip_dirs if item is not None]
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_image(path):
            continue
        if any(_is_within(path.resolve(), skip) for skip in skips):
            continue
        found.append(path)
    return found


def special_dir(base: Path | None, kind: str) -> Path | None:
    """在 ``base`` 下按大小写不敏感前缀找 ``fake`` / ``composition`` / ``duplicate``。"""
    if base is None or not base.is_dir():
        return None
    for child in sorted(path for path in base.iterdir() if path.is_dir()):
        if child.name.strip().lower().startswith(kind):
            return child
    return None


def find_special_root(data_root: Path, kinds: Sequence[str] = SPECIAL_KINDS) -> Path | None:
    """定位含任一特殊目录的标注根目录（``data_root`` 自身或其子目录）。"""
    if not data_root.is_dir():
        return None
    candidates = [data_root, data_root / "annotation"]
    candidates += sorted(path for path in data_root.iterdir() if path.is_dir())
    for candidate in candidates:
        if any(special_dir(candidate, kind) is not None for kind in kinds):
            return candidate
    for child in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for grandchild in sorted(path for path in child.iterdir() if path.is_dir()):
            if any(special_dir(grandchild, kind) is not None for kind in kinds):
                return grandchild
    return None


def find_annotation_root(data_root: Path) -> Path | None:
    """含 ``Composition``（或 ``fake``）的标注根目录。"""
    for kinds in (("composition",), ("fake",)):
        root = find_special_root(data_root, kinds=kinds)
        if root is not None:
            return root
    return None


def resolve_annotation_root(
    data_root: str | Path,
    annotation_root: str | Path | None = None,
    positive_kind: str = "composition",
) -> Path | None:
    """解析标注根目录，容忍「标注目录被改名」的情况。

    顺序：显式 ``--annotation-root`` → 含目标正类目录的根 → 按同义词在全树里找到候选目录后
    取其父目录 → 含 ``fake`` / ``composition`` 的根。
    """
    root = Path(data_root).expanduser()
    if annotation_root is not None:
        return Path(annotation_root).expanduser()
    if not root.is_dir():
        return None
    found = find_special_root(root, kinds=(positive_kind,))
    if found is not None:
        return found
    if positive_kind in SPECIAL_KINDS:
        for candidate in candidate_kind_dirs(root, positive_kind, limit=5):
            return Path(candidate).parent
    else:
        prefix = str(positive_kind).strip().lower()
        if prefix:
            for path in sorted(root.rglob("*")):
                if path.is_dir() and path.name.strip().lower().startswith(prefix):
                    return path.parent
    return find_annotation_root(root)


def labeled_volumes(
    data_root: str | Path,
    *,
    annotation_root: str | Path | None = None,
    positive_kind: str = "composition",
    exclude_kinds: Sequence[str] = SPECIAL_KINDS,
) -> dict[str, list[Path]]:
    """``{"positives": [...], "negatives": [...], ...}``（正常影像 = 其余全部影像）。

    ``positive_kind`` 是要当正类的特殊目录前缀（拼接用 ``composition``，
    重复检测的正类来自金标准文件，不用本函数）。
    """
    root = Path(data_root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{root}")
    ann = resolve_annotation_root(root, annotation_root, positive_kind)
    positive_dir = special_dir(ann, positive_kind) if ann else None
    resolved_positive_kind = positive_kind
    if positive_dir is None and ann is not None and Path(ann).is_dir():
        # 目录被改名（例如 composition → compositing）：按同义词找到候选后自动采用
        for candidate in candidate_kind_dirs(ann, positive_kind, limit=5):
            name = Path(candidate).name
            found = special_dir(ann, name)
            if found is not None:
                positive_dir = found
                resolved_positive_kind = name
                break
    specials = tuple(
        path for path in (special_dir(ann, kind) for kind in exclude_kinds) if path
    )
    if positive_dir is not None and positive_dir not in specials:
        # 正类目录必须从负类里排除，否则标定会被“自己人”污染
        specials = specials + (positive_dir,)

    positives = iter_images(positive_dir) if positive_dir else []
    negatives: list[Path] = []
    seen = {str(path.resolve()) for path in positives}
    for base, skips in (
        (ann, specials),
        (root, (ann,) + specials if ann else specials),
    ):
        if base is None:
            continue
        for path in iter_images(base, skip_dirs=tuple(item for item in skips if item)):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            negatives.append(path)
    return {
        "positives": positives,
        "negatives": negatives,
        "data_root": [root],
        "annotation_root": [ann] if ann else [],
        "positive_dir": [positive_dir] if positive_dir else [],
        "positive_kind_requested": [str(positive_kind)],
        "positive_kind": [str(resolved_positive_kind)],
        "special_dirs": {
            kind: [special_dir(ann, kind)] if ann and special_dir(ann, kind) else []
            for kind in SPECIAL_KINDS
        },
    }


def _child_dirs(base: Path | None, limit: int = 40) -> list[str]:
    if base is None or not base.is_dir():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir())[:limit]


def candidate_kind_dirs(
    data_root: str | Path,
    kind: str,
    *,
    max_depth: int = 4,
    limit: int = 20,
) -> list[str]:
    """按同义词在全树里找“可能是某类标注目录”的候选（用于目录名变化时排障）。"""
    synonyms = KIND_SYNONYMS.get(kind, (kind,))
    root = Path(data_root).expanduser()
    if not root.is_dir():
        return []
    found: list[str] = []
    root_depth = len(root.parts)
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        if len(path.parts) - root_depth > max_depth:
            continue
        name = path.name.strip().lower()
        if any(word in name for word in synonyms):
            found.append(str(path))
        if len(found) >= limit:
            break
    return found


def effective_special_dir(ann: Path | None, kind: str) -> Path | None:
    """特殊目录的"实际位置"：先严格前缀，失败再按同义词候选（如 compositing）。"""
    if ann is None or not Path(ann).is_dir():
        return None
    direct = special_dir(ann, kind)
    if direct is not None:
        return direct
    for candidate in candidate_kind_dirs(ann, kind, limit=5):
        found = special_dir(ann, Path(candidate).name)
        if found is not None:
            return found
    return None


def inspect_layout(
    data_root: str | Path,
    *,
    annotation_root: str | Path | None = None,
) -> dict[str, object]:
    """把数据目录的真实布局打印出来（排障用：目录改名/换层级时一眼看清）。"""
    root = Path(data_root).expanduser()
    ann = resolve_annotation_root(root, annotation_root)
    exact = {kind: special_dir(ann, kind) for kind in SPECIAL_KINDS}
    specials = {kind: (exact[kind] or effective_special_dir(ann, kind)) for kind in SPECIAL_KINDS}
    candidates = {
        kind: candidate_kind_dirs(root, kind)
        for kind in SPECIAL_KINDS
        if exact[kind] is None
    }
    suggested_positive_kind = None
    if exact["composition"] is None:
        if specials["composition"] is not None:
            suggested_positive_kind = Path(specials["composition"]).name
        elif candidates.get("composition"):
            suggested_positive_kind = Path(candidates["composition"][0]).name
    hints: list[str] = []
    if exact["composition"] is None:
        if suggested_positive_kind:
            hints.append(
                f"正类目录名不是 composition，实际为 {suggested_positive_kind}：已自动采用"
                f"（见 special_dirs.composition），也可显式传 --positive-kind {suggested_positive_kind}"
            )
        else:
            hints.append(
                "未找到可作为拼接正类的目录：candidate_dirs 里没有明显候选，"
                "请用 --annotation-root 指定含正类目录的那一层"
            )
    if exact["duplicate"] is None:
        hints.append("未找到名字以 duplicate 开头的目录：重复影像的金标准文件在该目录下")
    return {
        "data_root": str(root),
        "data_root_children": _child_dirs(root),
        "annotation_root": None if ann is None else str(ann),
        "annotation_children": _child_dirs(ann),
        "special_dirs": {
            kind: (None if path is None else str(path)) for kind, path in specials.items()
        },
        "special_dirs_exact": {
            kind: (None if path is None else str(path)) for kind, path in exact.items()
        },
        "special_image_counts": {
            kind: (0 if path is None else len(iter_images(path)))
            for kind, path in specials.items()
        },
        "candidate_dirs": candidates,
        "suggested_positive_kind": suggested_positive_kind,
        "hints": hints,
    }


def make_subset(
    data_root: str | Path,
    *,
    count: int = 8,
    target_root: str | Path | None = None,
) -> Path:
    """从训练集切一个小样本，摆成**测试集布局**（``<检查号>/<序列号>/<序列号>.nii.gz``）。

    用途：手上没有真实测试集时，也能跑完整的 ``/call`` 端到端 mock 与离线打分。
    取样顺序刻意让 Goal2 有东西可测：先取 ``annotation/duplicate``（重复病例，通常带金标准）、
    再取 ``annotation/Composition``（拼接病例），最后用正常影像补齐。
    """
    import shutil
    import tempfile

    source = Path(data_root).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"训练集根目录不存在：{source}")
    annotation = find_annotation_root(source)
    if annotation is None:
        raise FileNotFoundError(
            f"在 {source} 下找不到 annotation/（可用 --data-root 指定训练集根目录）"
        )

    duplicate_dir = special_dir(annotation, "duplicate")
    composition_dir = special_dir(annotation, "composition")
    fake_dir = special_dir(annotation, "fake")
    specials = tuple(
        path for path in (duplicate_dir, composition_dir, fake_dir) if path
    )

    wanted = max(2, int(count))
    picks: list[tuple[Path, Path]] = []
    quota_special = max(1, wanted // 2)
    for kind_dir in (duplicate_dir, composition_dir):
        if kind_dir is None:
            continue
        for path in iter_images(kind_dir)[:quota_special]:
            picks.append((path, kind_dir))
        if len(picks) >= quota_special:
            break
    if len(picks) < wanted:
        for base, skips in (
            (annotation, specials),
            (source, tuple([annotation] + list(specials))),
        ):
            for path in iter_images(base, skip_dirs=tuple(item for item in skips if item)):
                if len(picks) >= wanted:
                    break
                picks.append((path, base))
            if len(picks) >= wanted:
                break
    if not picks:
        raise FileNotFoundError(f"{source} 下没有可复制的 NIfTI 影像")

    destination_root = (
        Path(target_root).expanduser()
        if target_root is not None
        else Path(tempfile.mkdtemp(prefix="goal2_subset_"))
    )
    destination_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    for path, base in picks:
        relative = path.relative_to(base)
        accession = relative.parts[0] if len(relative.parts) > 1 else path.stem
        series_uid = relative.parts[-2] if len(relative.parts) >= 3 else path.stem
        suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else ".nii"
        target = destination_root / accession / series_uid / f"{series_uid}{suffix}"
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied += 1
    if copied == 0:
        raise FileNotFoundError(f"切样本失败：{source} 下的影像都无法复制")
    return destination_root


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="目标二数据目录体检（不跑模型）")
    parser.add_argument("--data-root", type=Path, default=Path("/2026aicompetition/datasets/training"))
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--positive-kind", default="composition")
    parser.add_argument("--make-subset", type=int, default=0,
                        help="从训练集切 N 例到测试集布局（无测试集时做端到端验证）")
    parser.add_argument("--target-dir", type=Path, default=None,
                        help="--make-subset 的输出目录（不存在则创建）")
    args = parser.parse_args(argv)

    if args.make_subset > 0:
        subset = make_subset(
            args.data_root,
            count=args.make_subset,
            target_root=args.target_dir,
        )
        print(json.dumps({"subset_root": str(subset)}, ensure_ascii=False))
        return 0

    raw_root = str(args.data_root).strip()
    resolved_root = Path(raw_root).expanduser()
    if raw_root in ("", ".") or not resolved_root.is_dir():
        print(
            json.dumps(
                {
                    "error": f"--data-root 无效：{raw_root!r}（最常见的原因为 $DATA_ROOT 为空）",
                    "resolved": str(resolved_root.resolve()) if raw_root else "",
                    "hint": "先 `echo \"[$DATA_ROOT]\"` 检查变量；不确定就直接写绝对路径："
                    "--data-root /2026aicompetition/datasets/training",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2
    if looks_like_code_repo(resolved_root.resolve()):
        print(
            json.dumps(
                {
                    "error": f"--data-root 指向的是代码仓库，不是数据集根：{resolved_root}",
                    "data_root_children": sorted(
                        path.name for path in resolved_root.iterdir() if path.is_dir()
                    )[:20],
                    "hint": "$DATA_ROOT 多半是空的（新 shell/新 pod 后 export 会丢）。"
                    "用绝对路径重跑：--data-root /2026aicompetition/datasets/training",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    report = labeled_volumes(
        args.data_root,
        annotation_root=args.annotation_root,
        positive_kind=args.positive_kind,
    )
    layout = inspect_layout(args.data_root, annotation_root=args.annotation_root)
    payload = {
        "data_root": str(args.data_root),
        "annotation_root": [str(path) for path in report["annotation_root"]],
        "positive_kind_requested": report["positive_kind_requested"][0],
        "positive_kind": report["positive_kind"][0],
        "positive_dir": [str(path) for path in report["positive_dir"]],
        "positives": len(report["positives"]),
        "negatives": len(report["negatives"]),
        "positive_examples": [str(path) for path in report["positives"][:3]],
        "negative_examples": [str(path) for path in report["negatives"][:3]],
        "special_dirs": layout["special_dirs"],
        "special_dirs_exact": layout["special_dirs_exact"],
        "special_image_counts": layout["special_image_counts"],
        "annotation_children": layout["annotation_children"],
        "data_root_children": layout["data_root_children"],
        "candidate_dirs": layout["candidate_dirs"],
        "suggested_positive_kind": layout["suggested_positive_kind"],
        "hints": layout["hints"],
    }
    if report["positive_kind"][0] != report["positive_kind_requested"][0]:
        payload["hints"].append(
            f"正类目录名不是 composition，已自动采用 “{report['positive_kind'][0]}”"
            f"（来自 candidate_dirs）；如需固定，可显式传 "
            f"--positive-kind {report['positive_kind'][0]}"
        )
    if looks_like_code_repo(Path(args.data_root).resolve()):
        payload["hints"].append("警告：--data-root 看起来是代码仓库目录（检查 $DATA_ROOT 是否为空）")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report["positives"] and report["negatives"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
