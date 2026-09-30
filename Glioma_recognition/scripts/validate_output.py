"""独立校验已产出的评测目录（规范 §5、§19.3）。

与运行期 ``OutputValidator`` 的区别：本脚本是**离线**入口，用于

- 赛前自查：拿平台返回的目录抽查一遍，确认不是"看起来写完了但结构不合规"；
- 复现排障：某个 evaluation 失败后，单独对 staging/正式目录跑校验，不启动服务；
- 演练"Validator 能抓住人为破坏"：规范 §19.3 要求验证"改字段、删文件、
  改 affine、写入值 2"时必须失败，本脚本配合 ``--selftest`` 做这件事。

用法::

    python scripts/validate_output.py --dir /path/to/answer/<evaluation_id>
    python scripts/validate_output.py --dir ... --expect ACC001,ACC002   # 额外检查完整性
    python scripts/validate_output.py --selftest --dir ...               # 故意破坏后校验必须失败
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from output.validator import OutputValidator                          # noqa: E402


def _load_accessions(evaluation_dir: Path) -> set[str]:
    return {p.name for p in evaluation_dir.iterdir() if p.is_dir()}


def validate(evaluation_dir: Path, expect: set[str] | None = None) -> int:
    """对目录做最终布局校验；返回退出码。"""
    v = OutputValidator()
    accessions = _load_accessions(evaluation_dir)
    if expect:
        missing = expect - accessions
        extra = accessions - expect
        if missing or extra:
            print(f"[validate] ✗ 目录集合与预期不符 缺失={sorted(missing)} 多出={sorted(extra)}")
            return 1
    try:
        v.validate_final_layout(evaluation_dir, accessions)
        v.validate_duplicates(evaluation_dir / "duplicate_pairs.jsonl", accessions)
    except Exception as exc:                                      # noqa: BLE001
        print(f"[validate] ✗ {type(exc).__name__}: {exc}")
        return 1

    n_bad = 0
    for acc in sorted(accessions):
        pj = evaluation_dir / acc / "prediction.json"
        if not pj.is_file():
            print(f"[validate] ✗ {acc}: 缺 prediction.json")
            n_bad += 1
            continue
        try:
            payload = json.loads(pj.read_text(encoding="utf-8"))
        except Exception as exc:                                  # noqa: BLE001
            print(f"[validate] ✗ {acc}: prediction.json 无法解析（{exc}）")
            n_bad += 1
            continue
        # URI 指向的文件必须真实存在
        uris = payload.get("SegmentationMaskURI") or {}
        if isinstance(uris, dict):
            for key, uri in uris.items():
                rel = str(uri).lstrip("./")
                if not (evaluation_dir / acc / rel).is_file():
                    print(f"[validate] ✗ {acc}: {key} 的 URI 指向不存在的文件：{uri}")
                    n_bad += 1

    print(f"[validate] {'✓ 全部通过' if n_bad == 0 else f'✗ {n_bad} 例有问题'}"
          f"（{len(accessions)} 例）")
    return 0 if n_bad == 0 else 1


def selftest(evaluation_dir: Path) -> int:
    """破坏性自检：人为改坏一处，Validator **必须**报错（规范 §19.3）。"""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "copy"
        shutil.copytree(evaluation_dir, work)
        accs = sorted(_load_accessions(work))
        if not accs:
            print("[selftest] ✗ 目录里没有病例，无法自检")
            return 1
        acc = accs[0]

        cases = []

        # ① 概率越界
        pj = work / acc / "prediction.json"
        payload = json.loads(pj.read_text(encoding="utf-8"))
        payload["IsNotHumanBodyProb"] = 1.5
        pj.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        cases.append(("概率越界", work))

        # ② 删掉 duplicate 文件
        work2 = Path(tmp) / "copy2"
        shutil.copytree(evaluation_dir, work2)
        (work2 / "duplicate_pairs.jsonl").unlink(missing_ok=True)
        cases.append(("缺 duplicate_pairs.jsonl", work2))

        # ③ 掩码写入值 2（规范明写会导致该例分割 0 分）
        work3 = Path(tmp) / "copy3"
        shutil.copytree(evaluation_dir, work3)
        masks = list((work3 / acc).rglob("*.nii.gz"))
        if masks:
            import nibabel as nib
            import numpy as np
            img = nib.load(str(masks[0]))
            arr = np.asanyarray(img.dataobj).copy()
            arr.reshape(-1)[:1] = 2
            nib.save(nib.Nifti1Image(arr, img.affine), str(masks[0]))
            cases.append(("掩码含非 0/1 值", work3))

        n_ok = 0
        for name, path in cases:
            caught = validate(path, None) != 0
            print(f"[selftest] {'✓ 已捕获' if caught else '✗ 漏检'}：{name}")
            n_ok += int(caught)
        print(f"[selftest] {n_ok}/{len(cases)} 项被正确捕获")
        return 0 if n_ok == len(cases) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="待校验的 evaluation 目录")
    ap.add_argument("--expect", default=None, help="逗号分隔的期望 accession 全集")
    ap.add_argument("--selftest", action="store_true", help="破坏性自检：Validator 必须能捕获")
    a = ap.parse_args()

    d = Path(a.dir)
    if not d.is_dir():
        print(f"[validate] ✗ 目录不存在：{d}")
        return 2
    expect = {x.strip() for x in a.expect.split(",") if x.strip()} if a.expect else None
    return selftest(d) if a.selftest else validate(d, expect)


if __name__ == "__main__":
    raise SystemExit(main())
