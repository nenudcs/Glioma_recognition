#!/usr/bin/env python3
"""体素判别一致性检查（第三条腿的**准确率 + 概率饱和度**）。

用训练集里**表能匹配上**的序列当金标准：那些序列自带权威模态标签，
拿它测 ``data.voxel_modality`` 的准确率；同时统计 **top-2 间隔** 判断概率是否饱和
（``置信 1.0 / 次优 0.0`` 说明输入远超训练分布，0.5 的置信门槛形同虚设）。

用法::

    # 前台（实时看输出）
    python3 scripts/voxel_consistency.py
    python3 scripts/voxel_consistency.py /2026aicompetition/datasets/training/annotation 120

    # 后台（推荐：不占终端，日志落盘）
    mkdir -p logs
    nohup python3 scripts/voxel_consistency.py > logs/voxel_consistency.log 2>&1 &
    echo $! > logs/voxel_consistency.pid
    tail -f logs/voxel_consistency.log

环境变量：

* ``GLIOMA_MODALITY_MODEL`` —— 外部模型 JSON（不设则用**内嵌系数**）
* ``GLIOMA_VOXEL_MAX`` —— 最多比对多少例（默认 60；设大些更可靠，如 300）

**不占 GPU、不读权重**，可与推理任务并行跑。

判据（脚本末尾也会打印）：

===============  ==============  ==============================================
一致率           饱和占比        结论
===============  ==============  ==============================================
>=80%            <20%            判别器可信 → 保留"覆盖表里 `其他`"的逻辑
>=80%            高              "看起来准"但概率不可信 → 需人工判断
50~80%           —               边缘 → 用训练集重训（见下）
<50%             —               ≈随机猜 → **不该覆盖**表里的 `其他`
===============  ==============  ==============================================

重训（纯 CPU，约 3~6 分钟，不碰 GPU / 不碰 ``best.pth``）::

    cd ../glioma_track4
    python3 scripts/31_train_modality_model.py --root /2026aicompetition/datasets/training/annotation
    cp -v data/modality_model.json ../Glioma_recognition-main/data/
    export GLIOMA_MODALITY_MODEL=$PWD/../Glioma_recognition-main/data/modality_model.json
"""
from __future__ import annotations

import itertools
import os
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import nibabel as nib

# 允许 `python3 scripts/voxel_consistency.py` 直接跑：脚本在 scripts/ 下，
# sys.path[0] 是 scripts/ 而不是工程根，因此手动补上工程根。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.modality_fallback import _norm, _read_rows, _table_in      # noqa: E402
from data.voxel_modality import describe, load_model                 # noqa: E402

#: 官方训练集（含 ``SeriesType.xlsx`` 的那一层）
DEFAULT_ROOT = "/2026aicompetition/datasets/training/annotation"

#: 本模型的类别空间（与训练脚本的 ``OFFICIAL_CLASSES`` 一致，**没有 T1 类**）
MODEL_CLASSES = ("T1CE", "T2", "FLAIR")


def expected_class(label: str) -> str | None:
    """表标签 → 本模型能表达的类别；**表达不了就返回 ``None``（不参与一致率）**。

    官方表的 5 个取值与实际类别：

    ==================  ==========  ===================================
    表取值              期望类别     说明
    ==================  ==========  ===================================
    ``T1CE (增强)``     T1CE        —
    ``T2WI``            T2          —
    ``T2-Flair``        FLAIR       注意它也含 "T2"，但官方口径是 FLAIR
    ``T1``              **None**    模型没有 T1 类 → 必然判错，不该计入分母
    ``其他``            **None**    权威排除，本来就不该猜
    ==================  ==========  ===================================

    ⚠️ 早期版本的评分写成"预测标签是不是真值标签的子串"，把 ``T1``（250/1735 = 14.4%）
    与 ``其他``（49 = 2.8%）也计入了分母 —— 等于**强加 17.2% 的必然错误**，
    把一致率整体压低约 1/6，还会把"模型没用"这个结论提前坐实。本函数修掉这个偏差。
    """
    low = re.sub(r"[\s\-_（）()]+", "", str(label)).lower()
    if low.startswith("t1ce") or low.startswith("t1c"):
        return "T1CE"
    if low.startswith("t2flair") or low.startswith("flair"):
        return "FLAIR"
    if low.startswith("t2"):
        return "T2"
    return None


def main(argv: list[str]) -> int:
    root = Path(argv[0]).expanduser() if argv else Path(DEFAULT_ROOT)
    limit = (int(argv[1]) if len(argv) > 1
             else int(os.environ.get("GLIOMA_VOXEL_MAX", "60")))

    print(f"[1/5] 数据根: {root} | 存在={root.is_dir()}", flush=True)
    print(f"      体素模型: {describe()}", flush=True)
    print(f"      最多比对: {limit} 例（第二个参数或 GLIOMA_VOXEL_MAX 可改）", flush=True)

    try:
        table = _table_in(root)
        print(f"[2/5] 序列表: {table}", flush=True)
        if table is None:
            print("!! 数据根下没找到 SeriesType.xlsx", flush=True)
            return 2
        rows = _read_rows(table)
        print(f"      表条数: {len(rows)}", flush=True)

        # 索引：(检查号归一, 序列号归一) → 文件。两个关键点：
        #   ① **绝不用 glob 拼 uid** —— uid 里可能含 `*`（验证集实测就有 `*2.25.…*`），
        #      那会被当成通配符、匹配到别的文件；
        #   ② 表的键经 ``_norm``（去空白 + casefold），而磁盘目录名可能大小写/空格不同，
        #      所以**两侧都归一**后再比。这也顺手兼容 2 层 / 3 层任意目录层数。
        index: dict[tuple[str, str], Path] = {}
        n_files = 0
        for acc_dir in root.iterdir():
            if not acc_dir.is_dir():
                continue
            acc_key = _norm(acc_dir.name)
            for path in acc_dir.rglob("*"):
                if not path.is_file():
                    continue
                low = path.name.lower()
                if not low.endswith((".nii", ".nii.gz")):
                    continue
                n_files += 1
                stem = path.name[:-7] if low.endswith(".nii.gz") else path.stem
                for name in {_norm(stem), _norm(path.parent.name)}:
                    index.setdefault((acc_key, name), path)
        print(f"[3/5] 磁盘影像数: {n_files}   索引键数: {len(index)}", flush=True)

        model = load_model()
        hits = tot = miss = skipped_label = 0
        margins: list[float] = []
        per_class: dict[str, list[int]] = {}          # 期望类别 -> [命中数, 总数]
        cm: dict[str, dict[str, int]] = {}            # 期望类别 -> {预测类别: 次数}
        for (acc, uid), label in itertools.islice(rows.items(), 0, 8000):
            want = expected_class(label)
            if want is None:                          # 模型没有这个类别 → 不计入分母
                skipped_label += 1
                continue
            path = index.get((acc, uid))
            if path is None:
                miss += 1
                if miss <= 3:                         # 打前 3 个未命中，暴露键口径差异
                    print(f"      [miss] acc={acc!r} uid={uid!r}", flush=True)
                continue
            try:
                vol = np.squeeze(
                    np.asanyarray(nib.load(str(path)).dataobj)).astype(np.float32)
            except Exception as exc:                                  # noqa: BLE001
                print(f"      [skip] {uid[:20]}: {type(exc).__name__}: {exc}", flush=True)
                continue
            lab, probs = model.predict(vol)
            ranked = sorted(probs.values(), reverse=True)
            margins.append(ranked[0] - ranked[1])     # top-2 间隔：判断概率是否饱和
            hit = lab.upper() == want                 # 类别必须**相等**，不是子串包含
            tot += 1
            hits += hit
            bucket = per_class.setdefault(want, [0, 0])
            bucket[0] += hit
            bucket[1] += 1
            cm.setdefault(want, {})
            cm[want][lab.upper()] = cm[want].get(lab.upper(), 0) + 1
            if tot <= 12:
                print(f"  {uid[:22]:<24} 表={str(label):<14}->{want:<5} 判别={lab:<6} "
                      f"top={ranked[0]:.3f} 次优={ranked[1]:.3f} {'OK' if hit else 'X'}",
                      flush=True)
            if tot >= limit:
                break

        print(f"[4/5] 参与比对 {tot} 例；"
              f"标签模型表不出而跳过 {skipped_label} 行（T1 / 其他）；"
              f"磁盘找不到文件跳过 {miss} 行", flush=True)
        print(f"      一致率: {hits}/{tot} = {hits / max(1, tot):.1%}"
              f"（随机基线 {1 / len(MODEL_CLASSES):.0%}）", flush=True)
        if margins:
            m = np.asarray(margins)
            print(f"      top-2 间隔：中位数={np.median(m):.3f} 最小={m.min():.3f} "
                  f"饱和(<0.01)占比={float((m < 0.01).mean()):.1%}", flush=True)
        if per_class:
            print("      各期望类别召回（看是哪一类在拖后腿）：", flush=True)
            for label, (h, n) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
                print(f"        {label:<8} {h}/{n} = {h / max(1, n):.1%}", flush=True)
        if cm:
            print("      混淆矩阵（行=期望，列=预测）：", flush=True)
            print("           " + "".join(f"{c:>8}" for c in MODEL_CLASSES) + "    合计", flush=True)
            for want in MODEL_CLASSES:
                row = cm.get(want)
                if not row:
                    continue
                total_row = sum(row.values())
                print(f"        {want:<8}" + "".join(f"{row.get(c, 0):>8}" for c in MODEL_CLASSES)
                      + f"{total_row:>8}", flush=True)
        print("[5/5] 判据：一致率 >=80% 且 饱和占比 <20% → 可覆盖「其他」；"
              "一致率 <50%（≈随机）→ 不该覆盖", flush=True)
        return 0
    except Exception:
        print("!!!! 脚本中途异常，下面是完整回溯 —— 请整段贴回来 !!!!", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
