"""pytest 内存探针：**直接把"导入成本"和"真泄漏"分开**，最后给一个明确结论。

为什么需要它：RSS 从 35 MB 涨到 374 MB 看起来像泄漏，但那 339 MB 是
``import torch/numpy`` 的**一次性**成本 —— 每个测试进程都会付，且不随测试数增长。
真正要区分的是：

* **一次性导入成本**：前几个测试处的一次跳变，之后不再增长；
* **真泄漏**：后一半测试的平均增量**持续高于**前一半。

本探针把基线定在**收集完成之后**（导入已发生），再比较前后半段的增量趋势，
输出 ``无泄漏`` / ``疑似泄漏``。同时统计**模型加载次数**（每次 = 一份常驻模型）。

用法::

    python -m pytest tests -p scripts.memprobe -q

输出末尾会给：
  - 模型加载次数（0 次说明测试根本没加载权重）
  - 基线 / 峰值 / 结束 RSS
  - 前后半段平均增量 + 结论
"""
from __future__ import annotations

import gc
import os

try:
    import psutil

    _PROC = psutil.Process(os.getpid())
except Exception:                                                 # noqa: BLE001
    _PROC = None

#: 每个测试的 ``(nodeid, 结束时 RSS, 相对上一测试的增量)``
_rows: list[tuple[str, float, float]] = []
_prev = [0.0]
_base = [0.0]
_loads: list[str] = []
_patched = False

#: 判据阈值：后一半平均增量比前一半高出这么多 MB 才算"疑似泄漏"
_LEAK_MB_PER_TEST = 3.0


def _rss() -> float:
    return _PROC.memory_info().rss if _PROC is not None else 0.0


def _install_counters() -> None:
    """给"会吃内存"的入口打计数桩 —— 加载几次权重是 OOM 排查的第一问。"""
    global _patched
    if _patched:
        return
    _patched = True
    try:
        import torch

        from tasks._common.backbone_runner import BackboneRunner

        _orig_load = BackboneRunner.load

        def counted_load(self) -> None:                           # noqa: ANN001
            _loads.append(f"BackboneRunner.load({self.ckpt_rel})")
            return _orig_load(self)

        BackboneRunner.load = counted_load                        # type: ignore[method-assign]

        _orig_torch_load = torch.load

        def counted_torch_load(*a, **kw):                         # noqa: ANN002, ANN003
            _loads.append(f"torch.load({os.path.basename(str(a[0]))})")
            return _orig_torch_load(*a, **kw)

        torch.load = counted_torch_load                           # type: ignore[assignment]
    except Exception as exc:                                      # noqa: BLE001
        print(f"\n[memprobe] 计数桩安装失败（不影响内存测量）: {exc}")


def pytest_sessionstart(session) -> None:                         # noqa: ANN001
    gc.collect()
    _prev[0] = _rss()
    if _PROC is None:
        print("\n[memprobe] !! 没有 psutil → 无法测量（pip install psutil）")


def pytest_collection_finish(session) -> None:                     # noqa: ANN001
    """基线定在**收集完成之后**：此时 torch/numpy 等一次性导入已经发生。

    这是本探针与"只看首尾差"的关键区别 —— 后者会把导入成本误报成泄漏。
    """
    if _PROC is None:
        return
    gc.collect()
    _orig = _prev[0]
    _base[0] = _rss()
    _prev[0] = _base[0]
    print(f"\n[memprobe] 起始 RSS = {_orig / 1e6:.0f} MB → "
          f"**基线（导入完成）= {_base[0] / 1e6:.0f} MB**"
          f"（差 {( _base[0] - _orig) / 1e6:.0f} MB 是 import torch/numpy 的一次性成本）")
    print(f"[memprobe] COMPETITION_CHECKPOINT_ROOT="
          f"{os.environ.get('COMPETITION_CHECKPOINT_ROOT')!r}")
    _install_counters()


def pytest_runtest_teardown(item, nextitem) -> None:              # noqa: ANN001
    if _PROC is None:
        return
    gc.collect()
    cur = _rss()
    _rows.append((item.nodeid, cur, cur - _prev[0]))
    _prev[0] = cur


def pytest_sessionfinish(session, exitstatus) -> None:            # noqa: ANN001
    if _PROC is None or not _rows:
        return
    from collections import Counter

    peak = max(r[1] for r in _rows)
    end = _rss()

    print("\n" + "=" * 92)
    print("模型加载次数（每次 torch.load + 建网 = 一份常驻模型）")
    print("=" * 92)
    if _loads:
        for k, v in Counter(_loads).most_common():
            print(f"  {v:>3}×  {k}")
        print(f"  合计 {len(_loads)} 次")
    else:
        print("  **0 次** —— 测试期间没有加载任何权重（OOM 不可能来自测试里的模型）")

    print("\n" + "=" * 92)
    print("逐测试 RSS 增量（降序 top 10）")
    print("=" * 92)
    for nid, cur, d in sorted(_rows, key=lambda r: -r[2])[:10]:
        print(f"  +{d / 1e6:8.1f} MB   累计 {cur / 1e6:8.1f} MB   {nid}")

    # ---- 前后半段趋势：这是"真泄漏"与"一次性成本"的分水岭 ----
    n = len(_rows)
    half = max(1, n // 2)
    # 从第二个测试开始算：第一个测试可能还带着剩余导入
    deltas = [d for _, _, d in _rows[1:]]
    first = deltas[:half] or [0.0]
    second = deltas[half:] or [0.0]
    avg1, avg2 = sum(first) / len(first), sum(second) / len(second)
    growth = end - _base[0]

    print("\n" + "=" * 92)
    print("结论")
    print("=" * 92)
    print(f"  基线（导入完成）      {_base[0] / 1e6:8.0f} MB")
    print(f"  峰值                  {peak / 1e6:8.0f} MB")
    print(f"  结束                  {end / 1e6:8.0f} MB")
    print(f"  基线→结束净增长        {growth / 1e6:+8.1f} MB（{n} 个测试）")
    print(f"  前半段平均增量        {avg1 / 1e6:+8.2f} MB/测试")
    print(f"  后半段平均增量        {avg2 / 1e6:+8.2f} MB/测试")

    if avg2 - avg1 > _LEAK_MB_PER_TEST * 1e6:
        print(f"\n  **疑似泄漏**：后半段比前半段平均多涨 "
              f"{(avg2 - avg1) / 1e6:.2f} MB/测试 → 某个测试在持续累积")
        print("  定位方法：看上面 top10 里**按顺序靠后**且增量大的那个测试")
    else:
        print(f"\n  **无泄漏**：后半段增量未超过前半段 "
              f"{_LEAK_MB_PER_TEST} MB/测试的阈值（{'%.2f' % ((avg2 - avg1) / 1e6)} MB/测试）")
        if growth > 20e6:
            print(f"  基线→结束净增长 {growth / 1e6:.0f} MB 属于正常的解释器/缓存碎片，"
                  f"不随测试数增长。")
    print("=" * 92)
