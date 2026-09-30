"""Goal2-B 的推理入口：嵌入提取。

组合逻辑在 ``task.py`` 的 ``update()`` 中（因为它是 DatasetTask，生命周期不同），
本文件提供可独立测试的嵌入提取函数，便于契约测试与离线评估复用。
"""
from __future__ import annotations

import numpy as np

from tasks._common.factory import build_shared_backbone  # noqa: F401
from tasks._common.volume import build_volume, global_view


def extract_embedding(model, study, device: str = "cpu", global_size: int = 96,
                      global_size_mm: float = 192.0,
                      common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """对单个 Study 提取 L2 归一化嵌入（numpy CPU 数组，便于长期持有）。"""
    import torch

    prepared = build_volume(study, common_spacing)
    g = global_view(prepared.volume, size_mm=global_size_mm, out=global_size)
    x = torch.from_numpy(np.ascontiguousarray(g))[None].to(device, torch.float32)
    with torch.inference_mode():
        out = model(x)
    emb = out["embed"].float().cpu().numpy().ravel().astype(np.float32)
    n = float(np.linalg.norm(emb))
    return emb / n if n > 1e-6 else emb
