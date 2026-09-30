"""共享骨干的**单次前向 + 逐 Study 复用**。

问题背景（真实踩到的坑）：
本工程是单骨干多任务，但按规范每个 Goal 都是独立插件、各自实现 ``predict()``。
若每个 Goal 都自己跑一次骨干前向，同一个 Study 就会被前向 **4 次**
（Goal1 / Goal2-stitched / Goal3 / Goal4），叠加多折集成与 TTA 后，
Mock Competition 直接超时（实测 900s 未回调）。

而规范 §8 的 ``PipelineContext`` 本就是**逐 Study 共享**的对象，
且它自带 ``diagnostics`` 字段用于承载诊断信息。因此这里把"骨干前向结果"
缓存在 ``context.diagnostics`` 上：

- 生命期与 context 完全一致（Study 处理完即随之释放，**不会造成显存泄漏**）；
- 第一个调用者真正计算，其余 Goal 直接复用；
- Goal5 用滑窗、不走全局视图，因此**不参与共享**（它的前向本就不同）。

这样既保持"每个 Goal 独立插件"的结构，又拿回"一次前向服务多个头"的性能。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from tasks._common.volume import build_volume, global_view

#: 缓存在 context.diagnostics 中的键名前缀。
#:
#: ⚠️ **必须按权重区分**（见 :meth:`BackboneRunner.cache_key`）：
#: 规范 §5.2 为每个 Goal 规定了**各自的**权重文件
#: （``goal1_authenticity/model.pt``、``goal3_tumor/model.pt``…），
#: 因此同一个 Study 上并存着"多个不同权重的前向结果"。
#: 早期实现用固定键 ``"_backbone_out"``，导致先跑的 Goal 把结果写进缓存后，
#: 后续 Goal 直接命中并返回**别人的权重**算出的输出——而且完全不报错。
#: 若任务只训练了自己那一路（其余头随机初始化），后果是该 Goal 输出纯噪声；
#: 即便各权重都有全头，也会静默换成另一个模型的预测。
CACHE_KEY = "_backbone_out"

#: 进程级**权重共享**缓存：``share_key -> (members, cls_spec, global_size, global_size_mm)``。
#:
#: 为什么必须有它（**这是"评测一启动容器就 OOM"的直接放大器**）：
#: 每个 ``SingleHeadStudyTask`` 都各建一个 :class:`BackboneRunner`，而
#: ``SingleHeadStudyTask.ckpt_rel`` 的**默认值就是同一个文件**
#: （``goal5_segmentation/core.pt``）—— 于是 goal1 / goal2_stitched / goal3 / goal4
#: 四条任务会把**同一份权重加载 4 遍**，各自持有一份常驻模型。
#: 再乘上多折集成的份数，常驻模型数就是 ``任务数 × 集成份数``。
#:
#: 这些模型是**只读**的（``eval()`` + ``torch.inference_mode()``），共享完全安全，
#: 与 ``context.diagnostics`` 上那个"前向结果缓存"是两件事：
#: 那个缓存复用**计算**，本缓存复用**权重本身**。
_SHARED_WEIGHTS: dict[tuple, tuple] = {}


def _share_key(ckpt_root, ckpt_rel: str, in_channels: int, device: str) -> tuple:
    from pathlib import Path

    return (str(Path(ckpt_root).expanduser().resolve()), ckpt_rel, in_channels, device)


def shared_weight_count() -> int:
    """已缓存的权重份数（诊断用；正常应 = 不同 ``ckpt_rel`` 的个数）。"""
    return len(_SHARED_WEIGHTS)


class BackboneRunner:
    """按 checkpoint 加载共享骨干，并提供"逐 Study 缓存"的前向。"""

    def __init__(self, ckpt_root, ckpt_rel: str, in_channels: int = 4,
                 device: str = "cuda", global_size: int = 96,
                 global_size_mm: float = 192.0,
                 common_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                 tta_flips: tuple[str, ...] = ("x", "y"), tta_batch: int = 2) -> None:
        from pathlib import Path

        self.ckpt_root = Path(ckpt_root)
        self.ckpt_rel = ckpt_rel
        self.in_channels = in_channels
        self.device = device
        self.global_size = global_size
        self.global_size_mm = global_size_mm
        self.common_spacing = common_spacing
        self.tta_flips = tta_flips
        self.tta_batch = tta_batch
        self._members: list = []
        self._cls_spec: list = []
        self.loaded = False

    # ------------------------------------------------------------------ #
    @property
    def cache_key(self) -> str:
        """本 runner 的前向缓存键（**含权重标识**）。

        只有 ``ckpt_rel`` 相同的两个 Task 才共享缓存——此时它们确实是
        同一个模型，复用是安全的；权重不同则各算各的（正确性优先于速度）。

        实践含义：把各 Goal 的权重都指向**同一个多任务权重**时，缓存
        依然全程命中（一次前向服务全部头），性能不受影响。
        """
        return f"{CACHE_KEY}::{self.ckpt_rel}"

    def _check_heads(self, name: str, missing: list[str]) -> None:
        """校验权重是否提供了本工程依赖的输出头。

        这里只拦"结构性缺失"（权重里根本没有这些参数），不判断精度：
        缺 ``cls_heads`` 时 ``load_state_dict(strict=False)`` 会把它归入
        ``unexpected_keys`` 而**不报错**，随后推理侧只能拿到未训练的随机
        分类头——一个会静默拉低指标的失效模式，必须在加载阶段就暴露出来。
        """
        groups = {
            "special": ("special",),              # 目标一/二-A 的分类头
            "embed": ("embed_head",),             # 目标二-B 的嵌入头
        }
        bad = [g for g, prefixes in groups.items()
               if any(k.startswith(prefixes) for k in missing)]
        if bad:
            raise ValueError(
                f"{name}: 权重缺少输出头 {bad}（推理依赖 special/embed）。"
                f"请确认这份权重是用**多任务**训练流程产出的；"
                f"只训练单一路的研发权重无法直接用于提交推理。"
            )

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        """加载权重（服务启动时一次）。多折时加载全部成员做集成。

        **同 ``(ckpt_root, ckpt_rel, in_channels, device)`` 的权重在整个进程内只加载一次**，
        后续任务直接复用同一批模型对象（见 ``_SHARED_WEIGHTS``）——
        否则 goal1/2/3/4 各持一份相同权重的副本，常驻内存直接 ×4。
        """
        import torch

        from tasks._common.factory import build_shared_backbone
        from tasks.goal5_segmentation.inference import resolve_ckpts

        key = _share_key(self.ckpt_root, self.ckpt_rel, self.in_channels, self.device)
        cached = _SHARED_WEIGHTS.get(key)
        if cached is not None:
            self._members, self._cls_spec, self.global_size, self.global_size_mm = cached
            self.loaded = True
            print(f"[ckpt] 复用已加载权重（{self.ckpt_rel}，进程内共 "
                  f"{len(self._members)} 个成员，未重复读盘）", flush=True)
            return

        paths = resolve_ckpts(self.ckpt_root, self.ckpt_rel)
        self._members = []
        for p in paths:
            ck = torch.load(str(p), map_location="cpu", weights_only=False)
            mc = ck.get("model_cfg") or {}
            in_ch = int(mc.get("in_ch", self.in_channels))
            if in_ch != self.in_channels:
                raise ValueError(f"{p.name}: in_ch={in_ch} 与配置 {self.in_channels} 不一致")
            m = build_shared_backbone(ck, None, self.in_channels)
            state = ck.get("model_ema") or ck.get("model") or ck
            missing, _ = m.load_state_dict(state, strict=False)
            core_missing = [k for k in missing
                            if k.startswith(("enc", "dec", "bottleneck", "stem"))]
            if core_missing:
                raise ValueError(f"{p.name}: 骨干缺失 {len(core_missing)} 层：{core_missing[:3]}")
            self._check_heads(p.name, missing)

            # cls_spec ⇄ cls_heads 必须同时存在。
            # 训练侧保存 ``cls_spec`` 时若导入失败（旧实现用裸 ``import model``），
            # 会**静默存成空列表**；而 ``build_shared_backbone`` 依 spec 建头，
            # spec 丢了就会建出空 ModuleList，权重里的分类头全部变成
            # unexpected_keys 被丢弃，推理侧再拿不到任何字段预测。
            spec = ck.get("cls_spec") or []
            cls_in_state = [k for k in state if k.startswith("cls_heads.")]
            if cls_in_state and not spec:
                raise ValueError(
                    f"{p.name}: 权重含 {len(cls_in_state)} 个 cls_heads 参数但未记录 "
                    f"cls_spec（头与字段无法对应，顺序错位会静默给出错误结论）"
                )
            if spec and not cls_in_state:
                raise ValueError(
                    f"{p.name}: 记录了 {len(spec)} 个 cls_spec 但权重中没有 cls_heads 参数"
                )
            self._cls_spec = spec or self._cls_spec
            self.global_size = int(ck.get("global_size", self.global_size))
            self.global_size_mm = float(ck.get("global_size_mm", self.global_size_mm))
            dev = self.device if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
            self._members.append(m.eval().to(dev))
            del ck
        _SHARED_WEIGHTS[key] = (self._members, self._cls_spec,
                                self.global_size, self.global_size_mm)
        print(f"[ckpt] 加载权重 {self.ckpt_rel}：{len(self._members)} 个成员"
              f"（存入进程级共享缓存，后续同权重任务直接复用）", flush=True)
        self.loaded = True

    # ------------------------------------------------------------------ #
    def forward(self, study, context) -> dict[str, Any]:
        """返回 ``{special:[2], cls:[...], embed:[E]}``（numpy，已跨折平均）。

        同一 context 内重复调用只算一次（**同一权重**才复用，见 ``cache_key``）。
        """
        cached = (context.diagnostics or {}).get(self.cache_key)
        if cached is not None:
            return cached
        if not self.loaded:
            self.load()

        import numpy as np
        import torch

        from tasks._common.sliding import _forward_batch, _tta_combos

        prepared = build_volume(study, self.common_spacing)
        if prepared.missing:
            context.warnings.append(f"缺通道 {list(prepared.missing)}（已零占位）")
        g = global_view(prepared.volume, size_mm=self.global_size_mm, out=self.global_size)

        if not self._members:
            raise RuntimeError(f"{self.ckpt_rel}: 没有已加载的模型成员（load() 未成功）")
        combos = _tta_combos(tuple(self.tta_flips))
        # 注意：这里必须用**已加载成员**取设备，不能引用 load() 里的循环变量
        dev = next(self._members[0].parameters()).device
        x = torch.from_numpy(np.ascontiguousarray(g))[None].to(dev, torch.float32)
        with torch.inference_mode():
            out = _forward_batch(self._members, x, combos, torch.bfloat16,
                                 tta_batch=int(self.tta_batch))

        result = {
            "special": np.asarray(out["special"].float().cpu().numpy(),
                                  dtype=np.float64).ravel(),
            "cls": [np.asarray(c.float().cpu().numpy(), dtype=np.float64) for c in out["cls"]],
            "embed": np.asarray(out["embed"].float().cpu().numpy(),
                                dtype=np.float32).ravel(),
            "cls_spec": list(self._cls_spec),
            "missing_channels": list(prepared.missing),
        }
        del out, x, g, prepared
        context.diagnostics[self.cache_key] = result
        return result
