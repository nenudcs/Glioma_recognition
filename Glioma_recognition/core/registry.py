from __future__ import annotations

import importlib

from pipeline.inference import InferencePipeline


def build_pipeline(factory_path: str | None) -> InferencePipeline:
    """Load ``module:function`` when real task plugins replace the baseline."""
    if not factory_path:
        # ⚠️ 这段日志必须留着：没有它，Dummy 基线是**静默生效**的 ——
        #    服务照常起来、/health 通、回调也正常，但答案是占位内容。
        #    等到发现"服务没报错却几乎不得分"时，已经很难定位到只是少了一个环境变量。
        print(
            "[registry] ⚠️ 未设置 COMPETITION_PIPELINE_FACTORY → 当前跑的是 **Dummy 基线**"
            "（接口格式正确，但不含任何真实模型输出）。\n"
            "[registry]    正式测评请设置，例如：\n"
            "[registry]      COMPETITION_PIPELINE_FACTORY=tasks.real_pipeline:build_pipeline"
            "   # 目标架构（自包含）\n"
            "[registry]      COMPETITION_PIPELINE_FACTORY=tasks.glioma.pipeline:build_pipeline"
            "   # 过渡桥接（用 glioma_track4 的权重）",
            flush=True,
        )
        return InferencePipeline()
    module_name, separator, function_name = factory_path.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(
            "COMPETITION_PIPELINE_FACTORY must use the form ''module:function''"
        )
    factory = getattr(importlib.import_module(module_name), function_name)
    pipeline = factory()
    if not isinstance(pipeline, InferencePipeline):
        raise TypeError(f"{factory_path} did not return InferencePipeline")
    print(f"[registry] ✓ 真实插件工厂已加载: {factory_path}", flush=True)
    return pipeline
