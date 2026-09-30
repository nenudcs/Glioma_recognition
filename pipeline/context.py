from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from data.structures import Study
from tasks.results import (
    Goal1Result,
    Goal3Result,
    Goal4Result,
    Goal5Result,
    StitchedResult,
)


@dataclass
class PipelineContext:
    study: Study
    goal1: Goal1Result | None = None
    goal2_stitched: StitchedResult | None = None
    goal3: Goal3Result | None = None
    goal4: Goal4Result | None = None
    goal5: Goal5Result | None = None
    processing_time_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
