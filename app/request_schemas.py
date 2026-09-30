"""比赛请求 / 响应模型（规范 §13.1）。

从 ``app/server.py`` 抽出，便于契约测试、OpenAPI 文档与错误定位。
字段名严格按规范：``request_id``、``input.evaluation_id``、``input.dataset_path``。

**不在这里做业务校验**（如 dataset_path 是否存在）：那是 Runner 的职责。
schema 只负责"结构正确"，混入业务判断会让 400 与 500 的边界变得含糊。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class CallInput(BaseModel):
    evaluation_id: str = Field(..., description="输出隔离键")
    dataset_path: str = Field(..., description="本次评测的数据集根目录")


class CallRequest(BaseModel):
    request_id: str | None = Field(None, description="请求追踪键；缺省时服务端生成")
    team_id: str | None = None
    track_code: str | None = None
    input: CallInput


class CallResponse(BaseModel):
    status: str = "accepted"
    request_id: str | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    status: str = "active"
