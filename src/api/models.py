from __future__ import annotations
from enum import StrEnum
from pydantic import BaseModel, field_validator


class TaskStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchRequest(BaseModel):
    project: str
    query: str
    context: str | None = None
    source: str | None = None
    api_key: str | None = None


class ProgressInfo(BaseModel):
    stage: str
    message: str


class OutputFile(BaseModel):
    name: str
    url: str
    type: str
    size: int


class SourceInfo(BaseModel):
    type: str
    name: str | None = None
    url: str | None = None
    title: str | None = None
    calls: int | None = None


class OutputInfo(BaseModel):
    format: str
    files: list[OutputFile]
    summary: str
    sources: list[SourceInfo]


class UsageInfo(BaseModel):
    model: str
    total_tokens: int
    research_time_seconds: float


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: ProgressInfo | None = None
    output: OutputInfo | None = None
    usage: UsageInfo | None = None
    created_at: str
    updated_at: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ProjectInfo(BaseModel):
    name: str
    description: str
    apis: int


class ProjectListResponse(BaseModel):
    projects: list[ProjectInfo]


class HealthResponse(BaseModel):
    status: str
    openai_api: str
    active_tasks: int
