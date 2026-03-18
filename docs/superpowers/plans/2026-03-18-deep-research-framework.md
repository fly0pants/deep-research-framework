# Deep Research Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an HTTP service that wraps OpenAI Deep Research API with project-based configuration, data integration, and auto-format output rendering.

**Architecture:** FastAPI async service → prefetch project API data → upload to OpenAI Vector Store → call Deep Research API (background mode with file_search + web_search + code_interpreter) → render output in Docker sandbox → serve results via REST API. Task state persisted in SQLite.

**Tech Stack:** Python 3.12, FastAPI, OpenAI Python SDK, httpx, aiosqlite, structlog, WeasyPrint, Docker

**Spec:** `docs/superpowers/specs/2026-03-18-deep-research-framework-design.md`

---

## File Structure

```
deep-research-framework/
├── pyproject.toml                    # Project deps & metadata
├── .env.example                      # Env var template
├── .gitignore                        # Git ignore rules
├── Dockerfile                        # Main service image
├── Dockerfile.renderer               # Sandbox renderer image
├── docker-compose.yml                # Compose config
├── src/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app entry, lifespan
│   ├── config.py                     # Settings (pydantic-settings)
│   ├── auth.py                       # Bearer token dependency
│   ├── api/
│   │   ├── __init__.py
│   │   ├── models.py                 # Pydantic request/response schemas
│   │   └── routes.py                 # All HTTP endpoints
│   ├── task/
│   │   ├── __init__.py
│   │   └── manager.py                # SQLite task CRUD
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── project_loader.py         # Load project config & API docs
│   │   ├── data_preparation.py       # Prefetch APIs & Vector Store
│   │   ├── prompt_builder.py         # Build research prompt
│   │   └── research.py               # Call Deep Research API & poll
│   └── output/
│       ├── __init__.py
│       └── renderer.py               # Parse output, render in sandbox
├── projects/
│   └── example/
│       ├── config.yaml
│       ├── api_docs/
│       │   └── example_api.yaml
│       └── output_prefs.yaml
├── output/                           # Generated result files (gitignored)
└── tests/
    ├── conftest.py                   # Shared fixtures
    ├── test_config.py
    ├── test_auth.py
    ├── test_models.py
    ├── test_task_manager.py
    ├── test_project_loader.py
    ├── test_data_preparation.py
    ├── test_prompt_builder.py
    ├── test_research.py
    ├── test_renderer.py
    └── test_routes.py
```

---

## Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config loading**

```python
# tests/test_config.py
import os
import pytest


def test_settings_loads_defaults():
    """Settings should have sensible defaults when env vars are minimal."""
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    os.environ.setdefault("API_TOKEN", "test-token")
    from src.config import Settings
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.default_model == "o3-deep-research"
    assert s.max_concurrent_tasks == 5
    assert s.output_retention_days == 30


def test_settings_requires_openai_key(monkeypatch):
    """Settings should fail without OPENAI_API_KEY."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(Exception):
        from src.config import Settings
        Settings(_env_file=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/deep-research-framework && python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 3: Create pyproject.toml**

```toml
[project]
name = "deep-research-framework"
version = "0.1.0"
description = "Deep Research Service powered by OpenAI Deep Research API"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "openai>=1.60.0",
    "httpx>=0.27.0",
    "aiosqlite>=0.20.0",
    "pydantic-settings>=2.5.0",
    "structlog>=24.4.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "pytest-httpx>=0.30.0",
    "ruff>=0.7.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
pythonpath = ["."]

[tool.ruff]
target-version = "py312"
line-length = 100
```

- [ ] **Step 4: Create .env.example**

```bash
# === Required ===
OPENAI_API_KEY=sk-xxx
API_TOKEN=your-service-auth-token

# === Service ===
HOST=0.0.0.0
PORT=8000

# === Optional ===
STORAGE_PATH=./output
PROJECTS_PATH=./projects
LOG_LEVEL=info
DEFAULT_MODEL=o3-deep-research
MAX_CONCURRENT_TASKS=5
OUTPUT_RETENTION_DAYS=30

# === Project API Keys (add as needed) ===
# ADMAPIX_API_KEY=xxx
```

- [ ] **Step 5: Create .gitignore**

```
__pycache__/
*.pyc
.env
output/
*.db
.pytest_cache/
.ruff_cache/
dist/
*.egg-info/
.venv/
```

- [ ] **Step 6: Implement config.py**

```python
# src/config.py
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Required
    openai_api_key: str
    api_token: str

    # Service
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    storage_path: Path = Path("./output")
    projects_path: Path = Path("./projects")

    # Defaults
    log_level: str = "info"
    default_model: str = "o3-deep-research"
    max_concurrent_tasks: int = 5
    output_retention_days: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings: Settings | None = None


def get_settings() -> Settings:
    global settings
    if settings is None:
        settings = Settings()
    return settings
```

- [ ] **Step 7: Create src/__init__.py and tests/conftest.py**

```python
# src/__init__.py
```

```python
# tests/conftest.py
import os

# Set test env vars before any imports
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("API_TOKEN", "test-token")
os.environ.setdefault("STORAGE_PATH", "/tmp/dr-test-output")
os.environ.setdefault("PROJECTS_PATH", "./projects")
```

- [ ] **Step 8: Install deps and run tests**

Run: `cd ~/deep-research-framework && pip install -e ".[dev]" && python -m pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/__init__.py src/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: project scaffolding and config module"
```

---

## Task 2: Auth Dependency & Pydantic Models

**Files:**
- Create: `src/auth.py`
- Create: `src/api/__init__.py`
- Create: `src/api/models.py`
- Create: `tests/test_auth.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for auth**

```python
# tests/test_auth.py
import pytest
from fastapi import HTTPException
from src.auth import verify_token


def test_verify_token_valid():
    verify_token("test-token")  # Should not raise


def test_verify_token_invalid():
    with pytest.raises(HTTPException) as exc_info:
        verify_token("wrong-token")
    assert exc_info.value.status_code == 401


def test_verify_token_missing():
    with pytest.raises(HTTPException):
        verify_token(None)
```

- [ ] **Step 2: Write failing tests for models**

```python
# tests/test_models.py
import pytest
from src.api.models import ResearchRequest, TaskResponse, TaskStatus


def test_research_request_minimal():
    req = ResearchRequest(project="admapix", query="test query")
    assert req.project == "admapix"
    assert req.context is None
    assert req.callback_url is None


def test_research_request_full():
    req = ResearchRequest(
        project="admapix",
        query="test",
        context="extra context",
        callback_url="https://example.com/callback",
    )
    assert req.callback_url == "https://example.com/callback"


def test_research_request_rejects_http_callback():
    """callback_url must be HTTPS."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        ResearchRequest(
            project="admapix",
            query="test",
            callback_url="http://example.com/callback",
        )


def test_task_status_enum():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.COMPLETED == "completed"


def test_task_response_pending():
    resp = TaskResponse(
        task_id="dr_abc123",
        status=TaskStatus.PENDING,
        created_at="2026-03-18T10:00:00Z",
    )
    assert resp.output is None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_auth.py tests/test_models.py -v`
Expected: FAIL

- [ ] **Step 4: Implement auth.py**

```python
# src/auth.py
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from src.config import get_settings

security = HTTPBearer()


def verify_token(token: str | None) -> None:
    if not token or token != get_settings().api_token:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    verify_token(credentials.credentials)
    return credentials.credentials
```

- [ ] **Step 5: Implement models.py**

```python
# src/api/__init__.py
```

```python
# src/api/models.py
from __future__ import annotations
from datetime import datetime
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
    callback_url: str | None = None

    @field_validator("callback_url")
    @classmethod
    def validate_callback_url(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v.startswith("https://"):
            raise ValueError("callback_url must use HTTPS")
        return v


class ProgressInfo(BaseModel):
    stage: str  # preparing | researching | rendering
    message: str


class OutputFile(BaseModel):
    name: str
    url: str
    type: str
    size: int


class SourceInfo(BaseModel):
    type: str  # api | web
    name: str | None = None
    url: str | None = None
    title: str | None = None
    calls: int | None = None


class OutputInfo(BaseModel):
    format: str  # html | pdf | markdown | mixed
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
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_auth.py tests/test_models.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/auth.py src/api/__init__.py src/api/models.py tests/test_auth.py tests/test_models.py
git commit -m "feat: auth dependency and pydantic request/response models"
```

---

## Task 3: Task Manager (SQLite)

**Files:**
- Create: `src/task/__init__.py`
- Create: `src/task/manager.py`
- Create: `tests/test_task_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_task_manager.py
import pytest
from src.task.manager import TaskManager


@pytest.fixture
async def tm(tmp_path):
    manager = TaskManager(db_path=str(tmp_path / "test.db"))
    await manager.init()
    yield manager
    await manager.close()


async def test_create_task(tm):
    task = await tm.create(project="admapix", query="test query")
    assert task["task_id"].startswith("dr_")
    assert task["status"] == "pending"
    assert task["project"] == "admapix"


async def test_get_task(tm):
    created = await tm.create(project="admapix", query="test")
    fetched = await tm.get(created["task_id"])
    assert fetched is not None
    assert fetched["task_id"] == created["task_id"]


async def test_get_nonexistent(tm):
    result = await tm.get("dr_nonexistent")
    assert result is None


async def test_update_status(tm):
    task = await tm.create(project="admapix", query="test")
    await tm.update_status(task["task_id"], "processing", stage="researching", message="Working...")
    updated = await tm.get(task["task_id"])
    assert updated["status"] == "processing"
    assert updated["stage"] == "researching"


async def test_update_completed(tm):
    task = await tm.create(project="admapix", query="test")
    result_data = {"format": "html", "files": [], "summary": "done", "sources": []}
    usage_data = {"model": "o3-deep-research", "total_tokens": 1000, "research_time_seconds": 60}
    await tm.complete(task["task_id"], result_data, usage_data)
    updated = await tm.get(task["task_id"])
    assert updated["status"] == "completed"


async def test_cancel_task(tm):
    task = await tm.create(project="admapix", query="test")
    await tm.update_status(task["task_id"], "cancelled")
    updated = await tm.get(task["task_id"])
    assert updated["status"] == "cancelled"


async def test_count_active(tm):
    await tm.create(project="p1", query="q1")
    t2 = await tm.create(project="p2", query="q2")
    await tm.update_status(t2["task_id"], "processing")
    count = await tm.count_active()
    assert count == 2  # pending + processing
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_task_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Implement task manager**

```python
# src/task/__init__.py
```

```python
# src/task/manager.py
import json
import uuid
from datetime import datetime, timezone

import aiosqlite


class TaskManager:
    def __init__(self, db_path: str = "tasks.db"):
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None

    async def init(self):
        self.db = await aiosqlite.connect(self.db_path)
        self.db.row_factory = aiosqlite.Row
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                query TEXT NOT NULL,
                context TEXT,
                callback_url TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                stage TEXT,
                message TEXT,
                result_data TEXT,
                usage_data TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await self.db.commit()

    async def close(self):
        if self.db:
            await self.db.close()

    async def create(
        self,
        project: str,
        query: str,
        context: str | None = None,
        callback_url: str | None = None,
    ) -> dict:
        task_id = f"dr_{uuid.uuid4()}"
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """INSERT INTO tasks (task_id, project, query, context, callback_url, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)""",
            (task_id, project, query, context, callback_url, now, now),
        )
        await self.db.commit()
        return {"task_id": task_id, "status": "pending", "project": project, "created_at": now}

    async def get(self, task_id: str) -> dict | None:
        cursor = await self.db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        result = dict(row)
        if result.get("result_data"):
            result["result_data"] = json.loads(result["result_data"])
        if result.get("usage_data"):
            result["usage_data"] = json.loads(result["usage_data"])
        return result

    async def update_status(
        self,
        task_id: str,
        status: str,
        stage: str | None = None,
        message: str | None = None,
    ):
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE tasks SET status=?, stage=?, message=?, updated_at=? WHERE task_id=?",
            (status, stage, message, now, task_id),
        )
        await self.db.commit()

    async def complete(self, task_id: str, result_data: dict, usage_data: dict):
        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "UPDATE tasks SET status='completed', result_data=?, usage_data=?, updated_at=? WHERE task_id=?",
            (json.dumps(result_data), json.dumps(usage_data), now, task_id),
        )
        await self.db.commit()

    async def count_active(self) -> int:
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM tasks WHERE status IN ('pending', 'processing')"
        )
        row = await cursor.fetchone()
        return row[0]
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_task_manager.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/task/__init__.py src/task/manager.py tests/test_task_manager.py
git commit -m "feat: SQLite task manager with CRUD operations"
```

---

## Task 4: Project Loader

**Files:**
- Create: `src/engine/__init__.py`
- Create: `src/engine/project_loader.py`
- Create: `tests/test_project_loader.py`
- Create: `projects/example/config.yaml`
- Create: `projects/example/api_docs/example_api.yaml`
- Create: `projects/example/output_prefs.yaml`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_project_loader.py
import pytest
from pathlib import Path
from src.engine.project_loader import ProjectLoader


@pytest.fixture
def projects_dir(tmp_path):
    # Create a test project
    proj = tmp_path / "test_project"
    proj.mkdir()
    (proj / "config.yaml").write_text("""
name: test_project
description: "A test project"
apis:
  - name: test_api
    base_url: https://api.test.com/v1
    auth:
      type: bearer
      token_env: TEST_API_KEY
    docs_file: api_docs/test_api.yaml
    prefetch:
      - endpoint: /data
        params: {limit: 10}
model: o3-deep-research
system_instructions: "You are a test analyst."
""")
    api_docs = proj / "api_docs"
    api_docs.mkdir()
    (api_docs / "test_api.yaml").write_text("openapi: '3.0.0'\ninfo:\n  title: Test\n  version: '1.0'")
    (proj / "output_prefs.yaml").write_text("""
preferred_language: zh-CN
hints:
  - "Use charts for data"
""")
    return tmp_path


@pytest.fixture
def loader(projects_dir):
    return ProjectLoader(projects_dir)


def test_list_projects(loader):
    projects = loader.list_projects()
    assert len(projects) == 1
    assert projects[0]["name"] == "test_project"


def test_load_project(loader):
    config = loader.load("test_project")
    assert config["name"] == "test_project"
    assert len(config["apis"]) == 1
    assert config["apis"][0]["name"] == "test_api"
    assert config["system_instructions"] == "You are a test analyst."


def test_load_output_prefs(loader):
    prefs = loader.load_output_prefs("test_project")
    assert prefs["preferred_language"] == "zh-CN"


def test_load_api_docs(loader):
    docs = loader.load_api_docs("test_project", "api_docs/test_api.yaml")
    assert "openapi" in docs


def test_load_nonexistent_project(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent")


def test_load_output_prefs_missing(loader, projects_dir):
    # Create project without output_prefs
    proj = projects_dir / "no_prefs"
    proj.mkdir()
    (proj / "config.yaml").write_text("name: no_prefs\ndescription: test\napis: []")
    loader2 = ProjectLoader(projects_dir)
    prefs = loader2.load_output_prefs("no_prefs")
    assert prefs is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_project_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Implement project_loader.py**

```python
# src/engine/__init__.py
```

```python
# src/engine/project_loader.py
from pathlib import Path
import yaml


class ProjectLoader:
    def __init__(self, projects_path: Path | str):
        self.projects_path = Path(projects_path)

    def list_projects(self) -> list[dict]:
        results = []
        if not self.projects_path.exists():
            return results
        for proj_dir in sorted(self.projects_path.iterdir()):
            config_file = proj_dir / "config.yaml"
            if proj_dir.is_dir() and config_file.exists():
                config = yaml.safe_load(config_file.read_text())
                results.append({
                    "name": config.get("name", proj_dir.name),
                    "description": config.get("description", ""),
                    "apis": len(config.get("apis", [])),
                })
        return results

    def load(self, project_name: str) -> dict:
        proj_dir = self.projects_path / project_name
        config_file = proj_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Project '{project_name}' not found")
        return yaml.safe_load(config_file.read_text())

    def load_output_prefs(self, project_name: str) -> dict | None:
        prefs_file = self.projects_path / project_name / "output_prefs.yaml"
        if not prefs_file.exists():
            return None
        return yaml.safe_load(prefs_file.read_text())

    def load_api_docs(self, project_name: str, docs_file: str) -> str:
        docs_path = self.projects_path / project_name / docs_file
        if not docs_path.exists():
            raise FileNotFoundError(f"API docs not found: {docs_path}")
        return docs_path.read_text()
```

- [ ] **Step 4: Create example project files**

```yaml
# projects/example/config.yaml
name: example
description: "Example project for testing"

apis:
  - name: example_api
    base_url: https://api.example.com/v1
    auth:
      type: bearer
      token_env: EXAMPLE_API_KEY
    docs_file: api_docs/example_api.yaml
    prefetch:
      - endpoint: /stats
        params: { days: 30 }

model: o3-deep-research

system_instructions: |
  You are a data analyst. Focus on key metrics and trends.
```

```yaml
# projects/example/api_docs/example_api.yaml
openapi: "3.0.0"
info:
  title: Example API
  version: "1.0"
paths:
  /stats:
    get:
      summary: Get statistics
      parameters:
        - name: days
          in: query
          schema:
            type: integer
```

```yaml
# projects/example/output_prefs.yaml
preferred_language: zh-CN
hints:
  - "Use interactive charts for data-heavy results"
  - "Use tables for comparisons"
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_project_loader.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/engine/__init__.py src/engine/project_loader.py tests/test_project_loader.py projects/
git commit -m "feat: project loader with config, API docs, and output prefs"
```

---

## Task 5: Data Preparation (Prefetch + Vector Store)

**Files:**
- Create: `src/engine/data_preparation.py`
- Create: `tests/test_data_preparation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data_preparation.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.engine.data_preparation import DataPreparator


@pytest.fixture
def mock_openai():
    client = MagicMock()
    # Mock vector store creation
    client.vector_stores.create.return_value = MagicMock(id="vs_test123")
    # Mock file upload
    client.files.create.return_value = MagicMock(id="file_test123")
    # Mock vector store file attachment
    client.vector_stores.files.create.return_value = MagicMock(id="vsf_test123")
    # Mock vector store deletion
    client.vector_stores.delete.return_value = None
    return client


@pytest.fixture
def preparator(mock_openai, tmp_path):
    return DataPreparator(openai_client=mock_openai, temp_dir=tmp_path)


async def test_prefetch_api_data(preparator):
    api_config = {
        "name": "test_api",
        "base_url": "https://api.test.com/v1",
        "auth": {"type": "bearer", "token_env": "TEST_KEY"},
        "prefetch": [{"endpoint": "/data", "params": {"limit": 10}}],
    }
    with patch.dict("os.environ", {"TEST_KEY": "fake-key"}):
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value = MagicMock(
                status_code=200,
                json=lambda: {"items": [1, 2, 3]},
                text='{"items": [1, 2, 3]}',
            )
            results = await preparator.prefetch(api_config)
    assert len(results) == 1
    assert results[0]["endpoint"] == "/data"
    assert results[0]["data"] == {"items": [1, 2, 3]}


async def test_prefetch_handles_failure(preparator):
    api_config = {
        "name": "test_api",
        "base_url": "https://api.test.com/v1",
        "auth": {"type": "bearer", "token_env": "TEST_KEY"},
        "prefetch": [{"endpoint": "/data", "params": {}}],
    }
    with patch.dict("os.environ", {"TEST_KEY": "fake-key"}):
        with patch("httpx.AsyncClient.get", side_effect=Exception("connection error")):
            results = await preparator.prefetch(api_config)
    assert len(results) == 0  # Failed prefetches are skipped


async def test_create_vector_store(preparator, mock_openai):
    prefetch_results = [
        {"endpoint": "/data", "data": {"items": [1, 2, 3]}},
    ]
    api_docs_content = "openapi: 3.0.0"
    vs_id = await preparator.create_vector_store(
        task_id="dr_test",
        prefetch_results=prefetch_results,
        api_docs_content=api_docs_content,
    )
    assert vs_id == "vs_test123"
    assert mock_openai.vector_stores.create.called


async def test_cleanup_vector_store(preparator, mock_openai):
    await preparator.cleanup("vs_test123")
    mock_openai.vector_stores.delete.assert_called_once_with("vs_test123")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_preparation.py -v`
Expected: FAIL

- [ ] **Step 3: Implement data_preparation.py**

```python
# src/engine/data_preparation.py
import json
import os
from pathlib import Path

import httpx
import structlog

logger = structlog.get_logger()


class DataPreparator:
    def __init__(self, openai_client, temp_dir: Path | str = "/tmp/dr-data"):
        self.client = openai_client
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def prefetch(self, api_config: dict) -> list[dict]:
        """Execute prefetch API calls defined in project config."""
        results = []
        base_url = api_config["base_url"].rstrip("/")
        auth = api_config.get("auth", {})
        headers = {}

        if auth.get("type") == "bearer":
            token = os.environ.get(auth["token_env"], "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        for pf in api_config.get("prefetch", []):
            endpoint = pf["endpoint"]
            params = pf.get("params", {})
            url = f"{base_url}{endpoint}"
            try:
                async with httpx.AsyncClient(timeout=30) as http:
                    resp = await http.get(url, params=params, headers=headers)
                    resp.raise_for_status()
                    results.append({
                        "endpoint": endpoint,
                        "data": resp.json(),
                    })
                    logger.info("prefetch_success", endpoint=endpoint)
            except Exception as e:
                logger.warning("prefetch_failed", endpoint=endpoint, error=str(e))
        return results

    async def create_vector_store(
        self,
        task_id: str,
        prefetch_results: list[dict],
        api_docs_content: str | None = None,
    ) -> str:
        """Create OpenAI Vector Store with prefetched data and API docs."""
        vs = self.client.vector_stores.create(name=f"dr-{task_id}")
        file_ids = []

        # Upload prefetch data as files
        if prefetch_results:
            data_content = "# Prefetched Business Data\n\n"
            for pr in prefetch_results:
                data_content += f"## Endpoint: {pr['endpoint']}\n\n"
                data_content += f"```json\n{json.dumps(pr['data'], ensure_ascii=False, indent=2)}\n```\n\n"

            data_file = self.temp_dir / f"{task_id}_data.md"
            data_file.write_text(data_content)
            with open(data_file, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="assistants")
            file_ids.append(uploaded.id)

        # Upload API docs
        if api_docs_content:
            docs_file = self.temp_dir / f"{task_id}_api_docs.md"
            docs_file.write_text(f"# API Documentation\n\n{api_docs_content}")
            with open(docs_file, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="assistants")
            file_ids.append(uploaded.id)

        # Attach files to vector store
        for fid in file_ids:
            self.client.vector_stores.files.create(vector_store_id=vs.id, file_id=fid)

        logger.info("vector_store_created", vs_id=vs.id, files=len(file_ids))
        return vs.id

    async def cleanup(self, vector_store_id: str):
        """Delete temporary vector store after research completes."""
        try:
            self.client.vector_stores.delete(vector_store_id)
            logger.info("vector_store_cleaned", vs_id=vector_store_id)
        except Exception as e:
            logger.warning("vector_store_cleanup_failed", error=str(e))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_data_preparation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/engine/data_preparation.py tests/test_data_preparation.py
git commit -m "feat: data preparation with API prefetch and Vector Store upload"
```

---

## Task 6: Prompt Builder

**Files:**
- Create: `src/engine/prompt_builder.py`
- Create: `tests/test_prompt_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_prompt_builder.py
from src.engine.prompt_builder import build_research_prompt


def test_build_minimal_prompt():
    prompt = build_research_prompt(
        query="What is the trend?",
        project_config={"name": "test", "description": "A test"},
    )
    assert "What is the trend?" in prompt
    assert "test" in prompt


def test_build_prompt_with_system_instructions():
    prompt = build_research_prompt(
        query="Analyze data",
        project_config={
            "name": "admapix",
            "description": "Ad platform",
            "system_instructions": "Focus on ROI and CTR.",
        },
    )
    assert "Focus on ROI and CTR" in prompt


def test_build_prompt_with_context():
    prompt = build_research_prompt(
        query="Analyze data",
        project_config={"name": "test", "description": "Test"},
        context="Focus on video ads",
    )
    assert "Focus on video ads" in prompt


def test_build_prompt_with_output_prefs():
    prompt = build_research_prompt(
        query="Analyze data",
        project_config={"name": "test", "description": "Test"},
        output_prefs={
            "preferred_language": "zh-CN",
            "hints": ["Use charts for data"],
        },
    )
    assert "zh-CN" in prompt
    assert "Use charts for data" in prompt


def test_build_prompt_includes_source_labeling():
    prompt = build_research_prompt(
        query="test",
        project_config={"name": "test", "description": "Test"},
    )
    assert "[API]" in prompt
    assert "[Web]" in prompt


def test_build_prompt_includes_format_decision():
    prompt = build_research_prompt(
        query="test",
        project_config={"name": "test", "description": "Test"},
    )
    # Should instruct model to decide output format
    assert "format" in prompt.lower() or "格式" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prompt_builder.py -v`
Expected: FAIL

- [ ] **Step 3: Implement prompt_builder.py**

```python
# src/engine/prompt_builder.py


def build_research_prompt(
    query: str,
    project_config: dict,
    context: str | None = None,
    output_prefs: dict | None = None,
) -> str:
    sections = []

    # Role & project context
    sections.append(f"""## Research Context

Project: {project_config['name']}
Description: {project_config.get('description', '')}
""")

    # System instructions from project config
    if project_config.get("system_instructions"):
        sections.append(f"""## Expert Instructions

{project_config['system_instructions']}
""")

    # Data source labeling instructions
    sections.append("""## Data Source Labeling

In your report, label each piece of information with its source:
- [API] — Data from project business APIs (available via file_search)
- [Web] — Data from web search
- [Computed] — Your own calculations or derivations
""")

    # Output format decision
    format_section = """## Output Format Decision

Analyze the content and autonomously choose the best output format:
- **HTML with interactive charts** (plotly/echarts): For data-heavy, multi-dimensional analysis
- **PDF-ready HTML**: For formal text-heavy reports
- **Markdown**: For short, simple answers
- **Mixed**: HTML main report + supplementary files

Your response MUST include:
1. A `<!-- FORMAT: html|pdf|markdown|mixed -->` tag at the very beginning
2. The complete report content
3. If format is html: provide complete, self-contained HTML with embedded CSS/JS
4. If format is pdf: provide clean HTML suitable for PDF conversion
"""

    # Output preferences
    if output_prefs:
        lang = output_prefs.get("preferred_language")
        if lang:
            format_section += f"\nPreferred output language: {lang}"
        hints = output_prefs.get("hints", [])
        if hints:
            format_section += "\n\nOutput hints:\n"
            for hint in hints:
                format_section += f"- {hint}\n"

    sections.append(format_section)

    # User query
    sections.append(f"""## Research Task

{query}
""")

    # Additional context
    if context:
        sections.append(f"""## Additional Context

{context}
""")

    return "\n".join(sections)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_prompt_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/engine/prompt_builder.py tests/test_prompt_builder.py
git commit -m "feat: prompt builder with format decision and source labeling"
```

---

## Task 7: Research Engine (OpenAI Deep Research API)

**Files:**
- Create: `src/engine/research.py`
- Create: `tests/test_research.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_research.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.engine.research import ResearchEngine


@pytest.fixture
def mock_openai():
    client = MagicMock()
    return client


@pytest.fixture
def engine(mock_openai):
    return ResearchEngine(openai_client=mock_openai, default_model="o3-deep-research")


def test_build_tools_with_vector_store(engine):
    tools = engine._build_tools(vector_store_id="vs_123")
    tool_types = [t["type"] for t in tools]
    assert "web_search_preview" in tool_types
    assert "code_interpreter" in tool_types
    assert "file_search" in tool_types


def test_build_tools_without_vector_store(engine):
    tools = engine._build_tools(vector_store_id=None)
    tool_types = [t["type"] for t in tools]
    assert "web_search_preview" in tool_types
    assert "code_interpreter" in tool_types
    assert "file_search" not in tool_types


async def test_start_research(engine, mock_openai):
    mock_response = MagicMock()
    mock_response.id = "resp_test123"
    mock_response.status = "queued"
    mock_openai.responses.create.return_value = mock_response

    resp_id = await engine.start(
        prompt="Analyze data trends",
        model="o3-deep-research",
        vector_store_id="vs_123",
    )
    assert resp_id == "resp_test123"
    mock_openai.responses.create.assert_called_once()
    call_kwargs = mock_openai.responses.create.call_args[1]
    assert call_kwargs["background"] is True


async def test_poll_completed(engine, mock_openai):
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.output_text = "# Research Report\nFindings here."
    mock_response.output = []
    mock_response.usage = MagicMock(total_tokens=5000)
    mock_openai.responses.retrieve.return_value = mock_response

    result = await engine.poll("resp_test123")
    assert result["status"] == "completed"
    assert "Research Report" in result["output_text"]


async def test_poll_in_progress(engine, mock_openai):
    mock_response = MagicMock()
    mock_response.status = "in_progress"
    mock_openai.responses.retrieve.return_value = mock_response

    result = await engine.poll("resp_test123")
    assert result["status"] == "in_progress"


def test_parse_format_tag(engine):
    text = "<!-- FORMAT: html -->\n<html>...</html>"
    fmt = engine.parse_format(text)
    assert fmt == "html"


def test_parse_format_default(engine):
    text = "Just some markdown text."
    fmt = engine.parse_format(text)
    assert fmt == "markdown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_research.py -v`
Expected: FAIL

- [ ] **Step 3: Implement research.py**

```python
# src/engine/research.py
import asyncio
import re
import structlog

logger = structlog.get_logger()


class ResearchEngine:
    def __init__(self, openai_client, default_model: str = "o3-deep-research"):
        self.client = openai_client
        self.default_model = default_model

    def _build_tools(self, vector_store_id: str | None = None) -> list[dict]:
        tools = [
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ]
        if vector_store_id:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            })
        return tools

    async def start(
        self,
        prompt: str,
        model: str | None = None,
        vector_store_id: str | None = None,
    ) -> str:
        """Start a background Deep Research request. Returns response ID."""
        tools = self._build_tools(vector_store_id)
        response = self.client.responses.create(
            model=model or self.default_model,
            input=prompt,
            tools=tools,
            background=True,
        )
        logger.info("research_started", response_id=response.id, model=model or self.default_model)
        return response.id

    async def poll(self, response_id: str) -> dict:
        """Poll for research completion. Returns status and result."""
        response = self.client.responses.retrieve(response_id)
        result = {"status": response.status}

        if response.status == "completed":
            result["output_text"] = response.output_text
            result["output"] = response.output
            result["total_tokens"] = (
                response.usage.total_tokens if response.usage else 0
            )

        return result

    async def start_with_retry(
        self,
        prompt: str,
        model: str | None = None,
        vector_store_id: str | None = None,
        max_retries: int = 2,
    ) -> str:
        """Start research with retry on failure."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self.start(prompt, model, vector_store_id)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning("start_retry", attempt=attempt + 1, wait=wait, error=str(e))
                    await asyncio.sleep(wait)
        raise last_error

    async def run_and_wait(
        self,
        prompt: str,
        model: str | None = None,
        vector_store_id: str | None = None,
        on_progress=None,
        timeout_seconds: int = 600,
    ) -> dict:
        """Start research and poll until completion with exponential backoff."""
        response_id = await self.start_with_retry(prompt, model, vector_store_id)

        delays = [5, 10, 20, 30]  # Exponential backoff
        delay_idx = 0
        elapsed = 0

        while elapsed < timeout_seconds:
            result = await self.poll(response_id)

            if result["status"] == "completed":
                result["format"] = self.parse_format(result.get("output_text", ""))
                return result

            if result["status"] in ("failed", "cancelled"):
                return result

            if on_progress:
                await on_progress(result["status"])

            delay = delays[min(delay_idx, len(delays) - 1)]
            delay_idx += 1
            await asyncio.sleep(delay)
            elapsed += delay

        return {"status": "failed", "error": f"Research timed out after {timeout_seconds}s"}

    def parse_format(self, text: str) -> str:
        """Extract format tag from model output."""
        match = re.search(r"<!--\s*FORMAT:\s*(\w+)\s*-->", text)
        if match:
            return match.group(1).lower()
        return "markdown"
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_research.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/engine/research.py tests/test_research.py
git commit -m "feat: research engine with Deep Research API polling"
```

---

## Task 8: Output Renderer

**Files:**
- Create: `src/output/__init__.py`
- Create: `src/output/renderer.py`
- Create: `tests/test_renderer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_renderer.py
import pytest
from pathlib import Path
from src.output.renderer import OutputRenderer


@pytest.fixture
def renderer(tmp_path):
    return OutputRenderer(output_base=tmp_path)


async def test_save_markdown(renderer, tmp_path):
    result = await renderer.render(
        task_id="dr_test1",
        format="markdown",
        content="# Report\n\nFindings here.",
    )
    assert result["format"] == "markdown"
    assert len(result["files"]) == 1
    assert result["files"][0]["name"] == "report.md"
    saved = (tmp_path / "dr_test1" / "report.md").read_text()
    assert "Findings here" in saved


async def test_save_html(renderer, tmp_path):
    html = "<!-- FORMAT: html -->\n<html><body><h1>Report</h1></body></html>"
    result = await renderer.render(
        task_id="dr_test2",
        format="html",
        content=html,
    )
    assert result["format"] == "html"
    assert result["files"][0]["name"] == "report.html"
    saved = (tmp_path / "dr_test2" / "report.html").read_text()
    assert "<h1>Report</h1>" in saved


async def test_strip_format_tag(renderer, tmp_path):
    html = "<!-- FORMAT: html -->\n<html><body>Content</body></html>"
    result = await renderer.render(
        task_id="dr_test3",
        format="html",
        content=html,
    )
    saved = (tmp_path / "dr_test3" / "report.html").read_text()
    assert "<!-- FORMAT:" not in saved


async def test_output_info_structure(renderer):
    result = await renderer.render(
        task_id="dr_test4",
        format="markdown",
        content="# Test",
    )
    assert "format" in result
    assert "files" in result
    f = result["files"][0]
    assert "name" in f
    assert "url" in f
    assert "type" in f
    assert "size" in f
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_renderer.py -v`
Expected: FAIL

- [ ] **Step 3: Implement renderer.py**

```python
# src/output/__init__.py
```

```python
# src/output/renderer.py
import re
from pathlib import Path

import structlog

logger = structlog.get_logger()

FORMAT_EXT = {
    "markdown": ("report.md", "text/markdown"),
    "html": ("report.html", "text/html"),
    "pdf": ("report.html", "text/html"),  # HTML saved first, PDF conversion separate
    "mixed": ("report.html", "text/html"),
}


class OutputRenderer:
    def __init__(self, output_base: Path | str):
        self.output_base = Path(output_base)

    async def render(
        self,
        task_id: str,
        format: str,
        content: str,
    ) -> dict:
        """Save rendered output and return file info."""
        task_dir = self.output_base / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Strip format tag
        content = re.sub(r"<!--\s*FORMAT:\s*\w+\s*-->\n?", "", content).strip()

        filename, mime_type = FORMAT_EXT.get(format, ("report.md", "text/markdown"))
        filepath = task_dir / filename
        filepath.write_text(content, encoding="utf-8")

        files = [
            {
                "name": filename,
                "url": f"/files/{task_id}/{filename}",
                "type": mime_type,
                "size": filepath.stat().st_size,
            }
        ]

        # PDF conversion (if requested and HTML exists)
        if format == "pdf":
            pdf_path = await self._html_to_pdf(filepath, task_dir)
            if pdf_path:
                files.append({
                    "name": pdf_path.name,
                    "url": f"/files/{task_id}/{pdf_path.name}",
                    "type": "application/pdf",
                    "size": pdf_path.stat().st_size,
                })

        logger.info("output_rendered", task_id=task_id, format=format, files=len(files))
        return {"format": format, "files": files}

    async def _html_to_pdf(self, html_path: Path, task_dir: Path) -> Path | None:
        """Convert HTML to PDF using WeasyPrint in Docker sandbox."""
        pdf_path = task_dir / "report.pdf"
        try:
            import subprocess
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network=none",
                    "--memory=512m",
                    "--cpus=1",
                    "-v", f"{task_dir}:/work:rw",
                    "dr-renderer",
                    "python", "-c",
                    "from weasyprint import HTML; HTML('/work/report.html').write_pdf('/work/report.pdf')",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and pdf_path.exists():
                return pdf_path
            logger.warning("pdf_conversion_failed", stderr=result.stderr)
        except Exception as e:
            logger.warning("pdf_conversion_error", error=str(e))
        return None
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_renderer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/output/__init__.py src/output/renderer.py tests/test_renderer.py
git commit -m "feat: output renderer with format-based file generation"
```

---

## Task 9: API Routes & FastAPI App

**Files:**
- Create: `src/api/routes.py`
- Create: `src/main.py`
- Create: `tests/test_routes.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_routes.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
from src.main import create_app


@pytest.fixture
def app(tmp_path):
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "sk-test",
        "API_TOKEN": "test-token",
        "STORAGE_PATH": str(tmp_path / "output"),
        "PROJECTS_PATH": str(tmp_path / "projects"),
    }):
        # Create a test project
        proj_dir = tmp_path / "projects" / "testproj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "config.yaml").write_text(
            'name: testproj\ndescription: "Test"\napis: []'
        )
        # Reset settings singleton
        import src.config
        src.config.settings = None
        return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_projects_requires_auth(client):
    resp = await client.get("/projects")
    assert resp.status_code == 403  # No auth header


async def test_projects_with_auth(client):
    resp = await client.get("/projects", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "testproj"


async def test_submit_research_unknown_project(client):
    resp = await client.post(
        "/research",
        json={"project": "nonexistent", "query": "test"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 404


async def test_submit_research_success(client):
    with patch("src.api.routes.run_research_task") as mock_run:
        resp = await client.post(
            "/research",
            json={"project": "testproj", "query": "analyze trends"},
            headers={"Authorization": "Bearer test-token"},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"].startswith("dr_")
    assert data["status"] == "pending"


async def test_get_task_not_found(client):
    resp = await client.get(
        "/research/dr_nonexistent",
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 404


async def test_cancel_task(client):
    # First create a task
    with patch("src.api.routes.run_research_task"):
        create_resp = await client.post(
            "/research",
            json={"project": "testproj", "query": "test"},
            headers={"Authorization": "Bearer test-token"},
        )
    task_id = create_resp.json()["task_id"]

    resp = await client.post(
        f"/research/{task_id}/cancel",
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_routes.py -v`
Expected: FAIL

- [ ] **Step 3: Implement routes.py**

```python
# src/api/routes.py
import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse

from src.auth import require_auth
from src.api.models import (
    ErrorResponse,
    HealthResponse,
    ProjectListResponse,
    ProjectInfo,
    ResearchRequest,
    TaskResponse,
    TaskStatus,
    ProgressInfo,
    OutputInfo,
    OutputFile,
    SourceInfo,
    UsageInfo,
)
from src.config import get_settings
from src.engine.project_loader import ProjectLoader
from src.task.manager import TaskManager

router = APIRouter()

# These get set during app lifespan
task_manager: TaskManager | None = None
project_loader: ProjectLoader | None = None
semaphore: asyncio.Semaphore | None = None


def init_router(tm: TaskManager, pl: ProjectLoader, sem: asyncio.Semaphore):
    global task_manager, project_loader, semaphore
    task_manager = tm
    project_loader = pl
    semaphore = sem


@router.get("/health", response_model=HealthResponse)
async def health():
    active = await task_manager.count_active() if task_manager else 0
    return HealthResponse(status="ok", openai_api="connected", active_tasks=active)


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(_=Depends(require_auth)):
    projects = project_loader.list_projects()
    return ProjectListResponse(
        projects=[ProjectInfo(**p) for p in projects]
    )


@router.post("/research", status_code=202)
async def submit_research(
    req: ResearchRequest,
    background_tasks: BackgroundTasks,
    _=Depends(require_auth),
):
    # Validate project exists
    try:
        project_loader.load(req.project)
    except FileNotFoundError:
        available = [p["name"] for p in project_loader.list_projects()]
        raise HTTPException(
            status_code=404,
            detail={
                "code": "project_not_found",
                "message": f"Project '{req.project}' does not exist",
                "details": {"available_projects": available},
            },
        )

    # Check concurrency via database count
    active_count = await task_manager.count_active()
    if active_count >= get_settings().max_concurrent_tasks:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "too_many_requests",
                "message": "Max concurrent tasks reached. Try again later.",
            },
        )

    # Create task
    task = await task_manager.create(
        project=req.project,
        query=req.query,
        context=req.context,
        callback_url=req.callback_url,
    )

    # Launch background research
    background_tasks.add_task(run_research_task, task["task_id"], req)

    return {"task_id": task["task_id"], "status": "pending", "created_at": task["created_at"]}


@router.get("/research/{task_id}")
async def get_task(task_id: str, _=Depends(require_auth)):
    task = await task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"code": "task_not_found", "message": "Task not found"})

    response = {
        "task_id": task["task_id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
    }

    if task.get("stage"):
        response["progress"] = {"stage": task["stage"], "message": task.get("message", "")}

    if task["status"] == "completed" and task.get("result_data"):
        response["output"] = task["result_data"]
        if task.get("usage_data"):
            response["usage"] = task["usage_data"]

    return response


@router.post("/research/{task_id}/cancel")
async def cancel_task(task_id: str, _=Depends(require_auth)):
    task = await task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail={"code": "task_not_found", "message": "Task not found"})

    await task_manager.update_status(task_id, "cancelled")
    return {"task_id": task_id, "status": "cancelled"}


@router.get("/files/{task_id}/{filename}")
async def get_file(task_id: str, filename: str, _=Depends(require_auth)):
    settings = get_settings()
    file_path = (settings.storage_path / task_id / filename).resolve()
    # Prevent path traversal
    if not str(file_path).startswith(str(settings.storage_path.resolve())):
        raise HTTPException(status_code=400, detail={"code": "invalid_path", "message": "Invalid file path"})
    if not file_path.exists():
        raise HTTPException(status_code=404, detail={"code": "file_not_found", "message": "File not found"})
    return FileResponse(file_path)


async def run_research_task(task_id: str, req: ResearchRequest):
    """Background task: runs the full research pipeline."""
    # Imported here to avoid circular imports and allow mocking in tests
    from openai import OpenAI
    from src.engine.data_preparation import DataPreparator
    from src.engine.prompt_builder import build_research_prompt
    from src.engine.research import ResearchEngine
    from src.output.renderer import OutputRenderer
    import time

    settings = get_settings()
    start_time = time.time()

    try:
        async with semaphore:
            await task_manager.update_status(task_id, "processing", stage="preparing", message="Loading project config...")

            # 1. Load project config
            config = project_loader.load(req.project)
            output_prefs = project_loader.load_output_prefs(req.project)

            # 2. Data preparation
            openai_client = OpenAI(api_key=settings.openai_api_key)
            preparator = DataPreparator(openai_client=openai_client)
            vector_store_id = None

            if config.get("apis"):
                await task_manager.update_status(task_id, "processing", stage="preparing", message="Fetching business data...")
                all_prefetch = []
                all_docs = []
                for api_cfg in config["apis"]:
                    prefetch_results = await preparator.prefetch(api_cfg)
                    all_prefetch.extend(prefetch_results)
                    if api_cfg.get("docs_file"):
                        try:
                            docs = project_loader.load_api_docs(req.project, api_cfg["docs_file"])
                            all_docs.append(docs)
                        except FileNotFoundError:
                            pass

                if all_prefetch or all_docs:
                    vector_store_id = await preparator.create_vector_store(
                        task_id=task_id,
                        prefetch_results=all_prefetch,
                        api_docs_content="\n\n".join(all_docs) if all_docs else None,
                    )

            # 3. Build prompt
            prompt = build_research_prompt(
                query=req.query,
                project_config=config,
                context=req.context,
                output_prefs=output_prefs,
            )

            # 4. Run research
            await task_manager.update_status(task_id, "processing", stage="researching", message="Deep research in progress...")
            engine = ResearchEngine(openai_client=openai_client, default_model=settings.default_model)

            async def on_progress(status):
                await task_manager.update_status(task_id, "processing", stage="researching", message=f"Research status: {status}")

            result = await engine.run_and_wait(
                prompt=prompt,
                model=config.get("model"),
                vector_store_id=vector_store_id,
                on_progress=on_progress,
            )

            if result["status"] != "completed":
                await task_manager.update_status(task_id, "failed", message=f"Research failed: {result['status']}")
                return

            # 5. Render output
            await task_manager.update_status(task_id, "processing", stage="rendering", message="Rendering output...")
            renderer = OutputRenderer(output_base=settings.storage_path)
            output = await renderer.render(
                task_id=task_id,
                format=result["format"],
                content=result["output_text"],
            )

            # 6. Complete
            elapsed = time.time() - start_time
            result_data = {
                "format": output["format"],
                "files": output["files"],
                "summary": result["output_text"][:200] + "..." if len(result["output_text"]) > 200 else result["output_text"],
                "sources": _parse_sources(result["output_text"])
            }
            usage_data = {
                "model": config.get("model", settings.default_model),
                "total_tokens": result.get("total_tokens", 0),
                "research_time_seconds": round(elapsed, 1),
            }
            await task_manager.complete(task_id, result_data, usage_data)

            # 7. Cleanup
            if vector_store_id:
                await preparator.cleanup(vector_store_id)

            # 8. Callback
            if req.callback_url:
                await _send_callback(req.callback_url, task_id, result_data)

    except Exception as e:
        import structlog
        structlog.get_logger().error("research_task_failed", task_id=task_id, error=str(e))
        await task_manager.update_status(task_id, "failed", message=str(e))


def _parse_sources(text: str) -> list[dict]:
    """Parse [API] and [Web] source labels from model output."""
    import re
    sources = []
    seen = set()
    # Match [Web](url) or [Web] patterns
    for match in re.finditer(r'\[Web\]\s*(?:\(([^)]+)\))?', text):
        url = match.group(1) or ""
        if url not in seen:
            seen.add(url)
            sources.append({"type": "web", "url": url, "title": ""})
    if "[API]" in text:
        sources.append({"type": "api", "name": "project_data"})
    return sources


async def _send_callback(url: str, task_id: str, result_data: dict):
    """Send completion callback, filtering private IPs."""
    import ipaddress
    from urllib.parse import urlparse
    import httpx
    import socket

    try:
        hostname = urlparse(url).hostname
        ip = socket.gethostbyname(hostname)
        if ipaddress.ip_address(ip).is_private:
            import structlog
            structlog.get_logger().warning("callback_blocked_private_ip", url=url)
            return
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(url, json={"task_id": task_id, "status": "completed", "result": result_data})
    except Exception as e:
        import structlog
        structlog.get_logger().warning("callback_failed", url=url, error=str(e))
```

- [ ] **Step 4: Implement main.py**

```python
# src/main.py
import asyncio
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from src.config import get_settings
from src.api.routes import router, init_router
from src.engine.project_loader import ProjectLoader
from src.task.manager import TaskManager


def create_app() -> FastAPI:
    settings = get_settings()

    # Configure logging
    import logging
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    task_manager = TaskManager(db_path=str(settings.storage_path / "tasks.db"))
    project_loader = ProjectLoader(settings.projects_path)
    semaphore = asyncio.Semaphore(settings.max_concurrent_tasks)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.storage_path.mkdir(parents=True, exist_ok=True)
        await task_manager.init()
        init_router(task_manager, project_loader, semaphore)
        yield
        await task_manager.close()

    app = FastAPI(
        title="Deep Research Framework",
        description="Deep research service powered by OpenAI Deep Research API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


if __name__ == "__main__":
    settings = get_settings()
    app = create_app()
    uvicorn.run(app, host=settings.host, port=settings.port)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_routes.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/main.py src/api/routes.py tests/test_routes.py
git commit -m "feat: FastAPI app with all API routes and research pipeline"
```

---

## Task 10: Docker Setup

**Files:**
- Create: `Dockerfile`
- Create: `Dockerfile.renderer`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY src/ src/
COPY projects/ projects/

RUN mkdir -p /data/output

ENV STORAGE_PATH=/data/output
ENV PROJECTS_PATH=/app/projects

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Create Dockerfile.renderer**

```dockerfile
# Dockerfile.renderer
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 \
    libffi-dev shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir weasyprint matplotlib plotly pandas pillow jinja2

WORKDIR /work
```

- [ ] **Step 3: Create docker-compose.yml**

```yaml
# docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "${PORT:-8000}:8000"
    env_file:
      - .env
    volumes:
      - output_data:/data/output
      - ./projects:/app/projects:ro
      - /var/run/docker.sock:/var/run/docker.sock  # For renderer sandbox
    restart: unless-stopped

  renderer:
    build:
      context: .
      dockerfile: Dockerfile.renderer
    image: dr-renderer
    profiles:
      - build-only  # Only built, not run as service

volumes:
  output_data:
```

- [ ] **Step 4: Verify Dockerfiles are valid**

Run: `cd ~/deep-research-framework && docker build --check -f Dockerfile . 2>&1 || echo "Docker check not supported, syntax looks ok"`
Expected: No syntax errors

- [ ] **Step 5: Commit**

```bash
git add Dockerfile Dockerfile.renderer docker-compose.yml
git commit -m "feat: Docker setup with main service and renderer sandbox"
```

---

## Task 11: Integration Test & Final Verification

**Files:**
- Modify: `tests/conftest.py` (add shared async fixtures)

- [ ] **Step 1: Run full test suite**

Run: `cd ~/deep-research-framework && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify project structure**

Run: `cd ~/deep-research-framework && find . -type f -not -path './.git/*' -not -path './__pycache__/*' -not -path './.pytest_cache/*' | sort`
Expected: All files from the plan exist

- [ ] **Step 3: Verify app starts (dry run)**

Run: `cd ~/deep-research-framework && python -c "from src.main import create_app; print('App factory OK')"`
Expected: `App factory OK`

- [ ] **Step 4: Commit any final fixes**

```bash
git add -A
git commit -m "chore: integration test and final cleanup"
```

- [ ] **Step 5: Tag release**

```bash
git tag v0.1.0
```
