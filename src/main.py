from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.config import get_settings
from src.api.routes import router, init_router
from src.engine.project_loader import ProjectLoader
from src.task.manager import TaskManager

try:
    import structlog

    log_lib = "structlog"
except ImportError:
    log_lib = "logging"


def create_app() -> FastAPI:
    settings = get_settings()

    if log_lib == "structlog":
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
