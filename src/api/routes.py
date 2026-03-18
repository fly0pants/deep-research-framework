from __future__ import annotations
import asyncio
import re
import ipaddress
import socket
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse

from src.auth import require_auth
from src.api.models import (
    HealthResponse,
    ProjectListResponse,
    ProjectInfo,
    ResearchRequest,
)
from src.config import get_settings
from src.engine.project_loader import ProjectLoader
from src.task.manager import TaskManager

try:
    import structlog

    logger = structlog.get_logger()
except ImportError:
    import logging

    logger = logging.getLogger(__name__)

router = APIRouter()

# Set during app lifespan
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
    return ProjectListResponse(projects=[ProjectInfo(**p) for p in projects])


@router.post("/research", status_code=202)
async def submit_research(
    req: ResearchRequest,
    background_tasks: BackgroundTasks,
    _=Depends(require_auth),
):
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

    active_count = await task_manager.count_active()
    if active_count >= get_settings().max_concurrent_tasks:
        raise HTTPException(
            status_code=429,
            detail={
                "code": "too_many_requests",
                "message": "Max concurrent tasks reached. Try again later.",
            },
        )

    task = await task_manager.create(
        project=req.project,
        query=req.query,
        context=req.context,
        callback_url=req.callback_url,
    )

    background_tasks.add_task(run_research_task, task["task_id"], req)
    return {"task_id": task["task_id"], "status": "pending", "created_at": task["created_at"]}


@router.get("/research/{task_id}")
async def get_task(task_id: str, _=Depends(require_auth)):
    task = await task_manager.get(task_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "task_not_found", "message": "Task not found"},
        )

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
        raise HTTPException(
            status_code=404,
            detail={"code": "task_not_found", "message": "Task not found"},
        )
    await task_manager.update_status(task_id, "cancelled")
    return {"task_id": task_id, "status": "cancelled"}


@router.get("/files/{task_id}/{filename}")
async def get_file(task_id: str, filename: str, _=Depends(require_auth)):
    settings = get_settings()
    file_path = (settings.storage_path / task_id / filename).resolve()
    if not str(file_path).startswith(str(settings.storage_path.resolve())):
        raise HTTPException(
            status_code=400,
            detail={"code": "invalid_path", "message": "Invalid file path"},
        )
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"code": "file_not_found", "message": "File not found"},
        )
    return FileResponse(file_path)


def _parse_sources(text: str) -> list[dict]:
    sources = []
    seen = set()
    for match in re.finditer(r"\[Web\]\s*(?:\(([^)]+)\))?", text):
        url = match.group(1) or ""
        if url not in seen:
            seen.add(url)
            sources.append({"type": "web", "url": url, "title": ""})
    if "[API]" in text:
        sources.append({"type": "api", "name": "project_data"})
    return sources


async def run_research_task(task_id: str, req: ResearchRequest):
    """Background task: runs the full research pipeline."""
    from openai import OpenAI

    from src.engine.data_preparation import DataPreparator
    from src.engine.prompt_builder import build_research_prompt
    from src.engine.research import ResearchEngine
    from src.output.renderer import OutputRenderer
    import time

    settings = get_settings()
    start_time = time.time()

    try:
        await task_manager.update_status(
            task_id, "processing", stage="preparing", message="Loading project config..."
        )

        config = project_loader.load(req.project)
        output_prefs = project_loader.load_output_prefs(req.project)

        openai_client = OpenAI(api_key=settings.openai_api_key)
        preparator = DataPreparator(openai_client=openai_client)
        vector_store_id = None

        if config.get("apis"):
            await task_manager.update_status(
                task_id,
                "processing",
                stage="preparing",
                message="Fetching business data...",
            )
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

        prompt = build_research_prompt(
            query=req.query,
            project_config=config,
            context=req.context,
            output_prefs=output_prefs,
        )

        await task_manager.update_status(
            task_id,
            "processing",
            stage="researching",
            message="Deep research in progress...",
        )
        engine = ResearchEngine(
            openai_client=openai_client, default_model=settings.default_model
        )

        async def on_progress(status):
            await task_manager.update_status(
                task_id,
                "processing",
                stage="researching",
                message=f"Research status: {status}",
            )

        result = await engine.run_and_wait(
            prompt=prompt,
            model=config.get("model"),
            vector_store_id=vector_store_id,
            on_progress=on_progress,
        )

        if result["status"] != "completed":
            await task_manager.update_status(
                task_id, "failed", message=f"Research failed: {result['status']}"
            )
            return

        await task_manager.update_status(
            task_id, "processing", stage="rendering", message="Rendering output..."
        )
        renderer = OutputRenderer(output_base=settings.storage_path)
        output = await renderer.render(
            task_id=task_id,
            format=result["format"],
            content=result["output_text"],
        )

        elapsed = time.time() - start_time
        result_data = {
            "format": output["format"],
            "files": output["files"],
            "summary": result["output_text"][:200] + "..."
            if len(result["output_text"]) > 200
            else result["output_text"],
            "sources": _parse_sources(result["output_text"]),
        }
        usage_data = {
            "model": config.get("model", settings.default_model),
            "total_tokens": result.get("total_tokens", 0),
            "research_time_seconds": round(elapsed, 1),
        }
        await task_manager.complete(task_id, result_data, usage_data)

        if vector_store_id:
            await preparator.cleanup(vector_store_id)

        if req.callback_url:
            await _send_callback(req.callback_url, task_id, result_data)

    except Exception as e:
        try:
            import structlog

            structlog.get_logger().error(
                "research_task_failed", task_id=task_id, error=str(e)
            )
        except ImportError:
            import logging

            logging.getLogger(__name__).error(f"research_task_failed: {task_id} - {e}")
        await task_manager.update_status(task_id, "failed", message=str(e))


async def _send_callback(url: str, task_id: str, result_data: dict):
    import httpx

    try:
        hostname = urlparse(url).hostname
        ip = socket.gethostbyname(hostname)
        if ipaddress.ip_address(ip).is_private:
            return
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(
                url,
                json={"task_id": task_id, "status": "completed", "result": result_data},
            )
    except Exception:
        pass
