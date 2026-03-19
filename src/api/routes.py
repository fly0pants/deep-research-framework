from __future__ import annotations
import asyncio
import re
import ipaddress
import socket
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
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
from src.memory.store import UserMemoryStore
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
memory_store: UserMemoryStore | None = None


def init_router(tm: TaskManager, pl: ProjectLoader, sem: asyncio.Semaphore, ms: UserMemoryStore):
    global task_manager, project_loader, semaphore, memory_store
    task_manager = tm
    project_loader = pl
    semaphore = sem
    memory_store = ms


def _hash_api_key(api_key: str | None) -> str | None:
    """Derive a stable, non-reversible user_id from API key."""
    if not api_key:
        return None
    import hashlib
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


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
    if not req.api_key:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "api_key_required",
                "message": "api_key is required. The caller must provide the user's API key.",
            },
        )

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
async def get_task(task_id: str, request: Request, _=Depends(require_auth)):
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

    if task["status"] == "failed" and task.get("message"):
        response["error"] = {"message": task["message"]}

    if task["status"] == "completed" and task.get("result_data"):
        result_data = task["result_data"]
        # Build full URLs for files
        base_url = str(request.base_url).rstrip("/")
        if result_data.get("files"):
            for f in result_data["files"]:
                if f.get("url", "").startswith("/"):
                    f["url"] = f"{base_url}{f['url']}"
        response["output"] = result_data
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
async def get_file(task_id: str, filename: str):
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


@router.get("/incidents")
async def list_incidents(_=Depends(require_auth)):
    """List recent API incidents for monitoring."""
    import json
    from pathlib import Path

    settings = get_settings()
    log_file = settings.storage_path / "_incidents" / "api_incidents.jsonl"
    if not log_file.exists():
        return {"incidents": [], "total": 0}

    incidents = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    incidents.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # Return most recent 50
    incidents = incidents[-50:]
    incidents.reverse()
    return {"incidents": incidents, "total": len(incidents)}


def _extract_summary(text: str) -> str:
    """Extract the SUMMARY comment tag from model output."""
    match = re.search(r"<!--\s*SUMMARY:\s*\n?(.*?)-->", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: no summary tag found
    return ""


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

        # Lookup user memory
        user_memory = None
        user_id = _hash_api_key(req.api_key)
        if user_id:
            mem_record = await memory_store.get(user_id)
            if mem_record:
                user_memory = mem_record["memory"]

        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        openai_client = OpenAI(**client_kwargs)
        preparator = DataPreparator(openai_client=openai_client, runtime_api_key=req.api_key)

        # Collect business data and API docs
        business_data_parts = []
        if config.get("apis"):
            await task_manager.update_status(
                task_id, "processing", stage="preparing", message="Fetching business data...",
            )
            for api_cfg in config["apis"]:
                prefetch_results = await preparator.prefetch(api_cfg)
                for pr in prefetch_results:
                    import json as _json
                    business_data_parts.append(
                        f"### Endpoint: {pr['endpoint']}\n```json\n{_json.dumps(pr['data'], ensure_ascii=False, indent=2)}\n```"
                    )
                # Load API docs
                if api_cfg.get("docs_files"):
                    docs = project_loader.load_all_api_docs(req.project, api_cfg["docs_files"])
                    if docs:
                        business_data_parts.append(f"### API Documentation\n{docs}")
                elif api_cfg.get("docs_file"):
                    try:
                        docs = project_loader.load_api_docs(req.project, api_cfg["docs_file"])
                        business_data_parts.append(f"### API Documentation\n{docs}")
                    except FileNotFoundError:
                        pass

        business_data = "\n\n".join(business_data_parts) if business_data_parts else None

        prompt = build_research_prompt(
            query=req.query,
            project_config=config,
            context=req.context,
            output_prefs=output_prefs,
            user_memory=user_memory,
        )

        await task_manager.update_status(
            task_id, "processing", stage="researching", message="Research in progress...",
        )
        engine = ResearchEngine(
            openai_client=openai_client, default_model=settings.default_model
        )

        async def on_progress(status):
            await task_manager.update_status(
                task_id, "processing", stage="researching", message=f"Research status: {status}",
            )

        result = await engine.run_and_wait(
            prompt=prompt,
            model=config.get("model"),
            on_progress=on_progress,
            business_data=business_data,
            api_configs=config.get("apis") or None,
            preparator=preparator,
        )

        if result["status"] == "data_unavailable":
            error_msg = result.get("error", "数据服务不可用")
            await task_manager.update_status(
                task_id, "failed",
                message=f"数据不足，无法生成报告：{error_msg}",
            )
            # Log API failure for technical monitoring
            await _log_api_incident(
                task_id=task_id,
                project=req.project,
                query=req.query,
                error=error_msg,
                api_stats=result.get("api_call_stats"),
            )
            return

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
        summary = _extract_summary(result["output_text"])
        result_data = {
            "format": output["format"],
            "files": output["files"],
            "summary": summary,
            "sources": _parse_sources(result["output_text"]),
        }
        usage_data = {
            "model": config.get("model", settings.default_model),
            "total_tokens": result.get("total_tokens", 0),
            "research_time_seconds": round(elapsed, 1),
        }
        await task_manager.complete(task_id, result_data, usage_data)

        if req.callback_url:
            await _send_callback(req.callback_url, task_id, result_data)

        # Memory update after callback — errors won't affect task
        if user_id:
            try:
                from src.memory.updater import MemoryUpdater
                memory_summary = summary or result["output_text"][:500]

                # Store this interaction for future reference
                await memory_store.add_interaction(user_id, req.query, memory_summary)

                # Get recent history for richer profile generation
                recent = await memory_store.get_recent_interactions(user_id)

                updater = MemoryUpdater(openai_client=openai_client)
                updated_memory = await updater.generate_updated_memory(
                    query=req.query,
                    summary=memory_summary,
                    existing_memory=user_memory,
                    recent_interactions=recent,
                )
                if updated_memory:
                    await memory_store.upsert(user_id, updated_memory)
                    logger.info("user_memory_updated", user_id=user_id)
            except Exception as e:
                logger.warning("user_memory_update_failed", user_id=user_id, error=str(e))

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


async def _log_api_incident(
    task_id: str,
    project: str,
    query: str,
    error: str,
    api_stats: dict | None = None,
):
    """Log API failures to a JSON Lines file for monitoring."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    settings = get_settings()
    log_dir = settings.storage_path / "_incidents"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "api_incidents.jsonl"

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "project": project,
        "query": query[:200],
        "error": error,
        "api_stats": api_stats,
    }
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.error(
            "api_incident_logged",
            task_id=task_id,
            project=project,
            error=error,
        )
    except Exception as e:
        logger.error("incident_log_write_failed", error=str(e))


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
