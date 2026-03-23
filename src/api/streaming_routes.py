from __future__ import annotations

import asyncio
import json as _json
import re
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import StreamingResponse

from src.api.models import InternalResearchRequest
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

streaming_router = APIRouter()

# Set during app lifespan
task_manager: TaskManager | None = None
project_loader: ProjectLoader | None = None
memory_store: UserMemoryStore | None = None

# ---------------------------------------------------------------------------
# In-memory event queues for SSE streaming (task_id → Queue)
# ---------------------------------------------------------------------------
_event_queues: dict[str, asyncio.Queue] = {}


def get_event_queue(task_id: str) -> asyncio.Queue | None:
    return _event_queues.get(task_id)


def create_event_queue(task_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _event_queues[task_id] = q
    return q


def remove_event_queue(task_id: str):
    _event_queues.pop(task_id, None)


def init_streaming_router(tm: TaskManager, pl: ProjectLoader, ms: UserMemoryStore):
    global task_manager, project_loader, memory_store
    task_manager = tm
    project_loader = pl
    memory_store = ms


def _verify_internal_key(request: Request):
    """Verify X-Internal-Key header for service-to-service calls."""
    settings = get_settings()
    if not settings.internal_api_key:
        raise HTTPException(status_code=500, detail="Internal API key not configured")
    key = request.headers.get("X-Internal-Key")
    if key != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid internal API key")


def _extract_summary(text: str) -> str:
    """Extract the SUMMARY comment tag from model output."""
    match = re.search(r"<!--\s*SUMMARY:\s*\n?(.*?)-->", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _parse_sources(text: str) -> list[dict]:
    """Parse source tags from report text."""
    sources = []
    seen = set()
    for match in re.finditer(r"\[Web\]\s*(?:\(([^)]+)\))?", text):
        url = match.group(1) or ""
        if url not in seen:
            seen.add(url)
            sources.append({"type": "web", "url": url, "title": ""})
    if "[AdMapix]" in text or "[API]" in text:
        sources.append({"type": "api", "name": "project_data"})
    return sources


@streaming_router.get("/research/{task_id}/stream")
async def stream_research_progress(task_id: str):
    """SSE endpoint for real-time streaming. Frontend connects directly."""
    queue = get_event_queue(task_id)

    # If no queue, task might be already done or not started
    if queue is None:
        # Check if task exists and is done
        task = await task_manager.get(task_id) if task_manager else None
        if task and task["status"] in ("completed", "failed"):
            async def done_gen():
                yield f"data: {_json.dumps({'type': 'complete', 'status': task['status']})}\n\n"
            return StreamingResponse(done_gen(), media_type="text/event-stream", headers={
                "Cache-Control": "no-cache", "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            })
        # Task not found or not started yet — wait a bit for queue to appear
        for _ in range(30):  # wait up to 30 seconds
            await asyncio.sleep(1)
            queue = get_event_queue(task_id)
            if queue:
                break
        if queue is None:
            return StreamingResponse(
                iter([f"data: {_json.dumps({'type': 'error', 'message': 'Task not found'})}\n\n"]),
                media_type="text/event-stream",
                headers={"Access-Control-Allow-Origin": "*"},
            )

    async def event_generator():
        try:
            while True:
                event = await asyncio.wait_for(queue.get(), timeout=600)  # 10 min max
                yield f"data: {_json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("type") in ("complete", "error", "failed"):
                    break
        except asyncio.TimeoutError:
            yield f"data: {_json.dumps({'type': 'error', 'message': 'Stream timeout'})}\n\n"
        finally:
            remove_event_queue(task_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",
        },
    )


@streaming_router.options("/research/{task_id}/stream")
async def stream_cors_preflight(task_id: str):
    return StreamingResponse(
        iter([""]),
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )


@streaming_router.post("/internal/research/streaming", status_code=202)
async def submit_internal_streaming_research(
    req: InternalResearchRequest,
    request: Request,
    background_tasks: BackgroundTasks,
):
    """Internal endpoint for streaming research pipeline. Requires X-Internal-Key."""
    _verify_internal_key(request)

    if not req.user_id:
        raise HTTPException(status_code=422, detail={
            "code": "user_id_required",
            "message": "user_id is required for internal research requests.",
        })

    # Reject garbled queries
    q = req.query.strip() if req.query else ""
    if not q or len(q) < 5:
        raise HTTPException(status_code=422, detail={
            "code": "query_too_short",
            "message": "Query is too short. Please provide a meaningful research question.",
        })
    q_chars = [c for c in q if not c.isspace()]
    if q_chars and sum(1 for c in q_chars if c == '?') / len(q_chars) > 0.3:
        raise HTTPException(status_code=422, detail={
            "code": "query_encoding_error",
            "message": "Query appears corrupted (too many '?' characters). Please check your terminal encoding is set to UTF-8.",
        })

    try:
        project_loader.load(req.project)
    except FileNotFoundError:
        available = [p["name"] for p in project_loader.list_projects()]
        raise HTTPException(status_code=404, detail={
            "code": "project_not_found",
            "message": f"Project '{req.project}' does not exist",
            "details": {"available_projects": available},
        })

    active_count = await task_manager.count_active()
    if active_count >= get_settings().max_concurrent_tasks:
        raise HTTPException(status_code=429, detail={
            "code": "too_many_requests",
            "message": "Max concurrent tasks reached. Try again later.",
        })

    task = await task_manager.create(
        project=req.project,
        query=req.query,
        context=req.context,
        source=req.source or "admapix-website",
    )

    # Create event queue for SSE streaming
    event_queue = create_event_queue(task["task_id"])

    background_tasks.add_task(
        run_streaming_research_task, task["task_id"], req, event_queue
    )
    return {"task_id": task["task_id"], "status": "pending", "created_at": task["created_at"]}


async def run_streaming_research_task(
    task_id: str, req: InternalResearchRequest, event_queue: asyncio.Queue
):
    """Background task: runs the streaming two-agent research pipeline."""
    import json as _json_mod

    from openai import OpenAI

    from src.engine.data_preparation import DataPreparator
    from src.engine.streaming_pipeline import StreamingResearchPipeline
    from src.output.renderer import OutputRenderer

    settings = get_settings()
    start_time = time.time()

    try:
        await task_manager.update_status(
            task_id, "processing", stage="preparing", message="Loading project config..."
        )

        config = project_loader.load(req.project)
        output_prefs = project_loader.load_output_prefs(req.project)

        client_kwargs = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        openai_client = OpenAI(**client_kwargs)
        preparator = DataPreparator(openai_client=openai_client, runtime_api_key=None)

        # Collect business data and API docs
        business_data_parts = []
        api_config = None
        if config.get("apis"):
            await task_manager.update_status(
                task_id, "processing", stage="preparing", message="Fetching business data...",
            )
            for api_cfg in config["apis"]:
                api_config = api_cfg  # keep last api_config for pipeline
                prefetch_results = await preparator.prefetch(api_cfg)
                for pr in prefetch_results:
                    business_data_parts.append(
                        f"### Endpoint: {pr['endpoint']}\n```json\n{_json_mod.dumps(pr['data'], ensure_ascii=False, indent=2)}\n```"
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

        # Define progress callback — pushes to SSE queue + Supabase
        async def on_progress(
            phase=None, stage=None, message=None, progress_pct=None,
            partial_content=None, chunk=None, **_
        ):
            if chunk is not None:
                # Streaming chunk — push to SSE only, don't touch Supabase
                event_queue.put_nowait({
                    "type": "chunk",
                    "content": chunk,
                    "progress_pct": progress_pct,
                })
                return  # Don't update Supabase for every chunk

            # Regular progress update — push to both SSE and Supabase
            event = {
                "type": "progress",
                "phase": phase,
                "stage": stage,
                "message": message,
                "progress_pct": progress_pct,
            }
            event_queue.put_nowait(event)

            await task_manager.update_status(
                task_id, "processing",
                phase=phase, stage=stage, message=message,
                progress_pct=progress_pct,
            )

        await task_manager.update_status(
            task_id, "processing", stage="researching", message="Streaming research in progress...",
        )

        pipeline = StreamingResearchPipeline(
            openai_client=openai_client, default_model=settings.default_model
        )

        result = await pipeline.run(
            query=req.query,
            project_config=config,
            preparator=preparator,
            api_config=api_config or {},
            on_progress=on_progress,
            context=req.context,
            output_prefs=output_prefs,
            business_data=business_data,
            model=config.get("model"),
        )

        if result["status"] == "data_unavailable":
            error_msg = result.get("error", result.get("output_text", "数据服务不可用"))
            await task_manager.update_status(
                task_id, "failed",
                message=f"数据不足，无法生成报告：{error_msg}",
            )
            event_queue.put_nowait({
                "type": "failed",
                "status": "failed",
                "message": f"数据不足，无法生成报告：{error_msg}",
            })
            return

        if result["status"] != "completed":
            error_msg = f"Research failed: {result.get('output_text', result['status'])}"
            await task_manager.update_status(
                task_id, "failed", message=error_msg
            )
            event_queue.put_nowait({
                "type": "failed",
                "status": "failed",
                "message": error_msg,
            })
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

        # Upload files to project's object storage if configured
        storage_config = config.get("storage")
        if storage_config:
            from src.output.uploader import upload_report_files
            task_dir = settings.storage_path / task_id
            result_data["files"] = upload_report_files(
                task_id, task_dir, result_data["files"], storage_config
            )

        usage_data = {
            "model": config.get("model", settings.default_model),
            "total_tokens": result.get("total_tokens", 0),
            "research_time_seconds": round(elapsed, 1),
        }
        await task_manager.complete(task_id, result_data, usage_data)

        # Push final completion event to SSE queue
        event_queue.put_nowait({
            "type": "complete",
            "status": "completed",
            "report_url": result_data["files"][0]["url"] if result_data.get("files") else None,
            "summary": summary,
            "usage_data": usage_data,
        })

    except Exception as e:
        try:
            import structlog

            structlog.get_logger().error(
                "streaming_research_task_failed", task_id=task_id, error=str(e)
            )
        except ImportError:
            import logging

            logging.getLogger(__name__).error(f"streaming_research_task_failed: {task_id} - {e}")
        await task_manager.update_status(task_id, "failed", message=str(e))
        event_queue.put_nowait({
            "type": "failed",
            "status": "failed",
            "message": str(e),
        })
