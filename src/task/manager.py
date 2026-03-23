import json
import os
import uuid
from datetime import datetime, timezone

import asyncpg

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# Supabase connection for syncing task status to admapix
SUPABASE_URL = os.environ.get("SUPABASE_DATABASE_URL", "")


class TaskManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None
        self._supabase_pool: asyncpg.Pool | None = None

    async def init(self):
        self.pool = await asyncpg.create_pool(self.database_url, min_size=2, max_size=10)
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    query TEXT NOT NULL,
                    context TEXT,
                    source TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    stage TEXT,
                    message TEXT,
                    result_data TEXT,
                    usage_data TEXT,
                    created_at TIMESTAMPTZ NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL
                )
            """)
            # Migrate: add streaming progress fields
            for col, col_type, default in [
                ("progress_pct", "INTEGER", "0"),
                ("partial_content", "TEXT", "NULL"),
                ("phase", "TEXT", "'pending'"),
            ]:
                await conn.execute(f"""
                    ALTER TABLE tasks ADD COLUMN IF NOT EXISTS {col} {col_type} DEFAULT {default}
                """)
        # Init Supabase connection pool
        if SUPABASE_URL:
            try:
                self._supabase_pool = await asyncpg.create_pool(
                    SUPABASE_URL, min_size=1, max_size=3,
                    statement_cache_size=0,  # Required for Supabase pgbouncer
                )
                logger.info("supabase_pool_initialized")
            except Exception as e:
                logger.warning("supabase_pool_init_failed", error=str(e))

    async def close(self):
        if self.pool:
            await self.pool.close()
        if self._supabase_pool:
            await self._supabase_pool.close()

    async def _sync_to_supabase(
        self,
        task_id: str,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        result_data: dict | None = None,
        usage_data: dict | None = None,
        *,
        progress_pct: int | None = None,
        partial_content: str | None = None,
        phase: str | None = None,
    ):
        """Sync task status to admapix's deep_research_tasks table in Supabase."""
        if not self._supabase_pool:
            return
        try:
            summary = None
            report_format = None
            report_files = None
            sources = None
            error_message = None

            if result_data:
                summary = result_data.get("summary")
                report_format = result_data.get("format")
                report_files = json.dumps(result_data.get("files")) if result_data.get("files") else None
                sources = json.dumps(result_data.get("sources")) if result_data.get("sources") else None

            if status == "failed":
                error_message = message

            async with self._supabase_pool.acquire() as conn:
                await conn.execute(
                    """UPDATE deep_research_tasks SET
                        status = $1, stage = $2, message = $3, summary = $4,
                        report_format = $5, report_files = $6,
                        sources = $7, usage_data = $8, error_message = $9,
                        progress_pct = COALESCE($10, progress_pct),
                        partial_content = $11, phase = $12,
                        updated_at = NOW()
                    WHERE task_id = $13""",
                    status, stage, message, summary,
                    report_format, report_files,
                    sources, json.dumps(usage_data) if usage_data else None,
                    error_message, progress_pct, partial_content, phase,
                    task_id,
                )
        except Exception as e:
            logger.warning("supabase_sync_failed", task_id=task_id, error=str(e))

    async def create(
        self,
        project: str,
        query: str,
        context: str | None = None,
        source: str | None = None,
    ) -> dict:
        task_id = f"dr_{uuid.uuid4()}"
        now = datetime.now(timezone.utc)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO tasks (task_id, project, query, context, source, status, created_at, updated_at)
                   VALUES ($1, $2, $3, $4, $5, 'pending', $6, $6)""",
                task_id, project, query, context, source, now,
            )
        return {"task_id": task_id, "status": "pending", "project": project, "created_at": now.isoformat()}

    async def get(self, task_id: str) -> dict | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM tasks WHERE task_id = $1", task_id)
        if row is None:
            return None
        result = dict(row)
        # Convert datetime to ISO string for JSON compatibility
        for key in ("created_at", "updated_at"):
            if result.get(key) and hasattr(result[key], "isoformat"):
                result[key] = result[key].isoformat()
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
        *,
        progress_pct: int | None = None,
        partial_content: str | None = None,
        phase: str | None = None,
    ):
        now = datetime.now(timezone.utc)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """UPDATE tasks SET status=$1, stage=$2, message=$3,
                   progress_pct=COALESCE($4, progress_pct), partial_content=$5, phase=$6,
                   updated_at=$7 WHERE task_id=$8""",
                status, stage, message, progress_pct, partial_content, phase, now, task_id,
            )
        await self._sync_to_supabase(
            task_id, status, stage=stage, message=message,
            progress_pct=progress_pct, partial_content=partial_content, phase=phase,
        )

    async def complete(self, task_id: str, result_data: dict, usage_data: dict):
        now = datetime.now(timezone.utc)
        async with self.pool.acquire() as conn:
            await conn.execute(
                """UPDATE tasks SET status='completed', result_data=$1, usage_data=$2,
                   progress_pct=100, partial_content=NULL, phase='completed',
                   updated_at=$3 WHERE task_id=$4""",
                json.dumps(result_data), json.dumps(usage_data), now, task_id,
            )
        await self._sync_to_supabase(
            task_id, "completed", result_data=result_data, usage_data=usage_data,
            progress_pct=100, partial_content=None, phase="completed",
        )

    async def count_active(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) FROM tasks WHERE status IN ('pending', 'processing')"
            )
            return row[0]

    async def list_all(self, limit: int = 20, offset: int = 0, source: str | None = None) -> list[dict]:
        async with self.pool.acquire() as conn:
            if source == "api":
                # Non-website: everything except admapix-website
                rows = await conn.fetch(
                    "SELECT * FROM tasks WHERE (source IS NULL OR source != 'admapix-website') ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, offset,
                )
            elif source:
                rows = await conn.fetch(
                    "SELECT * FROM tasks WHERE source = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                    source, limit, offset,
                )
            else:
                rows = await conn.fetch(
                    "SELECT * FROM tasks ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, offset,
                )
        results = []
        for row in rows:
            r = dict(row)
            for key in ("created_at", "updated_at"):
                if r.get(key) and hasattr(r[key], "isoformat"):
                    r[key] = r[key].isoformat()
            if r.get("result_data"):
                r["result_data"] = json.loads(r["result_data"])
            if r.get("usage_data"):
                r["usage_data"] = json.loads(r["usage_data"])
            results.append(r)
        return results

    async def count_all(self, source: str | None = None) -> int:
        async with self.pool.acquire() as conn:
            if source == "api":
                row = await conn.fetchrow(
                    "SELECT COUNT(*) FROM tasks WHERE (source IS NULL OR source != 'admapix-website')"
                )
            elif source:
                row = await conn.fetchrow(
                    "SELECT COUNT(*) FROM tasks WHERE source = $1", source,
                )
            else:
                row = await conn.fetchrow("SELECT COUNT(*) FROM tasks")
            return row[0]
