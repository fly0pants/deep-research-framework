import json
import uuid
from datetime import datetime, timezone

import asyncpg


class TaskManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: asyncpg.Pool | None = None

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

    async def close(self):
        if self.pool:
            await self.pool.close()

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
    ):
        now = datetime.now(timezone.utc)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET status=$1, stage=$2, message=$3, updated_at=$4 WHERE task_id=$5",
                status, stage, message, now, task_id,
            )

    async def complete(self, task_id: str, result_data: dict, usage_data: dict):
        now = datetime.now(timezone.utc)
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE tasks SET status='completed', result_data=$1, usage_data=$2, updated_at=$3 WHERE task_id=$4",
                json.dumps(result_data), json.dumps(usage_data), now, task_id,
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
