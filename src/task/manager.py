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
