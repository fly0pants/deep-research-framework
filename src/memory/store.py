from __future__ import annotations

import aiosqlite


class UserMemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db: aiosqlite.Connection | None = None

    async def init(self):
        self.db = await aiosqlite.connect(self.db_path)
        self.db.row_factory = aiosqlite.Row
        await self.db.execute("PRAGMA journal_mode=WAL")
        await self.db.execute("PRAGMA busy_timeout=5000")
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_memories (
                user_id TEXT PRIMARY KEY,
                memory TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                summary TEXT,
                created_at TEXT NOT NULL
            )
        """)
        await self.db.commit()

    async def close(self):
        if self.db:
            await self.db.close()

    async def get(self, user_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM user_memories WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def add_interaction(self, user_id: str, query: str, summary: str | None = None) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            "INSERT INTO user_interactions (user_id, query, summary, created_at) VALUES (?, ?, ?, ?)",
            (user_id, query, summary, now),
        )
        await self.db.commit()

    async def get_recent_interactions(self, user_id: str, limit: int = 10) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT query, summary, created_at FROM user_interactions WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def upsert(self, user_id: str, memory: str) -> None:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        await self.db.execute(
            """INSERT INTO user_memories (user_id, memory, version, created_at, updated_at)
               VALUES (?, ?, 1, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   memory = excluded.memory,
                   version = version + 1,
                   updated_at = excluded.updated_at""",
            (user_id, memory, now, now),
        )
        await self.db.commit()
