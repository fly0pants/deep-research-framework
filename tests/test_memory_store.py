import pytest
from src.memory.store import UserMemoryStore


@pytest.fixture
async def store(tmp_path):
    s = UserMemoryStore(db_path=str(tmp_path / "test.db"))
    await s.init()
    yield s
    await s.close()


async def test_get_nonexistent_returns_none(store):
    result = await store.get("unknown_user")
    assert result is None


async def test_upsert_and_get(store):
    await store.upsert("user_abc", "喜欢简洁报告，关注游戏行业")
    result = await store.get("user_abc")
    assert result is not None
    assert result["user_id"] == "user_abc"
    assert "游戏" in result["memory"]
    assert result["version"] == 1


async def test_upsert_increments_version(store):
    await store.upsert("user_abc", "v1 profile")
    await store.upsert("user_abc", "v2 profile updated")
    result = await store.get("user_abc")
    assert result["version"] == 2
    assert "v2" in result["memory"]


async def test_upsert_updates_timestamp(store):
    await store.upsert("user_abc", "first")
    r1 = await store.get("user_abc")
    await store.upsert("user_abc", "second")
    r2 = await store.get("user_abc")
    assert r2["updated_at"] >= r1["updated_at"]
