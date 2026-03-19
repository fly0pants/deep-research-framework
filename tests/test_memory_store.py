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


async def test_add_and_get_interactions(store):
    await store.add_interaction("user_abc", "分析Temu广告", "Temu东南亚为主")
    await store.add_interaction("user_abc", "Royal Match竞品", "休闲游戏赛道")
    await store.add_interaction("user_abc", "Shein投放策略", "欧美市场")
    results = await store.get_recent_interactions("user_abc", limit=2)
    assert len(results) == 2
    # Should be chronological order (oldest first)
    assert "Royal Match" in results[0]["query"]
    assert "Shein" in results[1]["query"]


async def test_get_interactions_empty(store):
    results = await store.get_recent_interactions("unknown_user")
    assert results == []
