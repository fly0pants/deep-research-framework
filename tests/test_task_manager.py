import pytest
from src.task.manager import TaskManager


@pytest.fixture
async def tm(tmp_path):
    manager = TaskManager(db_path=str(tmp_path / "test.db"))
    await manager.init()
    yield manager
    await manager.close()


async def test_create_task(tm):
    task = await tm.create(project="admapix", query="test query")
    assert task["task_id"].startswith("dr_")
    assert task["status"] == "pending"
    assert task["project"] == "admapix"


async def test_get_task(tm):
    created = await tm.create(project="admapix", query="test")
    fetched = await tm.get(created["task_id"])
    assert fetched is not None
    assert fetched["task_id"] == created["task_id"]


async def test_get_nonexistent(tm):
    result = await tm.get("dr_nonexistent")
    assert result is None


async def test_update_status(tm):
    task = await tm.create(project="admapix", query="test")
    await tm.update_status(task["task_id"], "processing", stage="researching", message="Working...")
    updated = await tm.get(task["task_id"])
    assert updated["status"] == "processing"
    assert updated["stage"] == "researching"


async def test_update_completed(tm):
    task = await tm.create(project="admapix", query="test")
    result_data = {"format": "html", "files": [], "summary": "done", "sources": []}
    usage_data = {"model": "o3-deep-research", "total_tokens": 1000, "research_time_seconds": 60}
    await tm.complete(task["task_id"], result_data, usage_data)
    updated = await tm.get(task["task_id"])
    assert updated["status"] == "completed"


async def test_cancel_task(tm):
    task = await tm.create(project="admapix", query="test")
    await tm.update_status(task["task_id"], "cancelled")
    updated = await tm.get(task["task_id"])
    assert updated["status"] == "cancelled"


async def test_count_active(tm):
    await tm.create(project="p1", query="q1")
    t2 = await tm.create(project="p2", query="q2")
    await tm.update_status(t2["task_id"], "processing")
    count = await tm.count_active()
    assert count == 2
