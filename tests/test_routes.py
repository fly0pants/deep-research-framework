from __future__ import annotations
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch


@pytest.fixture
def app(tmp_path):
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "sk-test",
        "API_TOKEN": "test-token",
        "STORAGE_PATH": str(tmp_path / "output"),
        "PROJECTS_PATH": str(tmp_path / "projects"),
    }):
        proj_dir = tmp_path / "projects" / "testproj"
        proj_dir.mkdir(parents=True)
        (proj_dir / "config.yaml").write_text(
            'name: testproj\ndescription: "Test"\napis: []'
        )
        import src.config
        src.config.settings = None
        from src.main import create_app
        yield create_app()
        # Reset settings singleton after test
        src.config.settings = None


@pytest.fixture
async def client(app):
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


async def test_projects_requires_auth(client):
    resp = await client.get("/projects")
    # HTTPBearer returns 401 when no credentials provided (newer FastAPI)
    assert resp.status_code in (401, 403)


async def test_projects_with_auth(client):
    resp = await client.get("/projects", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["projects"]) == 1
    assert data["projects"][0]["name"] == "testproj"


async def test_submit_research_unknown_project(client):
    resp = await client.post(
        "/research",
        json={"project": "nonexistent", "query": "test"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 404


async def test_submit_research_success(client):
    with patch("src.api.routes.run_research_task") as mock_run:
        resp = await client.post(
            "/research",
            json={"project": "testproj", "query": "analyze trends"},
            headers={"Authorization": "Bearer test-token"},
        )
    assert resp.status_code == 202
    data = resp.json()
    assert data["task_id"].startswith("dr_")
    assert data["status"] == "pending"


async def test_get_task_not_found(client):
    resp = await client.get(
        "/research/dr_nonexistent",
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 404


async def test_cancel_task(client):
    with patch("src.api.routes.run_research_task"):
        create_resp = await client.post(
            "/research",
            json={"project": "testproj", "query": "test"},
            headers={"Authorization": "Bearer test-token"},
        )
    task_id = create_resp.json()["task_id"]

    resp = await client.post(
        f"/research/{task_id}/cancel",
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"
