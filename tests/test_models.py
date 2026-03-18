import pytest
from src.api.models import ResearchRequest, TaskResponse, TaskStatus


def test_research_request_minimal():
    req = ResearchRequest(project="admapix", query="test query")
    assert req.project == "admapix"
    assert req.context is None
    assert req.callback_url is None


def test_research_request_full():
    req = ResearchRequest(
        project="admapix",
        query="test",
        context="extra context",
        callback_url="https://example.com/callback",
    )
    assert req.callback_url == "https://example.com/callback"


def test_research_request_rejects_http_callback():
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        ResearchRequest(
            project="admapix",
            query="test",
            callback_url="http://example.com/callback",
        )


def test_task_status_enum():
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.COMPLETED == "completed"


def test_task_response_pending():
    resp = TaskResponse(
        task_id="dr_abc123",
        status=TaskStatus.PENDING,
        created_at="2026-03-18T10:00:00Z",
    )
    assert resp.output is None
