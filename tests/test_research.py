from __future__ import annotations
import pytest
from unittest.mock import MagicMock
from src.engine.research import ResearchEngine


@pytest.fixture
def mock_openai():
    client = MagicMock()
    return client


@pytest.fixture
def engine(mock_openai):
    return ResearchEngine(openai_client=mock_openai, default_model="o3-deep-research")


def test_build_tools_with_vector_store(engine):
    tools = engine._build_tools(vector_store_id="vs_123")
    tool_types = [t["type"] for t in tools]
    assert "web_search_preview" in tool_types
    assert "code_interpreter" in tool_types
    assert "file_search" in tool_types


def test_build_tools_without_vector_store(engine):
    tools = engine._build_tools(vector_store_id=None)
    tool_types = [t["type"] for t in tools]
    assert "web_search_preview" in tool_types
    assert "code_interpreter" in tool_types
    assert "file_search" not in tool_types


@pytest.mark.asyncio
async def test_start_research(engine, mock_openai):
    mock_response = MagicMock()
    mock_response.id = "resp_test123"
    mock_response.status = "queued"
    mock_openai.responses.create.return_value = mock_response

    resp_id = await engine.start(
        prompt="Analyze data trends",
        model="o3-deep-research",
        vector_store_id="vs_123",
    )
    assert resp_id == "resp_test123"
    mock_openai.responses.create.assert_called_once()
    call_kwargs = mock_openai.responses.create.call_args[1]
    assert call_kwargs["background"] is True


@pytest.mark.asyncio
async def test_poll_completed(engine, mock_openai):
    mock_response = MagicMock()
    mock_response.status = "completed"
    mock_response.output_text = "# Research Report\nFindings here."
    mock_response.output = []
    mock_response.usage = MagicMock(total_tokens=5000)
    mock_openai.responses.retrieve.return_value = mock_response

    result = await engine.poll("resp_test123")
    assert result["status"] == "completed"
    assert "Research Report" in result["output_text"]


@pytest.mark.asyncio
async def test_poll_in_progress(engine, mock_openai):
    mock_response = MagicMock()
    mock_response.status = "in_progress"
    mock_openai.responses.retrieve.return_value = mock_response

    result = await engine.poll("resp_test123")
    assert result["status"] == "in_progress"


def test_parse_format_tag(engine):
    text = "<!-- FORMAT: html -->\n<html>...</html>"
    fmt = engine.parse_format(text)
    assert fmt == "html"


def test_parse_format_default(engine):
    text = "Just some markdown text."
    fmt = engine.parse_format(text)
    assert fmt == "markdown"
