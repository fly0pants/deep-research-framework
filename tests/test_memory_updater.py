import pytest
from unittest.mock import AsyncMock, MagicMock
from src.memory.updater import MemoryUpdater


def _make_mock_openai(response_text: str):
    """Create a mock OpenAI client that returns the given text."""
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = response_text
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create.return_value = response
    return client


def test_build_update_prompt_with_existing_memory():
    updater = MemoryUpdater(openai_client=MagicMock())
    prompt = updater._build_update_prompt(
        query="分析Temu的投放策略",
        summary="Temu在东南亚投放量大，视频素材为主",
        existing_memory="- 关注游戏行业",
    )
    assert "分析Temu" in prompt
    assert "关注游戏" in prompt
    assert "东南亚" in prompt


def test_build_update_prompt_without_existing_memory():
    updater = MemoryUpdater(openai_client=MagicMock())
    prompt = updater._build_update_prompt(
        query="分析Temu的投放策略",
        summary="Temu在东南亚投放量大",
        existing_memory=None,
    )
    assert "分析Temu" in prompt
    assert "首次" in prompt


def test_build_update_prompt_with_history():
    updater = MemoryUpdater(openai_client=MagicMock())
    prompt = updater._build_update_prompt(
        query="分析Shein的投放策略",
        summary="Shein在欧美投放量大",
        existing_memory="- 关注游戏行业",
        recent_interactions=[
            {"query": "分析Temu广告投放", "summary": "Temu东南亚为主", "created_at": "2026-03-18"},
            {"query": "Royal Match竞品分析", "summary": "休闲游戏赛道", "created_at": "2026-03-19"},
        ],
    )
    assert "历史研究记录" in prompt
    assert "Temu" in prompt
    assert "Royal Match" in prompt
    assert "Shein" in prompt


async def test_update_returns_new_memory():
    mock_client = _make_mock_openai(
        "- 关注电商和游戏行业\n- 关注市场：东南亚、欧美\n- 经常分析Temu、Shein"
    )
    updater = MemoryUpdater(openai_client=mock_client)
    result = await updater.generate_updated_memory(
        query="分析Shein的投放策略",
        summary="Shein在欧美投放量大",
        existing_memory=None,
        recent_interactions=[
            {"query": "分析Temu广告投放", "summary": "Temu东南亚为主", "created_at": "2026-03-18"},
        ],
    )
    assert "东南亚" in result or "Temu" in result
    mock_client.chat.completions.create.assert_called_once()
