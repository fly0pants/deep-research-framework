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


async def test_update_returns_new_memory():
    mock_client = _make_mock_openai(
        "- 角色：广告投放优化师\n- 关注市场：东南亚\n- 输出风格：数据驱动"
    )
    updater = MemoryUpdater(openai_client=mock_client)
    result = await updater.generate_updated_memory(
        query="分析Temu的投放策略",
        summary="Temu在东南亚投放量大",
        existing_memory=None,
    )
    assert "东南亚" in result
    mock_client.chat.completions.create.assert_called_once()
