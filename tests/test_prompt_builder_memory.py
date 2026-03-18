from src.engine.prompt_builder import build_research_prompt


def test_prompt_includes_user_memory():
    prompt = build_research_prompt(
        query="分析Temu广告投放",
        project_config={"name": "admapix", "description": "Ad intelligence"},
        user_memory="- 角色：游戏投放优化师\n- 关注市场：东南亚\n- 输出风格：简洁，数据驱动",
    )
    assert "## User Profile" in prompt
    assert "游戏投放优化师" in prompt
    assert "东南亚" in prompt


def test_prompt_without_user_memory():
    prompt = build_research_prompt(
        query="分析Temu广告投放",
        project_config={"name": "admapix", "description": "Ad intelligence"},
    )
    assert "## User Profile" not in prompt


def test_prompt_with_none_user_memory():
    prompt = build_research_prompt(
        query="分析Temu广告投放",
        project_config={"name": "admapix", "description": "Ad intelligence"},
        user_memory=None,
    )
    assert "## User Profile" not in prompt
