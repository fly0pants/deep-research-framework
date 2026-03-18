from src.engine.prompt_builder import build_research_prompt


def test_build_minimal_prompt():
    prompt = build_research_prompt(
        query="What is the trend?",
        project_config={"name": "test", "description": "A test"},
    )
    assert "What is the trend?" in prompt
    assert "test" in prompt


def test_build_prompt_with_system_instructions():
    prompt = build_research_prompt(
        query="Analyze data",
        project_config={
            "name": "admapix",
            "description": "Ad platform",
            "system_instructions": "Focus on ROI and CTR.",
        },
    )
    assert "Focus on ROI and CTR" in prompt


def test_build_prompt_with_context():
    prompt = build_research_prompt(
        query="Analyze data",
        project_config={"name": "test", "description": "Test"},
        context="Focus on video ads",
    )
    assert "Focus on video ads" in prompt


def test_build_prompt_with_output_prefs():
    prompt = build_research_prompt(
        query="Analyze data",
        project_config={"name": "test", "description": "Test"},
        output_prefs={
            "preferred_language": "zh-CN",
            "hints": ["Use charts for data"],
        },
    )
    assert "zh-CN" in prompt
    assert "Use charts for data" in prompt


def test_build_prompt_includes_source_labeling():
    prompt = build_research_prompt(
        query="test",
        project_config={"name": "test", "description": "Test"},
    )
    assert "[API]" in prompt
    assert "[Web]" in prompt


def test_build_prompt_includes_format_decision():
    prompt = build_research_prompt(
        query="test",
        project_config={"name": "test", "description": "Test"},
    )
    assert "format" in prompt.lower() or "格式" in prompt
