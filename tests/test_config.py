import os
import pytest


def test_settings_loads_defaults():
    """Settings should have sensible defaults when env vars are minimal."""
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    os.environ.setdefault("API_TOKEN", "test-token")
    from src.config import Settings
    s = Settings()
    assert s.host == "0.0.0.0"
    assert s.port == 8000
    assert s.default_model == "o3-deep-research"
    assert s.max_concurrent_tasks == 5
    assert s.output_retention_days == 30


def test_settings_requires_openai_key(monkeypatch):
    """Settings should fail without OPENAI_API_KEY."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(Exception):
        from src.config import Settings
        Settings(_env_file=None)
