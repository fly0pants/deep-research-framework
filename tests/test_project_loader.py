import pytest
from pathlib import Path
from src.engine.project_loader import ProjectLoader


@pytest.fixture
def projects_dir(tmp_path):
    proj = tmp_path / "test_project"
    proj.mkdir()
    (proj / "config.yaml").write_text("""
name: test_project
description: "A test project"
apis:
  - name: test_api
    base_url: https://api.test.com/v1
    auth:
      type: bearer
      token_env: TEST_API_KEY
    docs_file: api_docs/test_api.yaml
    prefetch:
      - endpoint: /data
        params: {limit: 10}
model: o3-deep-research
system_instructions: "You are a test analyst."
""")
    api_docs = proj / "api_docs"
    api_docs.mkdir()
    (api_docs / "test_api.yaml").write_text("openapi: '3.0.0'\ninfo:\n  title: Test\n  version: '1.0'")
    (proj / "output_prefs.yaml").write_text("""
preferred_language: zh-CN
hints:
  - "Use charts for data"
""")
    return tmp_path


@pytest.fixture
def loader(projects_dir):
    return ProjectLoader(projects_dir)


def test_list_projects(loader):
    projects = loader.list_projects()
    assert len(projects) == 1
    assert projects[0]["name"] == "test_project"


def test_load_project(loader):
    config = loader.load("test_project")
    assert config["name"] == "test_project"
    assert len(config["apis"]) == 1
    assert config["apis"][0]["name"] == "test_api"
    assert config["system_instructions"] == "You are a test analyst."


def test_load_output_prefs(loader):
    prefs = loader.load_output_prefs("test_project")
    assert prefs["preferred_language"] == "zh-CN"


def test_load_api_docs(loader):
    docs = loader.load_api_docs("test_project", "api_docs/test_api.yaml")
    assert "openapi" in docs


def test_load_nonexistent_project(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent")


def test_load_output_prefs_missing(loader, projects_dir):
    proj = projects_dir / "no_prefs"
    proj.mkdir()
    (proj / "config.yaml").write_text("name: no_prefs\ndescription: test\napis: []")
    loader2 = ProjectLoader(projects_dir)
    prefs = loader2.load_output_prefs("no_prefs")
    assert prefs is None
