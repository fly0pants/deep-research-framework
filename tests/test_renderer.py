from __future__ import annotations
import pytest
from pathlib import Path
from src.output.renderer import OutputRenderer


@pytest.fixture
def renderer(tmp_path):
    return OutputRenderer(output_base=tmp_path)


async def test_save_markdown(renderer, tmp_path):
    result = await renderer.render(
        task_id="dr_test1",
        format="markdown",
        content="# Report\n\nFindings here.",
    )
    assert result["format"] == "markdown"
    assert len(result["files"]) == 1
    assert result["files"][0]["name"] == "report.md"
    saved = (tmp_path / "dr_test1" / "report.md").read_text()
    assert "Findings here" in saved


async def test_save_html(renderer, tmp_path):
    html = "<!-- FORMAT: html -->\n<html><body><h1>Report</h1></body></html>"
    result = await renderer.render(
        task_id="dr_test2",
        format="html",
        content=html,
    )
    assert result["format"] == "html"
    assert result["files"][0]["name"] == "report.html"
    saved = (tmp_path / "dr_test2" / "report.html").read_text()
    assert "<h1>Report</h1>" in saved


async def test_strip_format_tag(renderer, tmp_path):
    html = "<!-- FORMAT: html -->\n<html><body>Content</body></html>"
    result = await renderer.render(
        task_id="dr_test3",
        format="html",
        content=html,
    )
    saved = (tmp_path / "dr_test3" / "report.html").read_text()
    assert "<!-- FORMAT:" not in saved


async def test_output_info_structure(renderer):
    result = await renderer.render(
        task_id="dr_test4",
        format="markdown",
        content="# Test",
    )
    assert "format" in result
    assert "files" in result
    f = result["files"][0]
    assert "name" in f
    assert "url" in f
    assert "type" in f
    assert "size" in f
