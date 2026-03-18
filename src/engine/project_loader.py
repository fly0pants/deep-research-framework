from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml


class ProjectLoader:
    def __init__(self, projects_path: Path | str):
        self.projects_path = Path(projects_path)

    def list_projects(self) -> list[dict]:
        results = []
        if not self.projects_path.exists():
            return results
        for proj_dir in sorted(self.projects_path.iterdir()):
            config_file = proj_dir / "config.yaml"
            if proj_dir.is_dir() and config_file.exists():
                config = yaml.safe_load(config_file.read_text())
                results.append({
                    "name": config.get("name", proj_dir.name),
                    "description": config.get("description", ""),
                    "apis": len(config.get("apis", [])),
                })
        return results

    def load(self, project_name: str) -> dict:
        proj_dir = self.projects_path / project_name
        config_file = proj_dir / "config.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Project '{project_name}' not found")
        return yaml.safe_load(config_file.read_text())

    def load_output_prefs(self, project_name: str) -> Optional[dict]:
        prefs_file = self.projects_path / project_name / "output_prefs.yaml"
        if not prefs_file.exists():
            return None
        return yaml.safe_load(prefs_file.read_text())

    def load_api_docs(self, project_name: str, docs_file: str) -> str:
        docs_path = self.projects_path / project_name / docs_file
        if not docs_path.exists():
            raise FileNotFoundError(f"API docs not found: {docs_path}")
        return docs_path.read_text()
