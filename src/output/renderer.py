from __future__ import annotations
import re
import subprocess
from pathlib import Path

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

FORMAT_EXT = {
    "markdown": ("report.md", "text/markdown"),
    "html": ("report.html", "text/html"),
    "pdf": ("report.html", "text/html"),
    "mixed": ("report.html", "text/html"),
}


class OutputRenderer:
    def __init__(self, output_base: Path | str):
        self.output_base = Path(output_base)

    async def render(
        self,
        task_id: str,
        format: str,
        content: str,
    ) -> dict:
        """Save rendered output and return file info."""
        task_dir = self.output_base / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        # Strip format tag
        content = re.sub(r"<!--\s*FORMAT:\s*\w+\s*-->\n?", "", content).strip()

        filename, mime_type = FORMAT_EXT.get(format, ("report.md", "text/markdown"))
        filepath = task_dir / filename
        filepath.write_text(content, encoding="utf-8")

        files = [
            {
                "name": filename,
                "url": f"/files/{task_id}/{filename}",
                "type": mime_type,
                "size": filepath.stat().st_size,
            }
        ]

        if format == "pdf":
            pdf_path = await self._html_to_pdf(filepath, task_dir)
            if pdf_path:
                files.append({
                    "name": pdf_path.name,
                    "url": f"/files/{task_id}/{pdf_path.name}",
                    "type": "application/pdf",
                    "size": pdf_path.stat().st_size,
                })

        logger.info("output_rendered", task_id=task_id, format=format, files=len(files))
        return {"format": format, "files": files}

    async def _html_to_pdf(self, html_path: Path, task_dir: Path) -> Path | None:
        """Convert HTML to PDF using WeasyPrint in Docker sandbox."""
        pdf_path = task_dir / "report.pdf"
        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network=none",
                    "--memory=512m",
                    "--cpus=1",
                    "-v", f"{task_dir}:/work:rw",
                    "dr-renderer",
                    "python", "-c",
                    "from weasyprint import HTML; HTML('/work/report.html').write_pdf('/work/report.pdf')",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and pdf_path.exists():
                return pdf_path
            logger.warning("pdf_conversion_failed", stderr=result.stderr)
        except Exception as e:
            logger.warning("pdf_conversion_error", error=str(e))
        return None
