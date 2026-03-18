import json
import os
from pathlib import Path

import httpx
try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging

    class _StructlogAdapter:
        """Minimal adapter so stdlib logger accepts structlog-style kwargs."""
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def info(self, msg: str, **kw: object) -> None:
            self._logger.info(msg, extra=kw)

        def warning(self, msg: str, **kw: object) -> None:
            self._logger.warning(msg, extra=kw)

    logger = _StructlogAdapter(__name__)


class DataPreparator:
    def __init__(self, openai_client, temp_dir: Path | str = "/tmp/dr-data"):
        self.client = openai_client
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def prefetch(self, api_config: dict) -> list[dict]:
        """Execute prefetch API calls defined in project config."""
        results = []
        base_url = api_config["base_url"].rstrip("/")
        auth = api_config.get("auth", {})
        headers = {}

        if auth.get("type") == "bearer":
            token = os.environ.get(auth["token_env"], "")
            if token:
                headers["Authorization"] = f"Bearer {token}"

        for pf in api_config.get("prefetch", []):
            endpoint = pf["endpoint"]
            params = pf.get("params", {})
            url = f"{base_url}{endpoint}"
            try:
                async with httpx.AsyncClient(timeout=30) as http:
                    resp = await http.get(url, params=params, headers=headers)
                    resp.raise_for_status()
                    results.append({
                        "endpoint": endpoint,
                        "data": resp.json(),
                    })
                    logger.info("prefetch_success", endpoint=endpoint)
            except Exception as e:
                logger.warning("prefetch_failed", endpoint=endpoint, error=str(e))
        return results

    async def create_vector_store(
        self,
        task_id: str,
        prefetch_results: list[dict],
        api_docs_content: str | None = None,
    ) -> str:
        """Create OpenAI Vector Store with prefetched data and API docs."""
        vs = self.client.vector_stores.create(name=f"dr-{task_id}")
        file_ids = []

        if prefetch_results:
            data_content = "# Prefetched Business Data\n\n"
            for pr in prefetch_results:
                data_content += f"## Endpoint: {pr['endpoint']}\n\n"
                data_content += f"```json\n{json.dumps(pr['data'], ensure_ascii=False, indent=2)}\n```\n\n"

            data_file = self.temp_dir / f"{task_id}_data.md"
            data_file.write_text(data_content)
            with open(data_file, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="assistants")
            file_ids.append(uploaded.id)

        if api_docs_content:
            docs_file = self.temp_dir / f"{task_id}_api_docs.md"
            docs_file.write_text(f"# API Documentation\n\n{api_docs_content}")
            with open(docs_file, "rb") as f:
                uploaded = self.client.files.create(file=f, purpose="assistants")
            file_ids.append(uploaded.id)

        for fid in file_ids:
            self.client.vector_stores.files.create(vector_store_id=vs.id, file_id=fid)

        logger.info("vector_store_created", vs_id=vs.id, files=len(file_ids))
        return vs.id

    async def cleanup(self, vector_store_id: str):
        """Delete temporary vector store after research completes."""
        try:
            self.client.vector_stores.delete(vector_store_id)
            logger.info("vector_store_cleaned", vs_id=vector_store_id)
        except Exception as e:
            logger.warning("vector_store_cleanup_failed", error=str(e))
