from __future__ import annotations

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


def _build_auth_headers(auth: dict, runtime_api_key: str | None = None) -> dict:
    """Build auth headers from auth config.

    Uses *runtime_api_key* if provided, otherwise falls back to the
    environment variable specified in the auth config.
    """
    headers = {}
    token = runtime_api_key or os.environ.get(auth.get("token_env", ""), "")
    if not token:
        return headers
    if auth.get("type") == "bearer":
        headers["Authorization"] = f"Bearer {token}"
    elif auth.get("type") == "header":
        header_name = auth.get("header_name", "X-API-Key")
        headers[header_name] = token
    return headers


def _strip_html_tags(obj):
    """Recursively strip HTML tags from string values in dicts/lists."""
    import re
    if isinstance(obj, str):
        return re.sub(r"<[^>]+>", "", obj)
    if isinstance(obj, dict):
        return {k: _strip_html_tags(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_html_tags(item) for item in obj]
    return obj


class DataPreparator:
    def __init__(self, openai_client, temp_dir: Path | str = "/tmp/dr-data", runtime_api_key: str | None = None):
        self.client = openai_client
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_api_key = runtime_api_key

    async def call_api(
        self,
        api_config: dict,
        endpoint: str,
        method: str = "GET",
        params: dict | None = None,
        body: dict | None = None,
    ) -> dict:
        """Make a dynamic API call using the project's auth config. Returns result or error dict."""
        base_url = api_config["base_url"].rstrip("/")
        headers = _build_auth_headers(api_config.get("auth", {}), self.runtime_api_key)
        url = f"{base_url}{endpoint}"
        method = method.upper()
        try:
            async with httpx.AsyncClient(timeout=60, verify=False) as http:
                for attempt in range(2):
                    try:
                        if method == "POST":
                            resp = await http.post(url, json=body or {}, params=params, headers=headers)
                        elif method == "PUT":
                            resp = await http.put(url, json=body or {}, params=params, headers=headers)
                        elif method == "DELETE":
                            resp = await http.delete(url, params=params, headers=headers)
                        else:
                            resp = await http.get(url, params=params or {}, headers=headers)
                        break
                    except (httpx.ReadTimeout, httpx.ConnectTimeout) as te:
                        if attempt == 0:
                            logger.warning("api_call_retry", endpoint=endpoint, error=type(te).__name__)
                            import asyncio
                            await asyncio.sleep(2)
                        else:
                            raise
                logger.info("api_call", endpoint=endpoint, method=method, status=resp.status_code)
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw": resp.text}
                if not resp.is_success:
                    return {
                        "error": f"HTTP {resp.status_code}",
                        "detail": data,
                        "hint": f"Check the API documentation for the correct parameters. Method: {method}, Endpoint: {endpoint}",
                    }
                return {"data": _strip_html_tags(data)}
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            logger.warning("api_call_failed", endpoint=endpoint, method=method, error=error_msg, url=url)
            return {"error": error_msg}

    async def prefetch(self, api_config: dict) -> list[dict]:
        """Execute prefetch API calls defined in project config."""
        results = []
        base_url = api_config["base_url"].rstrip("/")
        headers = _build_auth_headers(api_config.get("auth", {}), self.runtime_api_key)

        for pf in api_config.get("prefetch", []):
            endpoint = pf["endpoint"]
            params = pf.get("params", {})
            body = pf.get("body")
            method = pf.get("method", "GET").upper()
            url = f"{base_url}{endpoint}"
            try:
                async with httpx.AsyncClient(timeout=30, verify=False) as http:
                    if method == "POST" and body:
                        resp = await http.post(url, json=body, headers=headers)
                    else:
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
