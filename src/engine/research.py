from __future__ import annotations
import asyncio
import re

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


class ResearchEngine:
    def __init__(self, openai_client, default_model: str = "o3-deep-research"):
        self.client = openai_client
        self.default_model = default_model

    def _build_tools(self, vector_store_id: str | None = None) -> list[dict]:
        tools = [
            {"type": "web_search_preview"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
        ]
        if vector_store_id:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
            })
        return tools

    async def start(
        self,
        prompt: str,
        model: str | None = None,
        vector_store_id: str | None = None,
    ) -> str:
        """Start a background Deep Research request. Returns response ID."""
        tools = self._build_tools(vector_store_id)
        response = self.client.responses.create(
            model=model or self.default_model,
            input=prompt,
            tools=tools,
            background=True,
        )
        logger.info("research_started", response_id=response.id, model=model or self.default_model)
        return response.id

    async def poll(self, response_id: str) -> dict:
        """Poll for research completion. Returns status and result."""
        response = self.client.responses.retrieve(response_id)
        result = {"status": response.status}

        if response.status == "completed":
            result["output_text"] = response.output_text
            result["output"] = response.output
            result["total_tokens"] = (
                response.usage.total_tokens if response.usage else 0
            )

        return result

    async def start_with_retry(
        self,
        prompt: str,
        model: str | None = None,
        vector_store_id: str | None = None,
        max_retries: int = 2,
    ) -> str:
        """Start research with retry on failure."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self.start(prompt, model, vector_store_id)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning("start_retry", attempt=attempt + 1, wait=wait, error=str(e))
                    await asyncio.sleep(wait)
        raise last_error

    async def run_and_wait(
        self,
        prompt: str,
        model: str | None = None,
        vector_store_id: str | None = None,
        on_progress=None,
        timeout_seconds: int = 600,
    ) -> dict:
        """Start research and poll until completion with exponential backoff."""
        response_id = await self.start_with_retry(prompt, model, vector_store_id)

        delays = [5, 10, 20, 30]
        delay_idx = 0
        elapsed = 0

        while elapsed < timeout_seconds:
            result = await self.poll(response_id)

            if result["status"] == "completed":
                result["format"] = self.parse_format(result.get("output_text", ""))
                return result

            if result["status"] in ("failed", "cancelled"):
                return result

            if on_progress:
                await on_progress(result["status"])

            delay = delays[min(delay_idx, len(delays) - 1)]
            delay_idx += 1
            await asyncio.sleep(delay)
            elapsed += delay

        return {"status": "failed", "error": f"Research timed out after {timeout_seconds}s"}

    def parse_format(self, text: str) -> str:
        """Extract format tag from model output."""
        match = re.search(r"<!--\s*FORMAT:\s*(\w+)\s*-->", text)
        if match:
            return match.group(1).lower()
        return "markdown"
