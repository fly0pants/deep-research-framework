"""Two-agent streaming research pipeline.

Agent 1 (data collector): Non-streaming tool-calling LLM that gathers data via APIs.
Agent 2 (report generator): Streaming LLM that produces an HTML report from collected data.
"""

from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import parse_qs, urlparse

from openai import OpenAI

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

        def error(self, msg: str, **kw: object) -> None:
            self._logger.error(msg, extra=kw)

    logger = _StructlogAdapter(__name__)


from .prompts.data_collector import build_data_collector_prompt
from .prompts.report_generator import build_report_generator_prompt


# ---------------------------------------------------------------------------
# Tool definition (mirrors research.py)
# ---------------------------------------------------------------------------

CALL_API_TOOL = {
    "type": "function",
    "function": {
        "name": "call_api",
        "description": (
            "Call a project API endpoint to retrieve additional data. "
            "IMPORTANT RULES:\n"
            "1. ONLY use endpoints documented in the API documentation in the system message. Do NOT guess endpoint names.\n"
            "2. Check the HTTP method in the docs — most data endpoints are POST, not GET.\n"
            "3. For POST endpoints, pass parameters in the 'body' field as a JSON object. Do NOT put parameters in the URL.\n"
            "4. The 'endpoint' field should be ONLY the path (e.g. '/api/data/unified-product-search'), never include query strings.\n"
            "5. For GET endpoints, pass parameters in the 'params' field.\n"
            "6. If an API call returns an error, read the error message carefully and adjust your parameters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "description": "API endpoint path only (e.g. /api/data/unified-product-search). Do NOT include query string parameters here.",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method — check the API docs for the correct method for each endpoint",
                },
                "params": {
                    "type": "object",
                    "description": "Query string parameters (for GET requests only)",
                    "additionalProperties": True,
                },
                "body": {
                    "type": "object",
                    "description": "JSON request body (for POST requests). Must match the documented request body schema.",
                    "additionalProperties": True,
                },
            },
            "required": ["endpoint", "method"],
        },
    },
}


class StreamingResearchPipeline:
    """Two-agent pipeline: data collection (tool-calling) + streaming report generation."""

    def __init__(self, openai_client: OpenAI, default_model: str = "gpt-4o"):
        self.client = openai_client
        self.default_model = default_model

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        query: str,
        project_config: dict,
        preparator,
        api_config: dict,
        on_progress=None,
        context: str | None = None,
        output_prefs: dict | None = None,
        business_data: str | None = None,
        model: str | None = None,
        max_tool_iterations: int = 10,
    ) -> dict:
        """Execute the full two-agent pipeline and return a result dict.

        Returns
        -------
        dict with keys: status, output_text, total_tokens, format, api_call_stats
        """
        use_model = model or self.default_model

        try:
            # --- Agent 1: collect data ---
            collector_result = await self._run_data_collector(
                query=query,
                project_config=project_config,
                preparator=preparator,
                api_config=api_config,
                on_progress=on_progress,
                context=context,
                output_prefs=output_prefs,
                business_data=business_data,
                model=use_model,
                max_tool_iterations=max_tool_iterations,
            )

            if collector_result["status"] != "completed":
                return collector_result

            collected_data: str = collector_result["collected_data"]
            agent1_tokens: int = collector_result["total_tokens"]
            api_call_stats: dict = collector_result["api_call_stats"]

            # Notify transition
            if on_progress:
                result = on_progress(
                    phase="generating",
                    stage="starting",
                    message="Data collection complete. Generating report...",
                    progress_pct=0,
                    partial_content=None,
                )
                if asyncio.iscoroutine(result):
                    await result

            # --- Agent 2: generate report ---
            report_result = await self._run_report_generator(
                query=query,
                collected_data=collected_data,
                output_prefs=output_prefs,
                on_progress=on_progress,
                model=use_model,
            )

            agent2_tokens: int = report_result.get("total_tokens", 0)

            if report_result["status"] != "completed":
                report_result["api_call_stats"] = api_call_stats
                report_result["total_tokens"] = agent1_tokens + agent2_tokens
                return report_result

            output_text = report_result["output_text"]
            return {
                "status": "completed",
                "output_text": output_text,
                "total_tokens": agent1_tokens + agent2_tokens,
                "format": self._parse_format(output_text),
                "api_call_stats": api_call_stats,
            }

        except Exception as exc:
            logger.error("streaming_pipeline_failed", error=str(exc))
            return {
                "status": "failed",
                "output_text": f"Pipeline error: {exc}",
                "total_tokens": 0,
                "format": "html",
                "api_call_stats": {"success": 0, "fail": 0, "errors": []},
            }

    # ------------------------------------------------------------------
    # Agent 1 — data collector (non-streaming, tool-calling)
    # ------------------------------------------------------------------

    async def _run_data_collector(
        self,
        query: str,
        project_config: dict,
        preparator,
        api_config: dict,
        on_progress,
        context: str | None,
        output_prefs: dict | None,
        business_data: str | None,
        model: str,
        max_tool_iterations: int,
    ) -> dict:
        """Run Agent 1: iterative tool-calling data collection.

        Returns
        -------
        dict with keys: status, collected_data (str), total_tokens, api_call_stats
        """
        system_prompt = build_data_collector_prompt(
            query=query,
            project_config=project_config,
            context=context,
            output_prefs=output_prefs,
        )

        messages: list[dict] = []
        if business_data:
            messages.append({
                "role": "system",
                "content": (
                    "IMPORTANT: Below is pre-fetched business data from project APIs. "
                    "This data is ALREADY AVAILABLE — read it carefully before making any API calls. "
                    "It may contain the exact data you need (e.g., rankings, product lists, IDs). "
                    "Extract product IDs, names, and metrics from this data first, then use the call_api tool "
                    "only for additional details not already present.\n\n"
                    f"{business_data}"
                ),
            })
        messages.append({"role": "user", "content": system_prompt})

        total_tokens = 0
        api_call_stats: dict = {"success": 0, "fail": 0, "errors": []}
        call_count = 0

        logger.info("data_collector_started", model=model)

        for iteration in range(max_tool_iterations + 1):
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "max_tokens": 16000,
                "tools": [CALL_API_TOOL],
            }
            # Last iteration: force text output
            if iteration == max_tool_iterations:
                kwargs["tool_choice"] = "none"

            # Retry logic for transient errors
            response = None
            for retry in range(3):
                try:
                    response = self.client.chat.completions.create(**kwargs)
                    choice = response.choices[0]
                    has_content = choice.message.content or choice.message.tool_calls
                    if not has_content and retry < 2:
                        logger.warning("collector_empty_response_retry", retry=retry + 1)
                        await asyncio.sleep(3 * (retry + 1))
                        continue
                    break
                except Exception as api_err:
                    err_str = str(api_err)
                    if retry < 2 and ("502" in err_str or "503" in err_str or "Bad Gateway" in err_str):
                        logger.warning("collector_api_retry", retry=retry + 1, error=err_str[:200])
                        await asyncio.sleep(2 ** (retry + 1))
                    else:
                        raise

            if response is None:
                break

            total_tokens += response.usage.total_tokens if response.usage else 0
            msg = response.choices[0].message

            # No tool calls → agent finished collecting
            if not msg.tool_calls:
                collected_data = msg.content or ""

                # Safety check: all API calls failed
                total_calls = api_call_stats["success"] + api_call_stats["fail"]
                if total_calls > 0 and api_call_stats["success"] == 0:
                    logger.error(
                        "all_api_calls_failed",
                        total=total_calls,
                        errors=api_call_stats["errors"][:5],
                    )
                    return {
                        "status": "data_unavailable",
                        "output_text": "数据服务不可用：所有 API 调用均失败，无法生成有效报告",
                        "total_tokens": total_tokens,
                        "format": "html",
                        "api_call_stats": api_call_stats,
                    }

                return {
                    "status": "completed",
                    "collected_data": collected_data,
                    "total_tokens": total_tokens,
                    "api_call_stats": api_call_stats,
                }

            # Append assistant message (with tool_calls) to history
            messages.append(msg)

            # Execute each tool call
            for tc in msg.tool_calls:
                if tc.function.name != "call_api":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Unknown tool: {tc.function.name}"}),
                    })
                    continue

                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"Invalid JSON arguments: {e}"}),
                    })
                    continue

                endpoint = args.get("endpoint", "")
                method = args.get("method", "GET")
                params = args.get("params")
                body = args.get("body")

                # Fix: model sometimes embeds query params in endpoint URL
                if "?" in endpoint:
                    parsed = urlparse(endpoint)
                    endpoint = parsed.path
                    if parsed.query and not params:
                        params = {
                            k: v[0] if len(v) == 1 else v
                            for k, v in parse_qs(parsed.query).items()
                        }

                logger.info(
                    "tool_call_api",
                    iteration=iteration,
                    endpoint=endpoint,
                    method=method,
                    body=json.dumps(args, ensure_ascii=False)[:500],
                )

                api_result = await preparator.call_api(
                    api_config=api_config,
                    endpoint=endpoint,
                    method=method,
                    params=params,
                    body=body,
                )

                if "error" in api_result:
                    api_call_stats["fail"] += 1
                    api_call_stats["errors"].append({
                        "endpoint": endpoint,
                        "method": method,
                        "error": api_result.get("error", ""),
                        "iteration": iteration,
                    })
                else:
                    api_call_stats["success"] += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(api_result, ensure_ascii=False),
                })

                call_count += 1

                # Progress callback after each API call
                if on_progress:
                    if call_count <= 15:
                        pct = round(call_count / 15 * 80)
                    else:
                        excess = call_count - 15
                        pct = round(80 + excess / (excess + 5) * 20)
                    pct = min(99, pct)

                    prog = on_progress(
                        phase="collecting",
                        stage="api_call",
                        message=f"API call #{call_count}: {method} {endpoint}",
                        progress_pct=pct,
                        partial_content=None,
                    )
                    if asyncio.iscoroutine(prog):
                        await prog

        # Exhausted iterations — apply same safety checks
        total_calls = api_call_stats["success"] + api_call_stats["fail"]
        if total_calls > 0 and api_call_stats["success"] == 0:
            logger.error(
                "all_api_calls_failed_exhausted",
                total=total_calls,
                errors=api_call_stats["errors"][:5],
            )
            return {
                "status": "data_unavailable",
                "output_text": "数据服务不可用：所有 API 调用均失败，无法生成有效报告",
                "total_tokens": total_tokens,
                "format": "html",
                "api_call_stats": api_call_stats,
            }

        # Extract last content from message history
        last_content = ""
        for m in reversed(messages):
            content = m.content if hasattr(m, "content") else m.get("content")
            if content and (hasattr(m, "role") and m.role == "assistant" or isinstance(m, dict) and m.get("role") == "assistant"):
                last_content = content
                break

        return {
            "status": "completed",
            "collected_data": last_content,
            "total_tokens": total_tokens,
            "api_call_stats": api_call_stats,
        }

    # ------------------------------------------------------------------
    # Agent 2 — report generator (streaming)
    # ------------------------------------------------------------------

    async def _run_report_generator(
        self,
        query: str,
        collected_data: str,
        output_prefs: dict | None,
        on_progress,
        model: str,
    ) -> dict:
        """Run Agent 2: streaming report generation.

        Returns
        -------
        dict with keys: status, output_text, total_tokens
        """
        system_prompt = build_report_generator_prompt(
            query=query,
            collected_data=collected_data,
            output_prefs=output_prefs,
        )

        logger.info("report_generator_started", model=model)

        try:
            stream = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=16000,
                stream=True,
                stream_options={"include_usage": True},
            )

            chunks_collected: list[str] = []
            chunk_index = 0
            total_tokens = 0

            for chunk in stream:
                # Extract usage from the final chunk
                if chunk.usage:
                    total_tokens = chunk.usage.total_tokens

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if delta and delta.content:
                    chunks_collected.append(delta.content)
                    chunk_index += 1

                    if on_progress:
                        pct = min(95, round(len("".join(chunks_collected)) / 30000 * 100))
                        prog = on_progress(
                            phase="generating",
                            stage="streaming",
                            chunk=delta.content,  # delta text for SSE streaming
                            progress_pct=pct,
                        )
                        if asyncio.iscoroutine(prog):
                            await prog

            full_html = "".join(chunks_collected)
            logger.info("report_generator_completed", length=len(full_html), tokens=total_tokens)

            return {
                "status": "completed",
                "output_text": full_html,
                "total_tokens": total_tokens,
            }

        except Exception as exc:
            logger.error("report_generator_failed", error=str(exc))
            return {
                "status": "failed",
                "output_text": f"Report generation error: {exc}",
                "total_tokens": 0,
            }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_format(text: str) -> str:
        """Extract format from ``<!-- FORMAT: xxx -->`` tag, default ``html``."""
        match = re.search(r"<!--\s*FORMAT:\s*(\w+)\s*-->", text)
        if match:
            return match.group(1).lower()
        return "html"
