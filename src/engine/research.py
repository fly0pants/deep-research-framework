from __future__ import annotations
import asyncio
import json
import re

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging

    class _StructlogAdapter:
        def __init__(self, name: str):
            self._logger = logging.getLogger(name)

        def info(self, msg: str, **kw: object) -> None:
            self._logger.info(msg, extra=kw)

        def warning(self, msg: str, **kw: object) -> None:
            self._logger.warning(msg, extra=kw)

        def error(self, msg: str, **kw: object) -> None:
            self._logger.error(msg, extra=kw)

    logger = _StructlogAdapter(__name__)


class ResearchEngine:
    """Research engine that supports both Responses API (Deep Research) and Chat Completions API."""

    def __init__(self, openai_client, default_model: str = "o3-deep-research"):
        self.client = openai_client
        self.default_model = default_model

    def _supports_responses_api(self, model: str) -> bool:
        """Check if model likely supports Deep Research Responses API."""
        deep_research_models = ("o3-deep-research", "o4-mini-deep-research")
        return any(m in model for m in deep_research_models)

    # ---- Responses API (Deep Research) path ----

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

    async def start(self, prompt: str, model: str | None = None, vector_store_id: str | None = None) -> str:
        tools = self._build_tools(vector_store_id)
        response = self.client.responses.create(
            model=model or self.default_model,
            input=prompt,
            tools=tools,
            background=True,
        )
        logger.info("research_started_responses", response_id=response.id)
        return response.id

    async def poll(self, response_id: str) -> dict:
        response = self.client.responses.retrieve(response_id)
        result = {"status": response.status}
        if response.status == "completed":
            result["output_text"] = response.output_text
            result["output"] = response.output
            result["total_tokens"] = response.usage.total_tokens if response.usage else 0
        return result

    # ---- Chat Completions API path ----

    def _build_call_api_tool(self) -> dict:
        """Build the call_api tool definition for function calling."""
        return {
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

    async def _run_chat(
        self,
        prompt: str,
        model: str | None = None,
        business_data: str | None = None,
        on_progress=None,
        api_configs: list[dict] | None = None,
        preparator=None,
        max_tool_iterations: int = 10,
    ) -> dict:
        """Run research using Chat Completions API.

        If api_configs and preparator are provided, enables function calling so the
        model can autonomously call project APIs for additional data.
        """
        use_model = model or self.default_model

        messages = []
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
        messages.append({"role": "user", "content": prompt})

        if on_progress:
            result = on_progress("researching")
            if asyncio.iscoroutine(result):
                await result

        logger.info("research_started_chat", model=use_model)

        # Determine whether tool use is enabled
        use_tools = bool(api_configs and preparator)
        tools = [self._build_call_api_tool()] if use_tools else None
        primary_api_config = api_configs[0] if api_configs else None

        total_tokens = 0
        api_call_stats = {"success": 0, "fail": 0, "errors": []}
        try:
            for iteration in range(max_tool_iterations + 1):
                kwargs: dict = {"model": use_model, "messages": messages, "max_tokens": 16000}
                if use_tools:
                    kwargs["tools"] = tools
                    if iteration == max_tool_iterations:
                        kwargs["tool_choice"] = "none"

                # Retry on transient errors (502/503/empty response)
                response = None
                for retry in range(3):
                    try:
                        response = self.client.chat.completions.create(**kwargs)
                        choice = response.choices[0]
                        has_content = choice.message.content or choice.message.tool_calls
                        if not has_content and retry < 2:
                            logger.warning("chat_empty_response_retry", retry=retry + 1)
                            await asyncio.sleep(3 * (retry + 1))
                            continue
                        break
                    except Exception as api_err:
                        err_str = str(api_err)
                        if retry < 2 and ("502" in err_str or "503" in err_str or "Bad Gateway" in err_str):
                            logger.warning("chat_api_retry", retry=retry + 1, error=err_str[:200])
                            await asyncio.sleep(2 ** (retry + 1))
                        else:
                            raise
                total_tokens += response.usage.total_tokens if response.usage else 0
                choice = response.choices[0]
                msg = choice.message

                # No tool calls → model is done
                if not msg.tool_calls:
                    output_text = msg.content or ""

                    # Check 1: Model explicitly signals insufficient data
                    insufficient_match = re.search(
                        r"<!--\s*INSUFFICIENT_DATA:\s*(.+?)\s*-->", output_text
                    )
                    if insufficient_match:
                        reason = insufficient_match.group(1)
                        logger.error(
                            "insufficient_data",
                            reason=reason,
                            api_stats=api_call_stats,
                        )
                        return {
                            "status": "data_unavailable",
                            "error": reason,
                            "api_call_stats": api_call_stats,
                        }

                    # Check 2: All API calls failed (safety net)
                    total_calls = api_call_stats["success"] + api_call_stats["fail"]
                    if total_calls > 0 and api_call_stats["success"] == 0:
                        logger.error(
                            "all_api_calls_failed",
                            total=total_calls,
                            errors=api_call_stats["errors"][:5],
                        )
                        return {
                            "status": "data_unavailable",
                            "error": "数据服务不可用：所有 API 调用均失败，无法生成有效报告",
                            "api_call_stats": api_call_stats,
                        }

                    return {
                        "status": "completed",
                        "output_text": output_text,
                        "total_tokens": total_tokens,
                        "format": self.parse_format(output_text),
                        "api_call_stats": api_call_stats,
                    }

                # Append assistant message (with tool_calls) to history
                messages.append(msg)

                # Execute each tool call and append results
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
                        from urllib.parse import urlparse, parse_qs
                        parsed = urlparse(endpoint)
                        endpoint = parsed.path
                        if parsed.query and not params:
                            params = {k: v[0] if len(v) == 1 else v for k, v in parse_qs(parsed.query).items()}

                    logger.info("tool_call_api", iteration=iteration, endpoint=endpoint, method=method, body=json.dumps(args, ensure_ascii=False)[:500])

                    result = await preparator.call_api(
                        api_config=primary_api_config,
                        endpoint=endpoint,
                        method=method,
                        params=params,
                        body=body,
                    )

                    if "error" in result:
                        api_call_stats["fail"] += 1
                        api_call_stats["errors"].append({
                            "endpoint": endpoint,
                            "method": method,
                            "error": result.get("error", ""),
                            "iteration": iteration,
                        })
                    else:
                        api_call_stats["success"] += 1

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

                if on_progress:
                    prog = on_progress(f"tool calls completed, iteration {iteration + 1}")
                    if asyncio.iscoroutine(prog):
                        await prog

            # Exhausted iterations — apply same checks
            total_calls = api_call_stats["success"] + api_call_stats["fail"]
            if total_calls > 0 and api_call_stats["success"] == 0:
                logger.error(
                    "all_api_calls_failed_exhausted",
                    total=total_calls,
                    errors=api_call_stats["errors"][:5],
                )
                return {
                    "status": "data_unavailable",
                    "error": "数据服务不可用：所有 API 调用均失败，无法生成有效报告",
                    "api_call_stats": api_call_stats,
                }
            last_content = next(
                (m.content for m in reversed(messages) if hasattr(m, "content") and m.content),
                "",
            )
            return {
                "status": "completed",
                "output_text": last_content,
                "total_tokens": total_tokens,
                "format": self.parse_format(last_content),
            }

        except Exception as e:
            logger.error("chat_research_failed", error=str(e))
            return {"status": "failed", "error": str(e)}

    # ---- Unified interface ----

    async def start_with_retry(self, prompt: str, model: str | None = None, vector_store_id: str | None = None, max_retries: int = 2) -> str:
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
        business_data: str | None = None,
        api_configs: list[dict] | None = None,
        preparator=None,
    ) -> dict:
        """Run research. Tries Responses API first, falls back to Chat Completions."""
        use_model = model or self.default_model

        # Try Responses API first if model supports it
        if self._supports_responses_api(use_model):
            try:
                response_id = await self.start_with_retry(prompt, use_model, vector_store_id)
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
            except Exception as e:
                logger.warning("responses_api_failed_falling_back", error=str(e))

        # Fallback to Chat Completions API (with optional tool use)
        return await self._run_chat(
            prompt=prompt,
            model=use_model,
            business_data=business_data,
            on_progress=on_progress,
            api_configs=api_configs,
            preparator=preparator,
        )

    def parse_format(self, text: str) -> str:
        match = re.search(r"<!--\s*FORMAT:\s*(\w+)\s*-->", text)
        if match:
            return match.group(1).lower()
        return "markdown"
