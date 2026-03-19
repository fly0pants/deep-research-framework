from __future__ import annotations

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个用户画像分析器。根据用户的历史研究记录，生成或更新用户画像。

你的任务：
1. 分析用户的所有历史请求和报告摘要，发现有意义的模式
2. 自行决定哪些维度值得记录——不限于任何预设维度
3. 只记录有证据支持的特征，不要猜测
4. 关注那些能帮助未来研究更好服务该用户的信息

可能有价值的方向（仅作参考，不必局限于此）：
- 用户关注的行业、市场、产品
- 用户的分析习惯和偏好
- 用户的专业背景（从提问方式推断）
- 任何你认为对个性化服务有帮助的模式

输出格式为 Markdown 列表，每行一个特征。只输出画像内容，不要输出解释。
如果有已有画像，在其基础上更新和补充，不要丢弃已有信息（除非新证据明确矛盾）。
保持画像简洁，控制在15条以内。"""


class MemoryUpdater:
    def __init__(self, openai_client, model: str = "claude-haiku-4-5-20251001"):
        self.client = openai_client
        self.model = model

    def _build_update_prompt(
        self,
        query: str,
        summary: str,
        existing_memory: str | None,
        recent_interactions: list[dict] | None = None,
    ) -> str:
        parts = []

        # Recent history gives LLM full context to discover patterns
        if recent_interactions:
            history_lines = []
            for i, r in enumerate(recent_interactions, 1):
                line = f"{i}. [{r.get('created_at', '')}] {r['query']}"
                if r.get("summary"):
                    line += f"\n   摘要: {r['summary'][:200]}"
                history_lines.append(line)
            parts.append(f"## 历史研究记录（最近{len(recent_interactions)}条）\n" + "\n".join(history_lines))

        parts.append(f"## 本次研究请求\n{query}")
        if summary:
            parts.append(f"## 本次报告摘要\n{summary}")

        if existing_memory:
            parts.append(f"## 已有用户画像\n{existing_memory}")
        else:
            parts.append("## 已有用户画像\n首次使用，无历史记录。")

        parts.append("请基于以上所有信息，输出更新后的完整用户画像：")
        return "\n\n".join(parts)

    async def generate_updated_memory(
        self,
        query: str,
        summary: str,
        existing_memory: str | None,
        recent_interactions: list[dict] | None = None,
    ) -> str:
        import asyncio

        user_prompt = self._build_update_prompt(query, summary, existing_memory, recent_interactions)
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("memory_update_failed", error=str(e))
            return existing_memory or ""
