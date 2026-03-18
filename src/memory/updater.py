from __future__ import annotations

try:
    import structlog
    logger = structlog.get_logger()
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个用户画像分析器。根据用户的研究请求和报告摘要，生成或更新用户画像。

画像应包含以下维度（仅输出有证据支持的维度，不要猜测）：
- 语言偏好：中文/英文
- 输出风格：详细/简洁/数据驱动
- 专业程度：资深/中级/新手
- 关注行业：游戏/电商/工具/社交等
- 关注市场：国家/地区
- 关注维度：素材/投放/下载/收入等
- 关注产品：经常分析的产品或竞品
- 角色定位：投放优化师/产品经理/市场研究等
- 分析偏好：对比分析/趋势分析/单品深挖等

输出格式为 Markdown 列表，每行一个维度。只输出画像内容，不要输出任何解释。
如果有已有画像，在其基础上更新和补充，不要丢弃已有信息（除非新信息明确矛盾）。"""


class MemoryUpdater:
    def __init__(self, openai_client, model: str = "gpt-4o-mini"):
        self.client = openai_client
        self.model = model

    def _build_update_prompt(
        self,
        query: str,
        summary: str,
        existing_memory: str | None,
    ) -> str:
        parts = [f"## 本次研究请求\n{query}"]
        if summary:
            parts.append(f"## 报告摘要\n{summary}")
        if existing_memory:
            parts.append(f"## 已有用户画像\n{existing_memory}")
        else:
            parts.append("## 已有用户画像\n首次使用，无历史记录。")
        parts.append("请输出更新后的完整用户画像：")
        return "\n\n".join(parts)

    async def generate_updated_memory(
        self,
        query: str,
        summary: str,
        existing_memory: str | None,
    ) -> str:
        import asyncio

        user_prompt = self._build_update_prompt(query, summary, existing_memory)
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
