from __future__ import annotations


def build_research_prompt(
    query: str,
    project_config: dict,
    context: str | None = None,
    output_prefs: dict | None = None,
) -> str:
    sections = []

    sections.append(f"""## Research Context

Project: {project_config['name']}
Description: {project_config.get('description', '')}
""")

    if project_config.get("system_instructions"):
        sections.append(f"""## Expert Instructions

{project_config['system_instructions']}
""")

    sections.append("""## Data Source Labeling

In your report, label each piece of information with its source:
- [API] — Data from project business APIs
- [Web] — Data from web search or publicly known facts
- [行业经验] — Your professional knowledge and industry experience (use when API data is unavailable but you can provide informed analysis)
- [Computed] — Your own calculations or derivations based on the above data
""")

    sections.append("""## Tool Use Strategy

You have access to `call_api` tool to fetch additional data from project APIs. Follow this workflow:

**Step 1 — Extract from prefetch data first:**
The system message contains pre-fetched data (e.g., rankings, filter options). Scan it carefully for product IDs, names, and metrics before making any API calls.

**Step 2 — Search if needed:**
If the target product/entity is not in prefetch data, use the search APIs. Try multiple keyword variations:
- Original name, translated names, partial names, developer/company name
- Don't give up after one failed search — try at least 3 different keywords

**Step 3 — Collect detailed data:**
Once you have the product/entity ID, make multiple API calls to gather comprehensive data. For example, if analyzing an app's advertising:
- Call distribution APIs with different dimensions (country, media, type, trend, etc.)
- Call comparison/ranking APIs for competitive context
- Each API call is cheap — prefer more data over less

**Step 4 — Synthesize into report:**
Only generate the final report after collecting sufficient data. Never output a "need more info" response if you still have untried API calls available.

**⚠️ CRITICAL API calling rules:**
- ONLY use endpoints listed in the API documentation. Do NOT guess or invent endpoint names.
- Most data endpoints use POST method with a JSON body — check the docs for each endpoint.
- Pass POST parameters in the `body` field, NOT in the URL. The `endpoint` field should only contain the path.
- If a call returns an error, read the error details and adjust parameters according to the docs.
- Common workflow: search (unified-product-search) → get ID → distribution/detail calls.
""")

    sections.append("""## Output Quality Gate

**CRITICAL:** Your output is the FINAL deliverable shown directly to the end user. There is NO follow-up interaction — the user cannot provide additional information, answer your questions, or retry.

**绝对禁止：**
- 输出让用户补充信息、确认数据、或手动操作才能完成的报告
- 输出带有空图表、"待补充"、"TBD"占位符的模板报告
- 输出"如果给我XX数据，我可以做XX分析"之类的内容

**数据降级策略（按优先级执行）：**

1. **数据充足** → 生成完整报告，所有模块都有真实数据支撑
2. **部分数据缺失** → 采用以下补救措施：
   - 直接删掉无法获取数据的模块/图表（不要留空壳）
   - 如果某个维度的API数据拿不到，可以基于你的行业知识提供分析，但必须标注 [行业经验] 而非 [API]
   - 保留能用的数据模块，确保报告整体仍有价值
3. **数据严重不足** → 仅当以下情况同时满足时，输出失败标记：
   - 目标产品/实体在API中完全找不到（搜索多次都无结果）
   - 且无法从已有数据中推导出任何有意义的分析
   - 此时输出 ONLY 这一行：`<!-- INSUFFICIENT_DATA: [原因说明] -->`

**核心原则：有多少数据就做多少分析，宁可报告短一点也要保证每一段都有价值。**
""")

    format_section = """## Output Format Decision

Analyze the content and autonomously choose the best output format:
- **HTML with interactive charts** (plotly/echarts): For data-heavy, multi-dimensional analysis
- **PDF-ready HTML**: For formal text-heavy reports
- **Markdown**: For short, simple answers
- **Mixed**: HTML main report + supplementary files

Your response MUST include:
1. A `<!-- FORMAT: html|pdf|markdown|mixed -->` tag at the very beginning
2. A `<!-- SUMMARY: ... -->` tag containing a plain-text summary (3-5 bullet points, each on its own line starting with "- "). This summary will be shown to the user as a quick preview before they open the full report. Write it in the same language as the report. Example:
```
<!-- SUMMARY:
- Temu近30天广告投放以拉美和东南亚为核心，美国/巴西/墨西哥为Top3市场
- 视频素材占比超过95%，图片不到5%
- 投放趋势呈前高后低，2月中下旬为峰值，3月逐步回落
- 主要通过休闲游戏和工具类App的广告位触达用户
-->
```
3. The complete report content
4. If format is html: provide complete, self-contained HTML with embedded CSS/JS
5. If format is pdf: provide clean HTML suitable for PDF conversion

**HTML Styling Principles** (when output is html):
- Layout: card-based with subtle box-shadows, rounded corners (8–12px), generous padding/whitespace
- Color: professional palette — neutral backgrounds, one primary accent color, no garish combinations
- Typography: system font stack (`-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif`), 1.6 line-height, clear heading hierarchy (h1→h2→h3)
- Hero/header: dark background section at the top for visual impact
- Responsive: fluid widths, readable on mobile without horizontal scroll
- Charts: modern color schemes, always include legends and tooltips; prefer echarts or plotly
- **CRITICAL chart rule**: Every `<div>` with a chart class (e.g. `class="chart"`) MUST have a corresponding `echarts.init(document.getElementById('...'))` call in a `<script>` block at the bottom of `<body>`. If you create a chart container, you MUST write the JS to initialize it with real data. Empty chart containers with no JS are a serious defect.
- Restraint: clean and elegant — avoid over-decoration, excessive gradients, or clutter
"""

    if output_prefs:
        lang = output_prefs.get("preferred_language")
        if lang:
            format_section += f"\nPreferred output language: {lang}"
        hints = output_prefs.get("hints", [])
        if hints:
            format_section += "\n\nOutput hints:\n"
            for hint in hints:
                format_section += f"- {hint}\n"

    sections.append(format_section)

    sections.append(f"""## Research Task

{query}
""")

    if context:
        sections.append(f"""## Additional Context

{context}
""")

    return "\n".join(sections)
