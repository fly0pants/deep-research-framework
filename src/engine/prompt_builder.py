from __future__ import annotations

def _detect_query_language(query: str) -> str:
    """Simple heuristic: if >30% CJK chars, it's Chinese; otherwise English."""
    cjk = sum(1 for c in query if '\u4e00' <= c <= '\u9fff')
    if cjk / max(len(query), 1) > 0.3:
        return "Chinese"
    # Check for Japanese/Korean
    jp = sum(1 for c in query if '\u3040' <= c <= '\u30ff' or '\u31f0' <= c <= '\u31ff')
    kr = sum(1 for c in query if '\uac00' <= c <= '\ud7af')
    if jp > 0:
        return "Japanese"
    if kr > 0:
        return "Korean"
    return "English"


def build_research_prompt(
    query: str,
    project_config: dict,
    context: str | None = None,
    output_prefs: dict | None = None,
    user_memory: str | None = None,
) -> str:
    sections = []
    detected_lang = _detect_query_language(query)

    sections.append(f"""## Research Context

Project: {project_config['name']}
Description: {project_config.get('description', '')}
""")

    if user_memory:
        sections.append(f"""## User Profile

The following is a learned profile of the current user based on their past interactions. Adapt your research depth, focus areas, and terminology accordingly. NOTE: The language of this profile does NOT determine your output language — always follow the Language Rule section instead.

{user_memory}
""")

    if project_config.get("system_instructions"):
        sections.append(f"""## Expert Instructions

{project_config['system_instructions']}
""")

    _source_labels = {
        "Chinese": {"experience": "行业经验", "insufficient": "原因说明"},
        "Japanese": {"experience": "業界知見", "insufficient": "理由"},
        "Korean": {"experience": "업계 경험", "insufficient": "사유"},
    }
    labels = _source_labels.get(detected_lang, {"experience": "Industry Experience", "insufficient": "reason"})

    sections.append(f"""## Data Source Labeling

In your report, label each piece of information with its source:
- [API] — Data from project business APIs
- [Web] — Data from web search or publicly known facts
- [{labels['experience']}] — Your professional knowledge and industry experience (use when API data is unavailable but you can provide informed analysis)
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

    sections.append(f"""## Output Quality Gate

**CRITICAL:** Your output is the FINAL deliverable shown directly to the end user. There is NO follow-up interaction — the user cannot provide additional information, answer your questions, or retry.

**Absolutely forbidden:**
- Reports that ask the user to supplement info, confirm data, or perform manual steps to complete
- Reports with empty charts, "TBD" placeholders, or "to be supplemented" sections
- Responses like "if you provide XX data, I can do XX analysis"

**Data degradation strategy (by priority):**

1. **Data sufficient** → Generate a complete report with real data backing every section
2. **Partial data missing** → Apply these remedies:
   - Remove sections/charts where data is unavailable (do NOT leave empty shells)
   - When API data for a dimension is unavailable, provide analysis based on your industry knowledge, but label it [{labels['experience']}] not [API]
   - Keep usable data sections to ensure the report still delivers value
3. **Severely insufficient data** → Output a failure marker ONLY when ALL of the following are true:
   - The target product/entity cannot be found in API at all (multiple searches returned nothing)
   - AND no meaningful analysis can be derived from available data
   - In this case, output ONLY this line: `<!-- INSUFFICIENT_DATA: [{labels['insufficient']}] -->`

**Core principle: analyze with whatever data you have — a shorter report with real substance is better than a padded one.**
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

    sections.append(f"""## Language Rule (HIGHEST PRIORITY)

**The user's query is written in: {detected_lang}**

You MUST write your ENTIRE response in **{detected_lang}**, including:
- The report title, all headings, and body text
- The SUMMARY tag content
- Chart labels, legends, and axis titles
- All analysis and recommendations

This rule has the HIGHEST priority and overrides ALL other language instructions in this prompt, including any instructions in the Expert Instructions section above.
""")

    sections.append(f"""## Research Task

{query}
""")

    if context:
        sections.append(f"""## Additional Context

{context}
""")

    return "\n".join(sections)
