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
- [API] — Data from project business APIs (available via file_search)
- [Web] — Data from web search
- [Computed] — Your own calculations or derivations
""")

    format_section = """## Output Format Decision

Analyze the content and autonomously choose the best output format:
- **HTML with interactive charts** (plotly/echarts): For data-heavy, multi-dimensional analysis
- **PDF-ready HTML**: For formal text-heavy reports
- **Markdown**: For short, simple answers
- **Mixed**: HTML main report + supplementary files

Your response MUST include:
1. A `<!-- FORMAT: html|pdf|markdown|mixed -->` tag at the very beginning
2. The complete report content
3. If format is html: provide complete, self-contained HTML with embedded CSS/JS
4. If format is pdf: provide clean HTML suitable for PDF conversion
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
