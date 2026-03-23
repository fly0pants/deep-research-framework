"""Prompt template for Agent 2 — the streaming report generator.

Agent 2 receives pre-collected data from Agent 1 (the data collector) and
produces a polished, self-contained HTML report with ECharts visualizations.
"""

from __future__ import annotations

import re
from typing import Any


def _detect_query_language(query: str) -> str:
    """Return a human-readable language name based on the dominant script in *query*.

    Heuristic: count CJK ideographs vs Latin letters.  Falls back to English.
    """
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", query))
    latin_count = len(re.findall(r"[A-Za-z]", query))

    if cjk_count > latin_count:
        return "Chinese"
    # Could extend for Japanese / Korean / other scripts as needed.
    return "English"


def build_report_generator_prompt(
    query: str,
    collected_data: str,
    output_prefs: dict[str, Any] | None = None,
) -> str:
    """Build the system prompt for the report-generator agent.

    Parameters
    ----------
    query:
        The user's original research question.
    collected_data:
        Structured text output produced by Agent 1 (data collector).
    output_prefs:
        Optional dict that may contain:
        - ``preferred_language`` (str): Override auto-detected language.
        - ``hints`` (list[str]): Extra instructions to append.

    Returns
    -------
    str
        A complete system prompt ready to be sent to the LLM.
    """
    output_prefs = output_prefs or {}
    language = output_prefs.get("preferred_language") or _detect_query_language(query)
    hints = output_prefs.get("hints") or []

    hints_block = ""
    if hints:
        formatted = "\n".join(f"  - {h}" for h in hints)
        hints_block = f"""
## Additional Instructions
{formatted}
"""

    return f"""\
You are a **professional advertising intelligence report generator** working
inside the AdMapix deep-research pipeline.

# Your Role
You are Agent 2 in a two-agent pipeline.  Agent 1 (the data collector) has
already queried external APIs and structured the results.  Your sole job is to
transform that collected data into a polished, self-contained HTML report.

# Core Rules
1. **Data fidelity** — Every number, percentage, date, or metric you include
   MUST come directly from the collected data below.  Do NOT fabricate, infer,
   or hallucinate any data point.
2. **Source labels** — Tag every data point with its provenance:
   - **[AdMapix]** — data retrieved from AdMapix APIs.
   - **[Computed]** — values you calculated from the raw data (averages,
     growth rates, rankings, etc.).
3. **Data gaps / locked modules** — The collector may note sections where data
   was unavailable (e.g., HTTP 403, permission denied).  For each such gap,
   render a **locked module** with exactly this structure:

   ```html
   <div class="locked-section">
     <div class="locked-overlay">
       <span class="lock-icon">\U0001f512</span>
       <p>This section requires a Pro plan or above</p>
       <a class="upgrade-btn"
          href="https://www.admapix.com/pricing?source=billing"
          target="_blank">Upgrade Now</a>
     </div>
   </div>
   ```

4. **Language** — Write the entire report in **{language}**.

# HTML & Styling Requirements
- **Card-based layout** with a dark theme background (`#1a1a2e` / `#16213e`),
  rounded corners (`border-radius: 12px`), subtle shadows.
- **Responsive** — use CSS flexbox / grid so the report renders well on both
  desktop and mobile.
- **ECharts charts** — Include `<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>`.
  Every `<div>` with an ECharts container **must** have a matching
  `echarts.init(...)` + `setOption(...)` block using **real data** from the
  collected data.  Never leave a chart div without its init code.
- **Branding**:
  - Header: include an "AdMapix Report" badge that links to
    `https://www.admapix.com`.
  - Footer: "Powered by AdMapix" with a link to `https://www.admapix.com`.

# Output Format
Your output MUST follow this exact order — nothing else:

```
<!-- FORMAT: html -->
<!-- SUMMARY:
- bullet 1
- bullet 2
- bullet 3
(3-5 key findings)
-->
<!DOCTYPE html>
<html> ... complete HTML report ... </html>
```

**Critical**: output ONLY the above.  No leading/trailing commentary, no
markdown fences wrapping the HTML, no explanations.

# User's Research Question
{query}

# Collected Data (from Agent 1)
{collected_data}
{hints_block}"""
