"""Prompt template for the data collector agent (Agent 1).

The data collector agent uses tool calling to fetch data from AdMapix APIs.
It does NOT write the final report — that is handled by a separate report
generator agent (Agent 2) which consumes the structured output produced here.
"""

import re


def _detect_query_language(query: str) -> str:
    """Detect the dominant language of the query string.

    Returns one of: "Chinese", "Japanese", "Korean", "English".

    Heuristics:
    - If the query contains hiragana or katakana → Japanese
    - If the query contains hangul → Korean
    - If >30 % of characters are CJK unified ideographs → Chinese
    - Otherwise → English
    """
    if not query:
        return "English"

    # Japanese-specific kana ranges
    has_hiragana_katakana = bool(
        re.search(r"[\u3040-\u309F\u30A0-\u30FF]", query)
    )
    if has_hiragana_katakana:
        return "Japanese"

    # Korean hangul range
    has_hangul = bool(re.search(r"[\uAC00-\uD7AF\u1100-\u11FF]", query))
    if has_hangul:
        return "Korean"

    # CJK unified ideographs
    cjk_chars = len(re.findall(r"[\u4E00-\u9FFF]", query))
    total_chars = len(query.replace(" ", ""))
    if total_chars > 0 and cjk_chars / total_chars > 0.3:
        return "Chinese"

    return "English"


def build_data_collector_prompt(
    query: str,
    project_config: dict,
    context: str | None = None,
    output_prefs: dict | None = None,
) -> str:
    """Build the system prompt for the data collector agent.

    Parameters
    ----------
    query:
        The user's research question / analysis request.
    project_config:
        Project configuration dict with at least ``name`` and ``description``.
    context:
        Optional extra context to inject into the prompt (e.g. prior findings).
    output_prefs:
        Optional dict that may contain ``preferred_language`` to override
        automatic language detection.
    """
    # Resolve output language
    if output_prefs and output_prefs.get("preferred_language"):
        language = output_prefs["preferred_language"]
    else:
        language = _detect_query_language(query)

    project_name = project_config.get("name", "Unknown Project")
    project_description = project_config.get("description", "")

    context_block = ""
    if context:
        context_block = (
            f"\n## Additional Context\n"
            f"{context}\n"
        )

    prompt = f"""\
# Role & Mission

You are a **data collection agent** for the ad intelligence platform **AdMapix**.
You are operating within project "{project_name}" ({project_description}).

Your sole job is to **gather ALL the raw data** needed for an advertising
analysis report.  A separate report-generator agent will consume your output
and produce the final report — you must NOT write the report yourself.

# How to Collect Data

Use the `call_api` tool to fetch data from AdMapix's platform APIs.  You should
be thorough and systematic:

1. **Search for the target product / advertiser / app** — try the exact query
   first, then attempt keyword variations (synonyms, abbreviations, translated
   names) if the initial search yields no results.
2. **Distribution data** — fetch breakdowns by:
   - Country / region
   - Media / ad network
   - Ad creative type (image, video, playable, etc.)
   - Time-series trend data
3. **Competitive / market data** — comparable products, market share, ranking.
4. **Download & revenue data** — installs, revenue estimates, store rankings.
5. **Creative samples** — retrieve example ad creatives (images, videos) with
   their metadata (first-seen, last-seen, media, country, etc.).

# Error Handling

- If an API call returns **403 Forbidden**, note the endpoint and reason, then
  move on to the next data point.  Do NOT retry 403 errors.
- If a search returns **no results**, try at least 2-3 alternative keyword
  variations before concluding the data is unavailable.

# Output Format

When you have finished collecting data, output a **structured data collection
report** in the following format.  Write it in **{language}**.

```
## Research Target
- Entity name: ...
- Entity ID (if found): ...
- Query interpretation: (how you understood the user's request)

## Data Collected

### <Category 1, e.g. "Ad Distribution by Country">
- API called: ...
- Request parameters: ...
- Result summary: (brief human-readable summary)
- Key data points: (bullet list of the most important numbers)
- Raw data:
  (include the full numerical dataset — tables, lists, JSON snippets, etc.)

### <Category 2>
...

## Data Gaps
- <What data could not be retrieved>: <reason — e.g. 403, no results, API not available>

## Analysis Hints
- (Key observations, anomalies, or patterns you noticed while collecting data
  that the report generator should pay attention to.)
```

# Critical Rules

1. **Include ALL numerical data.**  The report generator needs exact numbers,
   percentages, and time series — do not summarize them away.
2. **Do NOT write a final analysis report.**  Your output is raw data +
   structure, not a polished document.
3. **Do NOT skip collecting data.**  Even if early results seem sufficient,
   continue to fetch all relevant categories listed above.
4. **Write your output in {language}.**
{context_block}
# User's Research Query

{query}
"""
    return prompt
