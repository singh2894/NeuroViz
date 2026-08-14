# app/parsers/nlp.py

import difflib
import json
import os
import re
import urllib.request

from pydantic import BaseModel

from .synonyms import AGG_MAP, INTENT_KEYWORDS, METRIC_SYNONYMS, TIME_GRAIN_MAP

KINDS = {"trend", "compare", "rank", "distribution"}
AGGS = {"min", "max", "mean", "sum"}
TIME_GRAINS = {"1d", "1w", "1mo", "1y"}


class Intent(BaseModel):
    kind: str  # "trend" | "compare" | "rank" | "distribution"
    metric: str | None = None  # generic metric name, e.g. "sales", "temperature"
    agg: str | None = None  # "min" | "max" | "mean" | "sum"
    time_grain: str | None = None  # "1d" | "1w" | "1mo" | "1y"
    columns: list[str] = []  # dataset columns the user referenced
    filters: dict[str, str] = {}
    text: str  # original user question


def detect_kind(text: str) -> str:
    t = text.lower()
    for kind, words in INTENT_KEYWORDS.items():
        if any(w in t for w in words):
            return kind
    return "trend"  # sensible default


def detect_time_grain(text: str) -> str | None:
    t = text.lower()
    for word, rule in TIME_GRAIN_MAP.items():
        if word in t:
            return rule
    return None


def detect_agg(text: str) -> str | None:
    t = text.lower()
    for agg, words in AGG_MAP.items():
        if any(w in t for w in words):
            return agg
    return None


def detect_metric(text: str) -> str | None:
    t = text.lower()
    for metric, words in METRIC_SYNONYMS.items():
        if any(w in t for w in words):
            return metric
    return None


def detect_columns(text: str, columns: list[str]) -> list[str]:
    """
    Fuzzy-match query words to dataset columns (stdlib difflib, no deps).
    Tolerates typos and case differences: 'bmi' -> 'BMI', 'wieght' -> 'weight'.
    """
    tokens = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    by_lower = {c.lower(): c for c in columns}
    found: list[str] = []
    for tok in tokens:
        match = difflib.get_close_matches(tok, by_lower, n=1, cutoff=0.8)
        if match and by_lower[match[0]] not in found:
            found.append(by_lower[match[0]])
    return found


def _parse_intent_rules(text: str, columns: list[str]) -> Intent:
    """Rule-based parser: keyword + fuzzy column matching, fully offline."""
    return Intent(
        kind=detect_kind(text),
        metric=detect_metric(text),
        agg=detect_agg(text),
        time_grain=detect_time_grain(text),
        columns=detect_columns(text, columns),
        text=text,
    )


def _chat(system: str, user: str) -> str:
    """One round-trip to the configured OpenAI-compatible endpoint."""
    headers = {
        "Content-Type": "application/json",
        # Some hosts (e.g. Groq behind Cloudflare) 403 the default urllib UA
        "User-Agent": "neuroviz/0.1",
    }
    if key := os.environ.get("LLM_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"

    body = json.dumps(
        {
            "model": os.environ.get("LLM_MODEL", "llama3.2"),
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode()
    req = urllib.request.Request(os.environ["LLM_API_URL"], data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def _extract_json(content: str) -> dict:
    """Pull a JSON object out of an LLM reply (tolerates ```json fences)."""
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.removeprefix("json")
    start, end = content.find("{"), content.rfind("}")
    return json.loads(content[start : end + 1])


def _parse_intent_llm(text: str, columns: list[str]) -> Intent:
    """
    Parse via any OpenAI-compatible chat endpoint (Ollama, Groq, Gemini,
    OpenRouter, ...). Configured entirely through env vars, stdlib only:
      LLM_API_URL  e.g. http://localhost:11434/v1/chat/completions
      LLM_MODEL    e.g. llama3.2 (default)
      LLM_API_KEY  optional, for hosted free tiers
    """
    system = (
        "Convert the user's data-visualization request to JSON. Reply with "
        'ONLY a JSON object like {"kind": "...", "metric": null, "agg": null, '
        '"time_grain": null, "columns": []}. '
        "kind: one of trend|compare|rank|distribution "
        "(trend = change over time; rank = which/top/most/least by category; "
        "compare = values across categories; distribution = histogram/spread). "
        "agg: one of min|max|mean|sum or null. "
        "time_grain: one of 1d|1w|1mo|1y or null. "
        "columns: names the user referenced, chosen only from this list: "
        f"{', '.join(columns) or '(none)'}"
    )
    content = _chat(system, text)
    data = _extract_json(content)
    return Intent(
        kind=data.get("kind") if data.get("kind") in KINDS else detect_kind(text),
        metric=data.get("metric"),
        agg=data.get("agg") if data.get("agg") in AGGS else None,
        time_grain=(
            data.get("time_grain") if data.get("time_grain") in TIME_GRAINS else None
        ),
        columns=[c for c in data.get("columns", []) if c in columns],
        text=text,
    )


def _data_profile(df) -> str:
    """Compact dataset profile the LLM can reason over (a few KB, not the data)."""
    parts = [
        f"Rows: {df.height}",
        "Columns: "
        + ", ".join(f"{c} ({t})" for c, t in zip(df.columns, df.dtypes, strict=False)),
    ]
    try:
        parts.append("Summary statistics:\n" + str(df.describe()))
    except Exception:
        pass
    for col, dtype in zip(df.columns, df.dtypes, strict=False):
        if str(dtype) == "String":
            try:
                top = df.get_column(col).value_counts(sort=True).head(5)
                vals = ", ".join(f"{r[0]} ({r[1]})" for r in top.iter_rows())
                parts.append(f"Top values of {col}: {vals}")
            except Exception:
                pass
    parts.append("First 5 rows:\n" + str(df.head(5)))
    return "\n\n".join(parts)[:8000]


def answer_question(text: str, df) -> str | None:
    """
    Answer a question about the dataset in words, using the free LLM.
    Returns None when no LLM is configured or the call fails.
    """
    if not os.environ.get("LLM_API_URL"):
        return None
    system = (
        "You are a careful data analyst. Answer the user's question about "
        "their dataset using ONLY the profile below (schema, summary "
        "statistics, top category values, sample rows). Cite the numbers you "
        "use. If the profile cannot answer the question, say exactly what is "
        "missing — never guess. Be concise.\n\n" + _data_profile(df)
    )
    try:
        return _chat(system, text)
    except Exception:
        return None


def parse_intent(text: str, columns: list[str] | None = None) -> Intent:
    """
    Parse a natural-language query into an Intent.

    If LLM_API_URL is set, a free OpenAI-compatible model (Ollama, Groq,
    Gemini, OpenRouter free tiers) does the parsing; otherwise — or on any
    error — the offline rule-based parser takes over, so the app always works.
    """
    if os.environ.get("LLM_API_URL"):
        try:
            return _parse_intent_llm(text, columns or [])
        except Exception:
            pass  # model down / bad reply must never break the app
    return _parse_intent_rules(text, columns or [])
