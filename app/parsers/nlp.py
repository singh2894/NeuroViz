# app/parsers/nlp.py

import spacy
from pydantic import BaseModel

from .synonyms import AGG_MAP, INTENT_KEYWORDS, METRIC_SYNONYMS, TIME_GRAIN_MAP

nlp = spacy.load("en_core_web_sm")


class Intent(BaseModel):
    kind: str  # "trend" | "compare" | "rank" | "distribution"
    metric: str | None = None  # generic metric name, e.g. "sales", "temperature"
    agg: str | None = None  # "min" | "max" | "mean" | "sum"
    time_grain: str | None = None  # "1d" | "1w" | "1mo" | "1y"
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


def parse_intent(text: str) -> Intent:
    """
    Very simple parser:
    - chooses intent kind (trend/compare/rank/distribution)
    - tries to detect metric name from synonyms
    - optional agg (min/max/avg/total)
    - optional time grain (daily/weekly/monthly/yearly)
    """
    kind = detect_kind(text)
    metric = detect_metric(text)
    agg = detect_agg(text)
    tg = detect_time_grain(text)

    return Intent(kind=kind, metric=metric, agg=agg, time_grain=tg, text=text)
