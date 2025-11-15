# app/parsers/synonyms.py

# ---- Intent types ----
INTENT_KEYWORDS = {
    "trend": ["trend", "over time", "by month", "by date", "evolution"],
    "compare": ["compare", "vs", "versus", "difference between"],
    "rank": ["top", "bottom", "highest", "lowest", "rank"],
    "distribution": ["distribution", "histogram", "spread"],
}

# ---- Time grain words: how to group time ----
TIME_GRAIN_MAP = {
    "daily": "1d",
    "day": "1d",
    "weekly": "1w",
    "week": "1w",
    "monthly": "1mo",
    "month": "1mo",
    "yearly": "1y",
    "annual": "1y",
    "annually": "1y",
}

# ---- Aggregation words: min / max / avg / sum ----
AGG_MAP = {
    "min": ["lowest", "minimum", "min", "smallest"],
    "max": ["highest", "maximum", "max", "largest"],
    "mean": ["average", "avg", "mean"],
    "sum": ["total", "sum"],
}

# ---- Metric synonyms: you can extend this for each dataset ----
# Keys are generic metric names, values are words that may appear in queries.
METRIC_SYNONYMS = {
    "sales": ["sales", "revenue", "turnover"],
    "profit": ["profit", "margin"],
    "temperature": ["temperature", "temp"],
    "co2": ["co2", "emissions", "carbon"],
}
