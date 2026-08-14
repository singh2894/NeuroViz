# NeuroViz

NeuroViz turns plain-English questions into interactive charts, and takes a
dataset from raw upload to a trained, exportable model in one local app.
It runs entirely on your machine: parsing works offline through a rule-based
parser, and can optionally use any OpenAI-compatible endpoint (Ollama, Groq,
Gemini, OpenRouter) at no cost. Data stays in memory unless you explicitly
save it to disk.

## Features

**Ask questions in English.** Type "compare revenue by region" or "trend of
sales over time" and get an Altair chart. Column matching is fuzzy and
typo-tolerant. With a free LLM endpoint configured, parsing gets smarter and
the app can also answer questions in words, grounded in the dataset's actual
schema and statistics; without one, everything still works offline. The app
always shows what it parsed and which parser answered.

**Business-intelligence workflow.** An auto-generated dashboard (KPI cards,
chart grid, click-to-cross-filter, drill-down), a manual chart builder with
aggregation, faceting and pivot tables, and a one-click report export: a
single self-contained HTML file with KPIs, charts and exact statistics that
can be printed or emailed.

**Complete analyst pipeline.** Pages follow the order an analyst works:
upload, diagnose (missing values, duplicates, outliers, correlations,
leakage), clean, engineer features, visualize, then model.

**Sound machine learning.** Model training uses scikit-learn pipelines in
which one-hot encoding and imputation are fit inside each cross-validation
fold and train split, so evaluation never leaks information. Datasets with a
date column can use time-aware validation (train on the past, test on the
future). A fairness check reports per-group performance with disparity
alerts. Trained models are usable, not just scored: new rows can be scored
in-app to a predictions CSV, and the entire fitted pipeline can be
downloaded as a `.pkl`.

**Flexible data loading.** Files (CSV, Excel, Parquet), direct URLs, shared
Google Sheets links, or a built-in sample dataset. Sessions are memory-only
by default; an optional workspace save keeps datasets and pinned charts on
your own disk.

## Quick start

Requires Python 3.11–3.13 and Poetry.

```bash
git clone https://github.com/singh2894/NeuroViz.git
cd NeuroViz
poetry install
poetry run streamlit run app/main_app.py
```

The app opens at `http://localhost:8501`. Use "Try sample data" on the Data
page to explore without bringing your own file.

### Optional: LLM-assisted parsing

NeuroViz accepts any OpenAI-compatible chat endpoint. Two free options:

```bash
# Ollama — local, no signup (install from https://ollama.com, then: ollama pull llama3.2)
export LLM_API_URL="http://localhost:11434/v1/chat/completions"
export LLM_MODEL="llama3.2"

# Groq free tier — hosted
export LLM_API_URL="https://api.groq.com/openai/v1/chat/completions"
export LLM_MODEL="llama-3.1-8b-instant"
export LLM_API_KEY="gsk_..."
```

On Windows, use `setx NAME "value"` instead of `export`. If `LLM_API_URL` is
unset or the endpoint is unreachable, NeuroViz falls back to the offline
parser and says so in the interface.

## How it works

1. A query is parsed into a validated `Intent` (kind, metric, aggregation,
   time grain, columns, filters) — by the configured LLM if available,
   otherwise by keyword rules with `difflib` fuzzy column matching. Any LLM
   failure falls back to the offline path.
2. The dataset schema is inferred (numeric, categorical, date, wide-year
   columns).
3. Chart builders are tried in intent-specific priority order until one fits
   the data; the result is rendered with a caption stating how to read it.

Intent kinds: trend (time series), compare (across categories), rank
(top/bottom by category), distribution (histogram/spread).

## Project structure

```
app/
  main_app.py            Streamlit entry point, navigation, Data/Dashboard/
                         Build/Ask pages, workspace persistence
  pages_ml.py            Diagnose, Clean, Engineer, Features, Recommend,
                         Train pages
  data_io.py             File/URL/Google Sheets loading, sample dataset
  report.py              Self-contained HTML report export
  parsers/               Intent parsing (LLM + rule-based fallback)
  compilers/             Schema inference and Altair chart generation
  components/            Data filtering interface
  aie/                   ML engine: understanding, diagnostics, cleaning,
                         selection, models, evaluation, fairness, runner
tests/                   Test suite, including universal-compatibility fuzz
                         tests against pathological dataset shapes
```

The ML engine originated as the Automated-Insight-Engine repository; its
history is merged into this repository and development continues here.

## Development

```bash
poetry install --with dev
poetry run pytest        # tests
poetry run ruff check .  # lint
poetry run black .       # format
pre-commit install       # optional git hooks
```

Optional extras: `poetry install --extras boosters` adds XGBoost and
LightGBM to the model zoo. PNG chart export uses `vl-convert-python`
(installed by default).

## Roadmap

Done: multi-chart dashboards, HTML/PNG report export, workspace
persistence, Google Sheets and URL loading, leak-free model training with
scoring and export.

Planned: richer aggregations, an API endpoint for non-Streamlit
integrations, OAuth connectors (Shopify, QuickBooks), streaming data
sources.

## License

MIT. See the LICENSE file.

## Author

Simran Singh — [github.com/singh2894](https://github.com/singh2894).
Issues and suggestions: [GitHub issues](https://github.com/singh2894/NeuroViz/issues).
