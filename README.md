# NeuroViz 📊

Plain-English requests → interactive Altair charts. Rule-based parser offline; any OpenAI-compatible endpoint optional (Ollama, Groq, Gemini free tiers — zero cost). Runs on your machine; your data goes nowhere.

## Overview

NeuroViz bridges the gap between natural language queries and data visualizations. Simply describe the chart you want, and the system automatically generates it. Perfect for exploratory data analysis and quick insights.

**Key Features:**
- 🎯 Natural language intent parsing with fuzzy column matching (typo-tolerant, fully offline)
- 🤖 Optional free AI parsing via any OpenAI-compatible endpoint — Ollama (local), Groq/Gemini/OpenRouter free tiers — with automatic offline fallback
- 📈 Support for trends, comparisons, rankings, distributions, and scatter plots
- 🔍 Intelligent column detection and schema inference
- 🎨 Interactive Altair/Vega visualizations
- 📤 Upload page for any tabular data (CSV, Excel, Parquet) with instant preview
- 📊 Auto-generated dashboard: KPI cards + chart grid, Power BI-style
- 🛠️ Manual chart builder: chart type, axes, aggregation, color — Tableau-style
- ⚡ Real-time data filtering
- 💾 Export visualizations to JSON

## Quick Start

### Prerequisites
- Python 3.13+
- Poetry 2.2.1+

### Installation

```bash
# Clone the repository
git clone https://github.com/singh2894/NeuroViz.git
cd NeuroViz

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Optional: free AI parsing

NeuroViz can use any **OpenAI-compatible** chat endpoint for smarter query
parsing. All of these are free:

```bash
# Option A — Ollama (100% free, runs locally, no signup)
#   1. Install from https://ollama.com, then:  ollama pull llama3.2
export LLM_API_URL="http://localhost:11434/v1/chat/completions"
export LLM_MODEL="llama3.2"

# Option B — Groq free tier (hosted, fast)
export LLM_API_URL="https://api.groq.com/openai/v1/chat/completions"
export LLM_MODEL="llama-3.1-8b-instant"
export LLM_API_KEY="gsk_..."
```

On Windows use `setx NAME "value"` instead of `export`. If `LLM_API_URL` is
not set — or the model is unreachable — NeuroViz silently falls back to its
built-in rule-based parser, so the app always works.

### Running the App

```bash
# Launch Streamlit UI
poetry run streamlit run app/main_app.py

# App will be available at http://localhost:8501
```

### Using with Data

1. On the **Data** page, drop any tabular file — CSV, Excel, or Parquet
   (sales, health, climate, anything). Data is held in memory for the
   session only — nothing is written to disk, and a refresh clears it.
2. Open the **Dashboard** for an instant BI overview: KPI cards plus an
   auto-generated chart grid (trend, top categories, distribution, scatter)
3. Use the **Chart Builder** to compose charts manually, Tableau-style —
   pick chart type, X/Y axes, aggregation, and color grouping
4. Or go to **Ask AI** and type natural language queries like:
   - "Show me the trend of sales over time"
   - "Compare revenue by region"
   - "Rank products by profit"
   - "Distribution of customer age"

## Project Structure

```
.
├── app/
│   ├── components/           # Reusable UI components
│   │   └── filters.py       # Data filtering interface
│   ├── compilers/           # Visualization compilation
│   │   └── altair_compile.py # Schema inference + chart generation
│   ├── parsers/             # Intent parsing
│   │   ├── nlp.py          # Free-LLM parser + rule-based fallback
│   │   └── synonyms.py     # Term mappings
│   └── main_app.py          # Streamlit entry point
├── data/                    # Sample datasets (Parquet format)
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
├── .github/
│   └── workflows/          # CI/CD pipelines
└── pyproject.toml          # Poetry configuration
```

## Architecture

### Data Flow
1. **Input**: Natural language query
2. **Parsing**: if a free LLM endpoint is configured (`LLM_API_URL`), it turns
   the query into a validated `Intent` JSON, picking columns from the dataset's
   actual schema. Otherwise keyword rules extract the intent and stdlib
   `difflib` fuzzy-matches query words to column names (typo-tolerant). Either
   way the result is a Pydantic-validated `Intent` — and any LLM failure
   automatically drops to the offline path.
3. **Schema Inference**: Analyze data columns and types
4. **Compilation**: Try chart builders in intent-specific priority order until one fits
5. **Output**: Interactive visualization + JSON export

### Intent Types
- **Trend**: Time-series analysis
- **Compare**: Category comparison
- **Rank**: Top/bottom N analysis
- **Distribution**: Histogram/distribution view

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
poetry install --with dev

# Run tests
poetry run pytest -v

# Format code
poetry run black .

# Lint code
poetry run ruff check . --fix
```

### Code Quality Tools

| Tool | Purpose | Command |
|------|---------|---------|
| **pytest** | Unit testing | `poetry run pytest` |
| **ruff** | Linting | `poetry run ruff check .` |
| **black** | Code formatting | `poetry run black .` |
| **pre-commit** | Git hooks | `pre-commit run --all-files` |

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Dependencies

**Core:**
- `streamlit` - Web UI framework
- `polars` - Fast data processing
- `altair` - Visualization grammar
- `pandas` - Date parsing for Altair
- `pydantic` - Intent schema validation

**Development:**
- `pytest` - Testing framework
- `ruff` - Fast Python linter
- `black` - Code formatter
- `pre-commit` - Git hooks manager

See `pyproject.toml` for complete dependency list.

## Examples

### Query: "Show trend of revenue by month"
```
Intent: trend
Metric: revenue
Dimension: month
Output: Line chart with revenue over time
```

### Query: "Compare sales by region"
```
Intent: compare
Metric: sales
Dimension: region
Output: Bar chart sorted by sales
```

## Troubleshooting

**ModuleNotFoundError: No module named 'app'**
- Ensure you're running from the project root
- Use `poetry run` prefix for all commands

**No data appears in the app**
- Place Parquet files in the `data/` directory
- Verify file format and schema compatibility

**Line too long or linting errors**
- Run `poetry run black .` to auto-format
- Run `poetry run ruff check . --fix` for linting fixes

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Roadmap

- [ ] Support for more complex aggregations
- [ ] Multi-chart dashboards
- [ ] Custom color schemes and themes
- [ ] Export to PNG/SVG formats
- [ ] API endpoint for non-Streamlit integrations
- [ ] Support for real-time data streams

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Singh** - [GitHub](https://github.com/singh2894)

## Support

For issues, questions, or suggestions, please open an [issue](https://github.com/singh2894/NeuroViz/issues) on GitHub.
