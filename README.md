# NeuroViz 📊

An intelligent visualization assistant that uses Natural Language Processing to help create and customize data visualizations through natural language commands.

## Overview

NeuroViz bridges the gap between natural language queries and data visualizations. Simply describe the chart you want, and the system automatically generates it. Perfect for exploratory data analysis and quick insights.

**Key Features:**
- 🎯 Natural language to visualization compilation
- 📈 Support for trends, comparisons, rankings, and distributions
- 🔍 Intelligent column detection and schema inference
- 🎨 Interactive Altair/Vega visualizations
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

### Running the App

```bash
# Launch Streamlit UI
poetry run streamlit run app/main_app.py

# App will be available at http://localhost:8501
```

### Using with Data

1. Place your Parquet files in the `data/` directory
2. Select a dataset from the sidebar
3. Enter natural language queries like:
   - "Show me the trend of sales over time"
   - "Compare revenue by region"
   - "Rank products by profit"
   - "Distribution of customer age"

## Project Structure

```
.
├── app/
│   ├── components/           # Reusable UI components
│   │   ├── filters.py       # Data filtering interface
│   │   └── quick_model.py   # Quick model generation
│   ├── compilers/           # Visualization compilation
│   │   ├── altair_compile.py # Chart generation logic
│   │   └── schema.py        # Schema inference
│   ├── parsers/             # NLP processing
│   │   ├── nlp.py          # Intent parsing
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
2. **Parsing**: Extract intent (trend, compare, rank, distribute)
3. **Schema Inference**: Analyze data columns and types
4. **Compilation**: Generate appropriate Altair chart
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
- `pandas` - Data manipulation
- `spacy` - NLP processing

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
