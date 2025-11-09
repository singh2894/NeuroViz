# Data Science Project

A data science project using modern Python tools and best practices.

## Setup

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run tests
pytest

# Run linting
ruff check .
black .
```

## Project Structure

```
.
├── app/
│   ├── components/    # Reusable components
│   ├── parsers/      # Data parsing utilities
│   ├── compilers/    # Data transformation logic
│   └── assets/       # Static assets
├── tests/            # Test suite
└── .github/
    └── workflows/    # CI/CD workflows
```

## Development

This project uses:
- Poetry for dependency management
- pytest for testing
- ruff and black for linting
- GitHub Actions for CI/CD

## License

MIT