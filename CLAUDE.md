# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Setup and Installation:**
```bash
make install-all      # Install all dependencies (core + dev + ui + ml)
make setup-dev        # Setup development environment with pre-commit hooks
```

**Testing:**
```bash
make test             # Run all tests with pytest
make test-cov         # Run tests with HTML coverage report
pytest stock_advisor/tests/unit/test_stock_data.py -v  # Run specific test file
```

**Code Quality:**
```bash
make lint             # Run flake8 and mypy type checking
make format           # Format code with black and isort
```

**Running the Application:**
```bash
make run-ui           # Start Streamlit web interface
make run-cli          # Run CLI interface
make predict          # Run 3-day prediction script
```

**Build and Package:**
```bash
make build            # Build Python package
make clean            # Clean build artifacts
```

## Architecture Overview

**Stock Advisor** is an AI-powered stock analysis system that combines technical analysis, machine learning, and sentiment analysis for stock predictions.

### Core Components

1. **StockAdvisorAgent** (`stock_advisor/core/advisor_agent.py`) - Main orchestrator that coordinates all analysis components
2. **Multi-tier Data Pipeline** - Graceful fallbacks: Yahoo Finance → IEX Cloud → Alpha Vantage → Demo data
3. **Ensemble ML System** - Combines multiple predictors for robust predictions
4. **Interactive Web UI** - Streamlit-based dashboard with real-time charts

### Key Directories

- `stock_advisor/core/` - Core functionality (data fetching, news collection, main agent)
- `stock_advisor/predictors/` - ML models (technical analysis, RL, hybrid ensemble)
- `stock_advisor/ui/` - Streamlit web interface
- `stock_advisor/utils/` - Utilities including backtesting simulator
- `configs/` - YAML configuration and requirements files
- `scripts/` - Entry point scripts for CLI and web UI

### Prediction Models

The system uses an ensemble approach combining:
- **Technical Predictor** - Traditional indicators (RSI, MACD, Bollinger Bands)
- **RL Predictor** - Reinforcement learning with Q-learning
- **Enhanced RL Predictor** - Deep Q-Network (DQN) implementation
- **Hybrid Predictor** - Weighted ensemble of multiple models

### Configuration

Main config file: `configs/config.yaml`
Required environment variables (via .env):
- `OPENAI_API_KEY` - For LLM-powered sentiment analysis
- `NEWS_API_KEY` - For enhanced news collection
- `ALPHAVANTAGE_API_KEY` - For alternative stock data source

### Testing Strategy

- **Unit tests**: `stock_advisor/tests/unit/` - Test individual components
- **Integration tests**: `stock_advisor/tests/integration/` - Test component interactions
- **Test markers**: `slow`, `integration`, `unit` for selective test running
- Coverage reports generated in `htmlcov/` directory

### Code Quality Standards

- **Type checking**: mypy with strict settings enabled
- **Formatting**: black (line length 88) and isort
- **Linting**: flake8 with custom rules
- **Pre-commit hooks**: Automatically enforce quality standards

### Entry Points

The package provides three console scripts:
- `stock-advisor` - CLI interface
- `stock-advisor-ui` - Web UI launcher
- `stock-advisor-predict` - Prediction script

### Model Persistence

Trained ML models are automatically saved to `stock_advisor/predictors/models/` as pickle files and reloaded on subsequent runs for efficiency.