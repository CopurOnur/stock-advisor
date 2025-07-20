# Stock Advisor Agent

An intelligent agentic stock advisor that tracks stock history and news for short-term price prediction. The system combines technical analysis, news sentiment analysis, and machine learning to provide trading recommendations.

## Features

- **Real-time Stock Data**: Fetch historical and current stock data using Yahoo Finance
- **Advanced Sentiment Analysis**: LLM-powered sentiment analysis using OpenAI API or local Hugging Face models
- **Technical Indicators**: Calculate RSI, MACD, Bollinger Bands, and moving averages
- **Multi-Day Predictions**: Ensemble ML models predict stock movements for the next 1-7 days (default: 3 days)
- **Trading Recommendations**: Generate BUY/SELL/HOLD recommendations with confidence levels
- **Watchlist Management**: Track multiple stocks simultaneously
- **Interactive Web UI**: Modern web interface with real-time charts and predictions

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-advisor.git
cd stock-advisor

# Install core dependencies
pip install -e .

# Install all dependencies (including UI and ML components)
pip install -e .[dev,ui,ml]
```

### Running the Application

```bash
# Start the web interface
make run-ui
# or
python scripts/run_ui.py

# Use the command line interface
make run-cli
# or
python scripts/main.py AAPL

# Run predictions
make predict
# or
python scripts/predict_3_days.py AAPL
```

## Project Structure

```
stock-advisor/
├── stock_advisor/              # Main package
│   ├── core/                   # Core functionality
│   │   ├── stock_data.py      # Data fetching and processing
│   │   ├── news_collector.py  # News collection and sentiment
│   │   ├── advisor_agent.py   # Main orchestrator
│   │   └── demo_data.py       # Demo data generation
│   ├── predictors/            # Prediction algorithms
│   │   ├── predictor_base.py  # Base predictor class
│   │   ├── technical_predictor.py  # Technical analysis
│   │   ├── rl_predictor.py    # Reinforcement learning
│   │   ├── enhanced_rl_predictor.py  # Enhanced RL with DQN
│   │   └── hybrid_predictor.py  # Hybrid ensemble
│   ├── ui/                    # User interface
│   │   └── web_ui.py         # Streamlit web interface
│   ├── utils/                 # Utility functions
│   │   └── backtest_simulator.py  # Backtesting tools
│   └── tests/                 # Test suite
│       ├── unit/             # Unit tests
│       └── integration/      # Integration tests
├── scripts/                   # Command-line scripts
│   ├── main.py               # CLI interface
│   ├── run_ui.py             # Web UI launcher
│   └── predict_3_days.py     # Prediction script
├── configs/                   # Configuration files
│   ├── config.yaml           # Main configuration
│   ├── requirements.txt      # Dependencies
│   └── requirements_ui.txt   # UI dependencies
├── docs/                     # Documentation
├── examples/                 # Example scripts
├── models/                   # Trained models (auto-generated)
├── logs/                     # Log files (auto-generated)
├── setup.py                  # Package setup
├── pyproject.toml           # Modern Python packaging
├── Makefile                 # Development commands
└── README.md                # This file
```

## API Keys (Optional)

For enhanced functionality, you can obtain API keys:

- **News API**: Get a free key at [newsapi.org](https://newsapi.org) for news collection
- **Alpha Vantage**: Get a free key at [alphavantage.co](https://www.alphavantage.co) for stock data
- **OpenAI API**: Get a key at [openai.com](https://openai.com) for advanced LLM sentiment analysis

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
# Edit .env file with your API keys
```

## Usage

### Web Interface

Start the web interface for interactive analysis:

```bash
make run-ui
```

Open your browser to `http://localhost:8501` to access the dashboard.

### Command Line Interface

Analyze a single stock:
```bash
python scripts/main.py AAPL
```

Analyze with custom prediction days:
```bash
python scripts/main.py --days=5 AAPL
```

Use demo mode:
```bash
python scripts/main.py --demo AAPL
```

Analyze multiple stocks:
```bash
python scripts/main.py AAPL GOOGL MSFT TSLA
```

### Interactive Mode

Run without arguments for interactive mode:
```bash
python scripts/main.py
```

Available commands:
- `analyze AAPL` - Analyze Apple stock
- `add AAPL` - Add Apple to watchlist
- `remove AAPL` - Remove Apple from watchlist
- `watchlist` - Analyze entire watchlist
- `show` - Display current watchlist
- `quit` - Exit

### Programmatic Usage

```python
from stock_advisor import StockAdvisorAgent

# Initialize advisor
advisor = StockAdvisorAgent()

# Add stocks to watchlist
advisor.add_to_watchlist(['AAPL', 'GOOGL', 'MSFT'])

# Analyze a single stock
analysis = advisor.analyze_stock('AAPL')
print(analysis['recommendation'])

# Analyze entire watchlist
results = advisor.analyze_watchlist()
print(results['summary'])
```

## Prediction Methods

### Technical Analysis
Pure technical analysis using traditional indicators:
```bash
python -m stock_advisor.predictors.technical_predictor AAPL
```

### Reinforcement Learning
Q-learning based trading agent:
```bash
python -m stock_advisor.predictors.rl_predictor AAPL
```

### Enhanced Reinforcement Learning
Deep Q-Network with ensemble agents:
```bash
python -m stock_advisor.predictors.enhanced_rl_predictor AAPL
```

### Hybrid Approach
Combines all methods with ensemble voting:
```bash
python -m stock_advisor.predictors.hybrid_predictor AAPL
```

## Architecture

### Core Components

1. **StockDataFetcher** (`stock_advisor/core/stock_data.py`)
   - Fetches historical stock data
   - Calculates technical indicators
   - Provides stock information

2. **NewsCollector** (`stock_advisor/core/news_collector.py`)
   - Collects news from RSS feeds and APIs
   - Performs sentiment analysis
   - Tracks market sentiment

3. **Predictors** (`stock_advisor/predictors/`)
   - Machine learning models for price prediction
   - Feature engineering from stock and news data
   - Ensemble predictions with confidence intervals

4. **StockAdvisorAgent** (`stock_advisor/core/advisor_agent.py`)
   - Main orchestrator class
   - Combines all components for comprehensive analysis
   - Generates trading recommendations

### Technical Indicators

- **RSI (Relative Strength Index)**: Momentum oscillator (0-100)
- **MACD**: Moving Average Convergence Divergence
- **Bollinger Bands**: Volatility bands around moving average
- **Moving Averages**: SMA (5, 10, 20 days) and EMA (12, 26 days)

### Machine Learning Models

- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Sequential weak learners
- **Linear Regression**: Linear relationship modeling

### Enhanced Reinforcement Learning

The enhanced RL predictor includes significant improvements:

- **Deep Q-Networks (DQN)**: Neural networks instead of simple Q-tables
- **Multi-Agent Ensemble**: Multiple agents with voting mechanism
- **Enhanced State Space**: 30 features including advanced technical indicators
- **Prioritized Experience Replay**: More efficient learning from past experiences
- **Risk Management**: Position sizing and drawdown control
- **PyTorch Support**: GPU acceleration when available
- **Sophisticated Reward Function**: Risk-adjusted returns with transaction costs

## Development

### Setup Development Environment

```bash
# Install development dependencies
make install-dev

# Set up pre-commit hooks
make setup-dev

# Run tests
make test

# Run linting
make lint

# Format code
make format
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test
pytest stock_advisor/tests/unit/test_stock_data.py -v
```

### Building and Distribution

```bash
# Build package
make build

# Clean build artifacts
make clean
```

## Configuration

The application uses a YAML configuration file (`configs/config.yaml`) for settings:

- Data sources and API configurations
- ML model parameters
- Risk management settings
- UI preferences
- Logging configuration

Environment variables can be set in `.env` file for sensitive information like API keys.

## Output Format

### Stock Analysis

```json
{
  "symbol": "AAPL",
  "timestamp": "2024-01-15T10:30:00",
  "current_metrics": {
    "current_price": 185.50,
    "price_change_pct": 1.25,
    "volume": 45000000,
    "rsi": 65.2
  },
  "technical_analysis": {
    "rsi_signal": "NEUTRAL",
    "macd_signal": "BULLISH",
    "trend": "UPTREND"
  },
  "news_sentiment": {
    "sentiment": "positive",
    "score": 0.024,
    "positive_keywords": 12,
    "negative_keywords": 3
  },
  "prediction": {
    "ensemble_prediction": {
      "direction": "UP",
      "predicted_price": 187.25,
      "confidence": 67.3,
      "uncertainty": 12.1
    }
  },
  "recommendation": {
    "action": "BUY",
    "confidence": 72.5,
    "risk_level": "MEDIUM",
    "reasoning": [
      "MACD shows bullish signal",
      "Price above 20-day moving average",
      "Positive news sentiment"
    ]
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `make test`
5. Format code: `make format`
6. Submit a pull request

## Risk Disclaimer

This tool is for educational and research purposes only. It should not be considered as financial advice. Always do your own research and consult with qualified financial advisors before making investment decisions.

Key considerations:
- Past performance does not guarantee future results
- Stock markets are inherently volatile and unpredictable
- Machine learning models can have limitations and biases
- News sentiment analysis is simplified and may not capture all nuances

## License

This project is open source and available under the MIT License.

## Support

For support, please:
- Check the documentation in the `docs/` directory
- Review existing issues on GitHub
- Create a new issue for bugs or feature requests
- Join our community discussions

## Changelog

### Version 1.0.0
- Initial release with technical analysis, RL, and hybrid predictions
- Web UI with interactive charts and dashboards
- Enhanced RL predictor with Deep Q-Networks
- Comprehensive test suite and documentation
- Modern Python packaging and development tools