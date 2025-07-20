# Stock Advisor Agent

An intelligent agentic stock advisor that tracks stock history and news for short-term price prediction. The system combines technical analysis, news sentiment analysis, and machine learning to provide trading recommendations.

## Features

- **Real-time Stock Data**: Fetch historical and current stock data using Yahoo Finance
- **Advanced Sentiment Analysis**: LLM-powered sentiment analysis using OpenAI API or local Hugging Face models
- **Technical Indicators**: Calculate RSI, MACD, Bollinger Bands, and moving averages
- **Multi-Day Predictions**: Ensemble ML models predict stock movements for the next 1-7 days (default: 3 days)
- **Trading Recommendations**: Generate BUY/SELL/HOLD recommendations with confidence levels
- **Watchlist Management**: Track multiple stocks simultaneously

## Installation

1. Clone or download the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional for enhanced news features):
```bash
cp .env.example .env
# Edit .env file with your API keys
```

## API Keys (Optional)

For enhanced functionality, you can obtain API keys:

- **News API**: Get a free key at [newsapi.org](https://newsapi.org) for news collection
- **Alpha Vantage**: Get a free key at [alphavantage.co](https://www.alphavantage.co) for stock data
- **OpenAI API**: Get a key at [openai.com](https://openai.com) for advanced LLM sentiment analysis

Add these to your `.env` file for enhanced features.

### LLM Sentiment Analysis

The system uses a sophisticated multi-tier sentiment analysis approach:

1. **OpenAI GPT-3.5/4**: If OpenAI API key is provided, uses advanced LLM analysis
2. **Hugging Face Transformers**: Falls back to local financial sentiment models (FinBERT)
3. **Keyword-based**: Final fallback to enhanced keyword analysis

Install optional dependencies for LLM features:
```bash
pip install openai transformers torch
```

## Usage

### Command Line

Analyze a single stock:
```bash
python main.py AAPL
```

Analyze with custom prediction days:
```bash
python main.py --days=5 AAPL  # Predict next 5 days
```

Use demo mode:
```bash
python main.py --demo AAPL
```

Analyze multiple stocks:
```bash
python main.py AAPL GOOGL MSFT TSLA
```

### Enhanced RL Predictor

Run the enhanced RL predictor separately:
```bash
python enhanced_rl_predictor.py AAPL                 # Predict (auto-train if needed)
python enhanced_rl_predictor.py AAPL --train-only    # Only train the ensemble
python enhanced_rl_predictor.py AAPL --episodes=200  # Train with 200 episodes
```

### Interactive Mode

Run without arguments for interactive mode:
```bash
python main.py
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
from src.advisor_agent import StockAdvisorAgent

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

## Architecture

### Components

1. **StockDataFetcher** (`src/stock_data.py`)
   - Fetches historical stock data
   - Calculates technical indicators
   - Provides stock information

2. **NewsCollector** (`src/news_collector.py`)
   - Collects news from RSS feeds and APIs
   - Performs sentiment analysis
   - Tracks market sentiment

3. **StockPredictor** (`src/predictor.py`)
   - Machine learning models for price prediction
   - Feature engineering from stock and news data
   - Ensemble predictions with confidence intervals

4. **StockAdvisorAgent** (`src/advisor_agent.py`)
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

The enhanced RL predictor (`enhanced_rl_predictor.py`) includes significant improvements over the basic RL approach:

- **Deep Q-Networks (DQN)**: Neural networks instead of simple Q-tables
- **Multi-Agent Ensemble**: Multiple agents with voting mechanism
- **Enhanced State Space**: 30 features including advanced technical indicators
- **Prioritized Experience Replay**: More efficient learning from past experiences
- **Risk Management**: Position sizing and drawdown control
- **PyTorch Support**: GPU acceleration when available
- **Sophisticated Reward Function**: Risk-adjusted returns with transaction costs

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

## Risk Disclaimer

This tool is for educational and research purposes only. It should not be considered as financial advice. Always do your own research and consult with qualified financial advisors before making investment decisions.

Key considerations:
- Past performance does not guarantee future results
- Stock markets are inherently volatile and unpredictable
- Machine learning models can have limitations and biases
- News sentiment analysis is simplified and may not capture all nuances

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.