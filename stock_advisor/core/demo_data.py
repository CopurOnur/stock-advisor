import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict


def generate_demo_stock_data(symbol: str, period: str = "1mo") -> pd.DataFrame:
    """Generate realistic demo stock data"""

    # Period mapping to days
    period_days = {
        "1d": 1,
        "5d": 5,
        "1mo": 30,
        "3mo": 90,
        "6mo": 180,
        "1y": 365,
        "2y": 730,
    }

    days = period_days.get(period, 30)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    # Base prices for different stocks
    base_prices = {
        "AAPL": 150.0,
        "GOOGL": 120.0,
        "MSFT": 300.0,
        "TSLA": 200.0,
        "AMZN": 130.0,
        "NVDA": 400.0,
        "META": 250.0,
    }

    base_price = base_prices.get(symbol, 100.0)

    # Generate realistic stock data
    np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol

    # Generate returns with some trend and volatility
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns

    # Add some trend
    trend = np.linspace(-0.001, 0.001, len(dates))
    returns += trend

    # Calculate prices
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Generate OHLCV data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = close_price * 0.02  # 2% daily volatility

        open_price = close_price * np.random.uniform(0.99, 1.01)
        high_price = max(open_price, close_price) * np.random.uniform(1.0, 1.02)
        low_price = min(open_price, close_price) * np.random.uniform(0.98, 1.0)

        # Volume (more volume on volatile days)
        base_volume = 50000000 if symbol == "AAPL" else 20000000
        volume_multiplier = abs(returns[i]) * 10 + 0.5
        volume = int(base_volume * volume_multiplier)

        data.append(
            {
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
            }
        )

    df = pd.DataFrame(data, index=dates)
    return df


def get_demo_stock_info(symbol: str) -> Dict:
    """Get demo stock information"""

    stock_info = {
        "AAPL": {
            "symbol": "AAPL",
            "company_name": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "market_cap": 3000000000000,
            "current_price": 150.25,
            "previous_close": 149.80,
            "volume": 45000000,
            "avg_volume": 50000000,
        },
        "GOOGL": {
            "symbol": "GOOGL",
            "company_name": "Alphabet Inc.",
            "sector": "Technology",
            "industry": "Internet Services",
            "market_cap": 1500000000000,
            "current_price": 120.50,
            "previous_close": 119.95,
            "volume": 25000000,
            "avg_volume": 30000000,
        },
        "MSFT": {
            "symbol": "MSFT",
            "company_name": "Microsoft Corporation",
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 2500000000000,
            "current_price": 300.75,
            "previous_close": 299.50,
            "volume": 35000000,
            "avg_volume": 40000000,
        },
        "TSLA": {
            "symbol": "TSLA",
            "company_name": "Tesla, Inc.",
            "sector": "Consumer Cyclical",
            "industry": "Auto Manufacturers",
            "market_cap": 800000000000,
            "current_price": 200.30,
            "previous_close": 198.75,
            "volume": 60000000,
            "avg_volume": 65000000,
        },
    }

    return stock_info.get(
        symbol,
        {
            "symbol": symbol,
            "company_name": f"{symbol} Corporation",
            "sector": "Technology",
            "industry": "Software",
            "market_cap": 100000000000,
            "current_price": 100.0,
            "previous_close": 99.5,
            "volume": 20000000,
            "avg_volume": 25000000,
        },
    )


def get_demo_news_sentiment(symbol: str) -> Dict:
    """Get demo news sentiment with LLM-style analysis"""

    sentiments = {
        "AAPL": {
            "score": 0.35,
            "sentiment": "positive",
            "confidence": 0.85,
            "reasoning": "Strong earnings report and product launch positive sentiment",
            "method": "demo_llm",
            "positive_keywords": 8,
            "negative_keywords": 2,
        },
        "GOOGL": {
            "score": 0.25,
            "sentiment": "positive",
            "confidence": 0.72,
            "reasoning": "AI developments and cloud growth driving optimism",
            "method": "demo_llm",
            "positive_keywords": 6,
            "negative_keywords": 3,
        },
        "MSFT": {
            "score": 0.15,
            "sentiment": "positive",
            "confidence": 0.68,
            "reasoning": "Cloud revenue growth and AI integration showing promise",
            "method": "demo_llm",
            "positive_keywords": 5,
            "negative_keywords": 4,
        },
        "TSLA": {
            "score": -0.30,
            "sentiment": "negative",
            "confidence": 0.78,
            "reasoning": "Production concerns and market competition pressures",
            "method": "demo_llm",
            "positive_keywords": 3,
            "negative_keywords": 7,
        },
    }

    return sentiments.get(
        symbol,
        {
            "score": 0.0,
            "sentiment": "neutral",
            "confidence": 0.5,
            "reasoning": "Mixed signals in market sentiment",
            "method": "demo_llm",
            "positive_keywords": 5,
            "negative_keywords": 5,
        },
    )


def get_demo_3day_predictions(symbol: str, current_price: float) -> Dict:
    """Get demo 3-day predictions"""

    # Different prediction patterns for different stocks
    prediction_patterns = {
        "AAPL": [0.015, 0.008, -0.005],  # Up, up, slight down
        "GOOGL": [0.012, 0.020, 0.010],  # Steady upward trend
        "MSFT": [0.005, -0.008, 0.012],  # Mixed with recovery
        "TSLA": [-0.025, -0.015, 0.030],  # Down then sharp recovery
    }

    # Default pattern for unknown symbols
    default_pattern = [0.005, 0.002, -0.003]

    changes = prediction_patterns.get(symbol, default_pattern)

    # Calculate cumulative predictions
    daily_predictions = []
    cumulative_change = 0

    for day, change in enumerate(changes, 1):
        cumulative_change += change
        predicted_price = current_price * (1 + cumulative_change)
        direction = "UP" if change > 0 else "DOWN"
        confidence = max(60 - (day * 5), 40)  # Decrease confidence over time

        daily_predictions.append(
            {
                "day": day,
                "predicted_change": change,
                "cumulative_change": cumulative_change,
                "predicted_price": predicted_price,
                "direction": direction,
                "confidence": confidence,
                "uncertainty": 10 + (day * 5),  # Increase uncertainty over time
            }
        )

    final_price = current_price * (1 + cumulative_change)
    overall_direction = "UP" if cumulative_change > 0 else "DOWN"
    avg_confidence = sum(p["confidence"] for p in daily_predictions) / len(
        daily_predictions
    )

    return {
        "ensemble_prediction": {
            "current_price": current_price,
            "days_predicted": 3,
            "daily_predictions": daily_predictions,
            "overall_summary": {
                "total_change": cumulative_change,
                "final_predicted_price": final_price,
                "overall_direction": overall_direction,
                "average_confidence": avg_confidence,
                "average_uncertainty": 15.0,
            },
            "num_models": 3,
        },
        "method": "demo_ensemble",
    }
