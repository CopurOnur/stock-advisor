#!/usr/bin/env python3
"""
3-Day Stock Price Prediction using Technical Analysis

This script uses the stock advisor's prediction models to forecast the next 3 days
of stock price movements based on technical analysis and machine learning.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from stock_advisor.core.stock_data import StockDataFetcher
from stock_advisor.core.news_collector import NewsCollector
from stock_advisor.core.predictor import StockPredictor


class ThreeDayPredictor:
    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.news_collector = NewsCollector()
        self.predictor = StockPredictor()

    def predict_stock_3_days(self, symbol: str, use_news: bool = True) -> dict:
        """
        Predict next 3 days price movements for a stock

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            use_news: Whether to include news sentiment analysis

        Returns:
            Dictionary with prediction results
        """
        symbol = symbol.upper()
        print(f"Fetching data for {symbol}...")

        # Get stock data with longer history for better predictions
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="3mo")

        if stock_data.empty:
            return {"error": f"No data available for {symbol}"}

        # Calculate technical indicators
        stock_data_with_indicators = self.stock_fetcher.calculate_technical_indicators(
            stock_data
        )

        # Get news sentiment if requested
        news_sentiment = None
        if use_news:
            print("Analyzing news sentiment...")
            news_data = self.news_collector.get_comprehensive_news(
                [symbol], days_back=7
            )
            news_sentiment = news_data["sentiment_summary"].get(symbol)

        # Train models on historical data
        print("Training prediction models...")
        training_results = self.predictor.train_models(
            stock_data_with_indicators, news_sentiment
        )

        if any("error" in result for result in training_results.values()):
            print("Warning: Some models failed to train")
            for model_name, result in training_results.items():
                if "error" in result:
                    print(f"  {model_name}: {result['error']}")

        # Make 3-day ensemble prediction
        print("Generating 3-day predictions...")
        prediction_result = self.predictor.get_ensemble_prediction(
            stock_data_with_indicators, news_sentiment, days=3
        )

        if "error" in prediction_result:
            return prediction_result

        # Add current stock information
        current_price = stock_data["Close"].iloc[-1]
        price_change = (
            (current_price - stock_data["Close"].iloc[-2])
            / stock_data["Close"].iloc[-2]
            * 100
        )

        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_info": {
                "current_price": current_price,
                "daily_change_pct": price_change,
                "volume": stock_data["Volume"].iloc[-1],
                "data_points_used": len(stock_data_with_indicators),
            },
            "news_sentiment_included": use_news,
            "prediction": prediction_result,
            "training_performance": training_results,
        }

        return result

    def print_prediction_summary(self, result: dict):
        """Print a formatted summary of the prediction"""
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return

        print(f"\n{'='*60}")
        print(f"3-DAY PRICE PREDICTION: {result['symbol']}")
        print(f"{'='*60}")
        print(f"Analysis Time: {result['timestamp']}")

        # Current info
        current = result["current_info"]
        print(f"\nCURRENT STATUS:")
        print(f"Price: ${current['current_price']:.2f}")
        print(f"Daily Change: {current['daily_change_pct']:+.2f}%")
        print(f"Volume: {current['volume']:,}")
        print(f"Data Points: {current['data_points_used']}")
        print(
            f"News Sentiment: {'Included' if result['news_sentiment_included'] else 'Not included'}"
        )

        # Ensemble prediction
        ensemble = result["prediction"]["ensemble_prediction"]
        print(f"\nENSEMBLE PREDICTION (3 Days):")
        print(f"Models Used: {ensemble['num_models']}")

        # Overall summary
        overall = ensemble["overall_summary"]
        print(f"\nOVERALL FORECAST:")
        print(f"Direction: {overall['overall_direction']}")
        print(f"Final Price: ${overall['final_predicted_price']:.2f}")
        print(f"Total Change: {overall['total_change']*100:+.2f}%")
        print(f"Confidence: {overall['average_confidence']:.1f}%")
        print(f"Uncertainty: ±{overall['average_uncertainty']:.1f}%")

        # Daily breakdown
        print(f"\nDAILY BREAKDOWN:")
        print(
            f"{'Day':<5} {'Direction':<10} {'Price':<10} {'Change':<10} {'Confidence':<12} {'Uncertainty':<12}"
        )
        print(f"{'-'*60}")

        for day_pred in ensemble["daily_predictions"]:
            print(
                f"{day_pred['day']:<5} "
                f"{day_pred['direction']:<10} "
                f"${day_pred['predicted_price']:<9.2f} "
                f"{day_pred['predicted_change']*100:+8.2f}% "
                f"{day_pred['confidence']:<11.1f}% "
                f"±{day_pred['uncertainty']:<10.1f}%"
            )

        # Model performance
        print(f"\nMODEL TRAINING PERFORMANCE:")
        for model_name, perf in result["training_performance"].items():
            if "error" not in perf:
                print(
                    f"{model_name.replace('_', ' ').title()}: "
                    f"R² = {perf['r2']:.3f}, RMSE = {perf['rmse']:.4f}"
                )
            else:
                print(
                    f"{model_name.replace('_', ' ').title()}: Failed - {perf['error']}"
                )


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python predict_3_days.py <SYMBOL> [--no-news]")
        print("Example: python predict_3_days.py AAPL")
        print("Example: python predict_3_days.py AAPL --no-news")
        return

    symbol = sys.argv[1].upper()
    use_news = "--no-news" not in sys.argv

    print("3-Day Stock Price Predictor")
    print("===========================")
    print(f"Symbol: {symbol}")
    print(f"News Analysis: {'Enabled' if use_news else 'Disabled'}")

    predictor = ThreeDayPredictor()
    result = predictor.predict_stock_3_days(symbol, use_news)
    predictor.print_prediction_summary(result)

    print(f"\n{'='*60}")
    print("DISCLAIMER: This is for educational purposes only.")
    print("Not financial advice. Past performance ≠ future results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
