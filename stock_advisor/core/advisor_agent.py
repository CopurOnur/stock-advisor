from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
import pandas as pd
import sys
import os

# Add the parent directory to the path so we can import from stock_advisor
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from stock_advisor.core.stock_data import StockDataFetcher
    from stock_advisor.core.news_collector import NewsCollector
    from stock_advisor.core.predictor import StockPredictor
    from stock_advisor.core.demo_data import (
        generate_demo_stock_data,
        get_demo_stock_info,
        get_demo_news_sentiment,
        get_demo_3day_predictions,
    )
except ImportError:
    # Fallback to relative imports if running from within the package
    from .stock_data import StockDataFetcher
    from .news_collector import NewsCollector
    from .predictor import StockPredictor
    from .demo_data import (
        generate_demo_stock_data,
        get_demo_stock_info,
        get_demo_news_sentiment,
        get_demo_3day_predictions,
    )

from stock_advisor.constants import TimeConstants, TechnicalIndicators, RiskManagement
from stock_advisor.exceptions import DataFetchError, InvalidSymbolError, PredictionError
from stock_advisor.types import (
    Symbol,
    DataFetcherProtocol,
    NewsCollectorProtocol,
    PredictorProtocol,
)


class StockAdvisorAgent:
    def __init__(
        self,
        demo_mode: bool = False,
        stock_fetcher: Optional[DataFetcherProtocol] = None,
        news_collector: Optional[NewsCollectorProtocol] = None,
        predictor: Optional[PredictorProtocol] = None,
    ) -> None:
        """
        Initialize the Stock Advisor Agent with dependency injection support.

        Args:
            demo_mode: Whether to use demo data instead of real data
            stock_fetcher: Stock data fetcher instance (optional, creates default if None)
            news_collector: News collector instance (optional, creates default if None)
            predictor: Predictor instance (optional, creates default if None)
        """
        self.demo_mode = demo_mode

        # Use dependency injection with fallback to default implementations
        self.stock_fetcher = stock_fetcher or StockDataFetcher()
        self.news_collector = news_collector or NewsCollector()
        self.predictor = predictor or StockPredictor()

        self.watchlist: List[Symbol] = []
        self.analysis_history: List[Dict[str, Any]] = []

    def add_to_watchlist(self, symbols: List[str]) -> None:
        """Add stocks to watchlist"""
        for symbol in symbols:
            if symbol.upper() not in self.watchlist:
                self.watchlist.append(symbol.upper())

    def remove_from_watchlist(self, symbols: List[str]) -> None:
        """Remove stocks from watchlist"""
        for symbol in symbols:
            if symbol.upper() in self.watchlist:
                self.watchlist.remove(symbol.upper())

    def analyze_stock(self, symbol: str, period: str = "3mo") -> Dict[str, Any]:
        """
        Comprehensive analysis of a single stock.

        Args:
            symbol: Stock symbol
            period: Historical data period

        Returns:
            Complete analysis dictionary
        """
        symbol = symbol.upper()
        analysis = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "period_analyzed": period,
        }

        try:
            # Fetch stock data and info
            stock_data, stock_info = self._fetch_stock_data(symbol, period)

            if stock_data.empty:
                analysis["error"] = f"No stock data found for {symbol}"
                return analysis

            # Add technical indicators
            stock_data_with_indicators = (
                self.stock_fetcher.calculate_technical_indicators(stock_data)
            )

            # Collect news and sentiment
            stock_sentiment = self._collect_news_sentiment(symbol)

            # Generate predictions
            prediction_result, training_results = self._generate_predictions(
                symbol, stock_data_with_indicators, stock_sentiment
            )

            # Calculate current metrics
            current_metrics = self._calculate_current_metrics(
                stock_data_with_indicators
            )

            # Build complete analysis
            analysis.update(
                {
                    "stock_info": stock_info,
                    "current_metrics": current_metrics,
                    "technical_analysis": self._analyze_technical_indicators(
                        stock_data_with_indicators
                    ),
                    "news_sentiment": stock_sentiment,
                    "prediction": prediction_result,
                    "model_performance": training_results,
                }
            )

            # Generate recommendation
            analysis["recommendation"] = self._generate_recommendation(analysis)

        except Exception as e:
            analysis["error"] = str(e)

        # Store in history
        self.analysis_history.append(analysis)

        return analysis

    def _fetch_stock_data(
        self, symbol: str, period: str
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch stock data and info with fallback to demo data.

        Args:
            symbol: Stock symbol
            period: Historical data period

        Returns:
            Tuple of (stock_data, stock_info)
        """
        print(f"Fetching stock data for {symbol}...")

        if self.demo_mode:
            print("(Using demo data)")
            stock_data = generate_demo_stock_data(symbol, period)
            stock_info = get_demo_stock_info(symbol)
        else:
            try:
                stock_data = self.stock_fetcher.get_stock_data(symbol, period)
                stock_info = self.stock_fetcher.get_stock_info(symbol)

                # Fallback to demo data if real data fails
                if stock_data.empty:
                    print("Real data unavailable, switching to demo data...")
                    stock_data = generate_demo_stock_data(symbol, period)
                    stock_info = get_demo_stock_info(symbol)
            except Exception as e:
                print(f"Data fetch error, using demo data: {e}")
                stock_data = generate_demo_stock_data(symbol, period)
                stock_info = get_demo_stock_info(symbol)

        return stock_data, stock_info

    def _collect_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Collect news and sentiment analysis with fallback to demo data.

        Args:
            symbol: Stock symbol

        Returns:
            News sentiment dictionary
        """
        print(f"Collecting news for {symbol}...")

        if self.demo_mode:
            print("(Using demo sentiment)")
            return get_demo_news_sentiment(symbol)

        try:
            news_data = self.news_collector.get_comprehensive_news([symbol])
            stock_sentiment = news_data["sentiment_summary"].get(symbol, {})

            # Fallback to demo sentiment if news collection fails
            if not stock_sentiment:
                print("News collection failed, using demo sentiment...")
                stock_sentiment = get_demo_news_sentiment(symbol)

            return stock_sentiment

        except Exception as e:
            print(f"News collection error, using demo sentiment: {e}")
            return get_demo_news_sentiment(symbol)

    def _generate_predictions(
        self, symbol: str, stock_data: pd.DataFrame, sentiment: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Generate model predictions with fallback to demo predictions.

        Args:
            symbol: Stock symbol
            stock_data: Stock data with technical indicators
            sentiment: News sentiment data

        Returns:
            Tuple of (prediction_result, training_results)
        """
        print(f"Training prediction models for {symbol}...")

        if self.demo_mode:
            print("(Using demo predictions)")
            current_price = stock_data["Close"].iloc[-1]
            prediction_result = get_demo_3day_predictions(symbol, current_price)
            training_results = {"demo_mode": True}
        else:
            try:
                training_results = self.predictor.train_models(stock_data, sentiment)
                prediction_result = self.predictor.get_ensemble_prediction(
                    stock_data, sentiment, days=3
                )
            except Exception as e:
                print(f"Prediction error, using demo predictions: {e}")
                current_price = stock_data["Close"].iloc[-1]
                prediction_result = get_demo_3day_predictions(symbol, current_price)
                training_results = {"error": str(e)}

        return prediction_result, training_results

    def _calculate_current_metrics(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate current market metrics from stock data.

        Args:
            stock_data: Stock data with technical indicators

        Returns:
            Dictionary of current metrics
        """
        current_data = stock_data.iloc[-1]
        previous_data = stock_data.iloc[-2] if len(stock_data) > 1 else current_data

        return {
            "current_price": float(current_data["Close"]),
            "previous_close": float(previous_data["Close"]),
            "price_change": float(current_data["Close"] - previous_data["Close"]),
            "price_change_pct": float(
                (current_data["Close"] - previous_data["Close"])
                / previous_data["Close"]
                * 100
            ),
            "volume": int(current_data["Volume"]),
            "rsi": (
                float(current_data.get("RSI", 0))
                if not pd.isna(current_data.get("RSI", float("nan")))
                else None
            ),
            "macd": (
                float(current_data.get("MACD", 0))
                if not pd.isna(current_data.get("MACD", float("nan")))
                else None
            ),
        }

    def analyze_watchlist(self) -> Dict[str, Any]:
        """
        Analyze all stocks in watchlist

        Returns:
            Dictionary with analysis for all watchlist stocks
        """
        if not self.watchlist:
            return {"error": "Watchlist is empty"}

        results = {
            "timestamp": datetime.now().isoformat(),
            "watchlist_size": len(self.watchlist),
            "analyses": {},
            "summary": {},
        }

        print(f"Analyzing {len(self.watchlist)} stocks in watchlist...")

        for symbol in self.watchlist:
            print(f"Analyzing {symbol}...")
            analysis = self.analyze_stock(symbol)
            results["analyses"][symbol] = analysis

        # Generate summary
        results["summary"] = self._generate_watchlist_summary(results["analyses"])

        return results

    def _analyze_technical_indicators(self, data: pd.DataFrame) -> Dict[str, str]:
        """Analyze technical indicators"""
        current = data.iloc[-1]

        analysis = {}

        # RSI Analysis
        if "RSI" in data.columns and not pd.isna(current["RSI"]):
            rsi = current["RSI"]
            if rsi > 70:
                analysis["rsi_signal"] = "OVERBOUGHT"
            elif rsi < 30:
                analysis["rsi_signal"] = "OVERSOLD"
            else:
                analysis["rsi_signal"] = "NEUTRAL"

        # MACD Analysis
        if "MACD" in data.columns and "MACD_Signal" in data.columns:
            if not pd.isna(current["MACD"]) and not pd.isna(current["MACD_Signal"]):
                if current["MACD"] > current["MACD_Signal"]:
                    analysis["macd_signal"] = "BULLISH"
                else:
                    analysis["macd_signal"] = "BEARISH"

        # Moving Average Analysis
        if "SMA_20" in data.columns and not pd.isna(current["SMA_20"]):
            if current["Close"] > current["SMA_20"]:
                analysis["trend"] = "UPTREND"
            else:
                analysis["trend"] = "DOWNTREND"

        # Bollinger Bands
        if all(col in data.columns for col in ["BB_Upper", "BB_Lower", "BB_Middle"]):
            if not any(
                pd.isna(current[col]) for col in ["BB_Upper", "BB_Lower", "BB_Middle"]
            ):
                if current["Close"] > current["BB_Upper"]:
                    analysis["bb_signal"] = "OVERBOUGHT"
                elif current["Close"] < current["BB_Lower"]:
                    analysis["bb_signal"] = "OVERSOLD"
                else:
                    analysis["bb_signal"] = "NEUTRAL"

        return analysis

    def _generate_recommendation(self, analysis: Dict) -> Dict:
        """Generate trading recommendation based on analysis"""
        recommendation = {
            "action": "HOLD",
            "confidence": 0,
            "reasoning": [],
            "risk_level": "MEDIUM",
        }

        score = 0
        max_score = 0

        # Technical indicators scoring
        technical = analysis.get("technical_analysis", {})

        if technical.get("rsi_signal") == "OVERSOLD":
            score += 1
            recommendation["reasoning"].append("RSI indicates oversold condition")
        elif technical.get("rsi_signal") == "OVERBOUGHT":
            score -= 1
            recommendation["reasoning"].append("RSI indicates overbought condition")
        max_score += 1

        if technical.get("macd_signal") == "BULLISH":
            score += 1
            recommendation["reasoning"].append("MACD shows bullish signal")
        elif technical.get("macd_signal") == "BEARISH":
            score -= 1
            recommendation["reasoning"].append("MACD shows bearish signal")
        max_score += 1

        if technical.get("trend") == "UPTREND":
            score += 1
            recommendation["reasoning"].append("Price above 20-day moving average")
        elif technical.get("trend") == "DOWNTREND":
            score -= 1
            recommendation["reasoning"].append("Price below 20-day moving average")
        max_score += 1

        # News sentiment scoring
        sentiment = analysis.get("news_sentiment", {})
        if sentiment.get("sentiment") == "positive":
            score += 1
            recommendation["reasoning"].append("Positive news sentiment")
        elif sentiment.get("sentiment") == "negative":
            score -= 1
            recommendation["reasoning"].append("Negative news sentiment")
        max_score += 1

        # Prediction scoring
        prediction = analysis.get("prediction", {})
        if "ensemble_prediction" in prediction:
            ensemble = prediction["ensemble_prediction"]
            overall_summary = ensemble.get("overall_summary", {})
            if (
                overall_summary.get("overall_direction") == "UP"
                and overall_summary.get("average_confidence", 0) > 60
            ):
                score += 2
                recommendation["reasoning"].append(
                    f"ML models predict {overall_summary['average_confidence']:.1f}% chance of price increase"
                )
            elif (
                overall_summary.get("overall_direction") == "DOWN"
                and overall_summary.get("average_confidence", 0) > 60
            ):
                score -= 2
                recommendation["reasoning"].append(
                    f"ML models predict {overall_summary['average_confidence']:.1f}% chance of price decrease"
                )
            max_score += 2

        # Determine action and confidence
        if max_score > 0:
            confidence = abs(score) / max_score * 100
            recommendation["confidence"] = round(confidence, 1)

            if score >= 2:
                recommendation["action"] = "BUY"
                recommendation["risk_level"] = "LOW" if confidence > 70 else "MEDIUM"
            elif score <= -2:
                recommendation["action"] = "SELL"
                recommendation["risk_level"] = "HIGH" if confidence > 70 else "MEDIUM"
            else:
                recommendation["action"] = "HOLD"

        return recommendation

    def _generate_watchlist_summary(self, analyses: Dict) -> Dict:
        """Generate summary of watchlist analysis"""
        summary = {
            "buy_candidates": [],
            "sell_candidates": [],
            "hold_recommendations": [],
            "market_sentiment": "NEUTRAL",
        }

        sentiment_scores = []

        for symbol, analysis in analyses.items():
            if "error" in analysis:
                continue

            recommendation = analysis.get("recommendation", {})
            action = recommendation.get("action", "HOLD")
            confidence = recommendation.get("confidence", 0)

            stock_summary = {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "current_price": analysis.get("current_metrics", {}).get(
                    "current_price", 0
                ),
                "price_change_pct": analysis.get("current_metrics", {}).get(
                    "price_change_pct", 0
                ),
            }

            if action == "BUY":
                summary["buy_candidates"].append(stock_summary)
            elif action == "SELL":
                summary["sell_candidates"].append(stock_summary)
            else:
                summary["hold_recommendations"].append(stock_summary)

            # Collect sentiment scores
            sentiment = analysis.get("news_sentiment", {})
            if "score" in sentiment:
                sentiment_scores.append(sentiment["score"])

        # Determine overall market sentiment
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            if avg_sentiment > 0.01:
                summary["market_sentiment"] = "POSITIVE"
            elif avg_sentiment < -0.01:
                summary["market_sentiment"] = "NEGATIVE"

        # Sort by confidence
        summary["buy_candidates"].sort(key=lambda x: x["confidence"], reverse=True)
        summary["sell_candidates"].sort(key=lambda x: x["confidence"], reverse=True)

        return summary

    def get_analysis_history(
        self, symbol: Optional[str] = None, days: int = 7
    ) -> List[Dict]:
        """Get analysis history"""
        cutoff_date = datetime.now() - timedelta(days=days)

        filtered_history = []
        for analysis in self.analysis_history:
            analysis_date = datetime.fromisoformat(analysis["timestamp"])
            if analysis_date >= cutoff_date:
                if symbol is None or analysis.get("symbol") == symbol.upper():
                    filtered_history.append(analysis)

        return filtered_history

    def save_analysis(self, analysis: Dict, filename: Optional[str] = None):
        """Save analysis to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            symbol = analysis.get("symbol", "unknown")
            filename = f"analysis_{symbol}_{timestamp}.json"

        filepath = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(analysis, f, indent=2, default=str)

        return filepath
