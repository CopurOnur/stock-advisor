#!/usr/bin/env python3
"""
Technical Analysis-Only Stock Price Predictor

Pure technical analysis approach using traditional indicators and price patterns
without machine learning. Predicts next 3 days based on:
- Moving averages and crossovers
- RSI momentum
- MACD signals
- Support/resistance levels
- Price patterns
"""

from datetime import datetime
from typing import Dict, List, Tuple
import sys
import os

# Add the parent directory to the path so we can import from stock_advisor
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

try:
    from stock_advisor.predictors.predictor_base import BasePredictor
    from stock_advisor.core.stock_data import StockDataFetcher
except ImportError:
    # Fallback to relative imports if running from within the package
    from .predictor_base import BasePredictor
    from ..core.stock_data import StockDataFetcher


class TechnicalAnalysisPredictor(BasePredictor):
    """Technical analysis-based predictor without machine learning"""

    def __init__(self):
        super().__init__()
        self.stock_fetcher = StockDataFetcher()

    def analyze_moving_averages(self, data) -> Dict:
        """Analyze moving average trends and crossovers"""
        latest = data.iloc[-1]
        signals = {}

        # Price vs moving averages
        if "SMA_5" in data.columns:
            signals["price_vs_sma5"] = (
                "above" if latest["Close"] > latest["SMA_5"] else "below"
            )
            signals["price_vs_sma10"] = (
                "above" if latest["Close"] > latest["SMA_10"] else "below"
            )
            signals["price_vs_sma20"] = (
                "above" if latest["Close"] > latest["SMA_20"] else "below"
            )

            # Moving average alignment (trend strength)
            ma_alignment = (
                "bullish"
                if (latest["SMA_5"] > latest["SMA_10"] > latest["SMA_20"])
                else "bearish"
            )
            signals["ma_alignment"] = ma_alignment

            # Recent crossovers (last 3 days)
            recent_data = data.tail(3)
            signals["recent_crossover"] = self._detect_ma_crossover(recent_data)

        return signals

    def _detect_ma_crossover(self, recent_data) -> str:
        """Detect recent moving average crossovers"""
        if len(recent_data) < 2:
            return "none"

        # Check for 5-day/10-day crossover
        prev = recent_data.iloc[-2]
        curr = recent_data.iloc[-1]

        if prev["SMA_5"] <= prev["SMA_10"] and curr["SMA_5"] > curr["SMA_10"]:
            return "golden_cross"
        elif prev["SMA_5"] >= prev["SMA_10"] and curr["SMA_5"] < curr["SMA_10"]:
            return "death_cross"

        return "none"

    def analyze_rsi(self, data) -> Dict:
        """Analyze RSI momentum signals"""
        if "RSI" not in data.columns:
            return {"signal": "no_data"}

        latest_rsi = data["RSI"].iloc[-1]
        rsi_trend = (
            "rising" if data["RSI"].iloc[-1] > data["RSI"].iloc[-2] else "falling"
        )

        # RSI interpretation
        if latest_rsi > 70:
            signal = "overbought"
        elif latest_rsi < 30:
            signal = "oversold"
        elif 50 <= latest_rsi <= 70:
            signal = "bullish"
        elif 30 <= latest_rsi <= 50:
            signal = "bearish"
        else:
            signal = "neutral"

        return {"value": latest_rsi, "signal": signal, "trend": rsi_trend}

    def analyze_macd(self, data) -> Dict:
        """Analyze MACD signals"""
        if "MACD" not in data.columns or "MACD_Signal" not in data.columns:
            return {"signal": "no_data"}

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        macd_value = latest["MACD"]
        signal_line = latest["MACD_Signal"]
        histogram = macd_value - signal_line

        # MACD signals
        if macd_value > signal_line and prev["MACD"] <= prev["MACD_Signal"]:
            signal = "bullish_crossover"
        elif macd_value < signal_line and prev["MACD"] >= prev["MACD_Signal"]:
            signal = "bearish_crossover"
        elif macd_value > signal_line:
            signal = "bullish"
        else:
            signal = "bearish"

        return {
            "macd": macd_value,
            "signal_line": signal_line,
            "histogram": histogram,
            "signal": signal,
        }

    def analyze_support_resistance(self, data) -> Dict:
        """Identify support and resistance levels"""
        highs = data["High"].rolling(window=5, center=True).max()
        lows = data["Low"].rolling(window=5, center=True).min()

        # Find recent peaks and troughs
        recent_high = data["High"].tail(10).max()
        recent_low = data["Low"].tail(10).min()
        current_price = data["Close"].iloc[-1]

        # Distance to support/resistance
        resistance_distance = (recent_high - current_price) / current_price
        support_distance = (current_price - recent_low) / current_price

        # Determine position
        if resistance_distance < 0.02:  # Within 2% of resistance
            position = "near_resistance"
        elif support_distance < 0.02:  # Within 2% of support
            position = "near_support"
        else:
            position = "mid_range"

        return {
            "resistance_level": recent_high,
            "support_level": recent_low,
            "current_price": current_price,
            "position": position,
            "resistance_distance_pct": resistance_distance * 100,
            "support_distance_pct": support_distance * 100,
        }

    def analyze_volume_trend(self, data) -> Dict:
        """Analyze volume trends"""
        recent_volume = data["Volume"].tail(5).mean()
        longer_volume = data["Volume"].tail(20).mean()

        volume_ratio = recent_volume / longer_volume

        if volume_ratio > 1.5:
            volume_signal = "high"
        elif volume_ratio > 1.2:
            volume_signal = "elevated"
        elif volume_ratio < 0.8:
            volume_signal = "low"
        else:
            volume_signal = "normal"

        return {
            "recent_avg": recent_volume,
            "longer_avg": longer_volume,
            "ratio": volume_ratio,
            "signal": volume_signal,
        }

    def calculate_price_momentum(self, data) -> Dict:
        """Calculate price momentum over different periods"""
        current_price = data["Close"].iloc[-1]

        momentum = {}
        for days in [1, 3, 5, 10]:
            if len(data) > days:
                past_price = data["Close"].iloc[-(days + 1)]
                change_pct = (current_price - past_price) / past_price * 100
                momentum[f"{days}d_change"] = change_pct

        return momentum

    def predict_next_3_days(self, symbol: str) -> Dict:
        """
        Predict next 3 days using pure technical analysis

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with technical analysis prediction
        """
        symbol = symbol.upper()

        # Get stock data with technical indicators
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="3mo")
        if stock_data.empty:
            return {"error": f"No data available for {symbol}"}

        stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)

        # Perform technical analysis
        ma_analysis = self.analyze_moving_averages(stock_data)
        rsi_analysis = self.analyze_rsi(stock_data)
        macd_analysis = self.analyze_macd(stock_data)
        sr_analysis = self.analyze_support_resistance(stock_data)
        volume_analysis = self.analyze_volume_trend(stock_data)
        momentum = self.calculate_price_momentum(stock_data)

        # Generate predictions for each day
        daily_predictions = []
        current_price = stock_data["Close"].iloc[-1]

        for day in range(1, 4):
            prediction = self._predict_single_day(
                day,
                current_price,
                ma_analysis,
                rsi_analysis,
                macd_analysis,
                sr_analysis,
                volume_analysis,
                momentum,
            )
            daily_predictions.append(prediction)
            # Update current price for next day's prediction
            current_price = prediction["predicted_price"]

        # Overall summary
        total_change = (
            daily_predictions[-1]["predicted_price"] - stock_data["Close"].iloc[-1]
        ) / stock_data["Close"].iloc[-1]
        overall_direction = "UP" if total_change > 0 else "DOWN"

        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": stock_data["Close"].iloc[-1],
            "technical_analysis": {
                "moving_averages": ma_analysis,
                "rsi": rsi_analysis,
                "macd": macd_analysis,
                "support_resistance": sr_analysis,
                "volume": volume_analysis,
                "momentum": momentum,
            },
            "daily_predictions": daily_predictions,
            "overall_summary": {
                "direction": overall_direction,
                "total_change_pct": total_change * 100,
                "final_price": daily_predictions[-1]["predicted_price"],
            },
        }

        return result

    def _predict_single_day(
        self,
        day: int,
        current_price: float,
        ma_analysis: Dict,
        rsi_analysis: Dict,
        macd_analysis: Dict,
        sr_analysis: Dict,
        volume_analysis: Dict,
        momentum: Dict,
    ) -> Dict:
        """Predict single day movement based on technical indicators"""

        # Initialize prediction components
        signals = []
        confidence_factors = []
        direction_votes = []

        # Moving average signals
        if ma_analysis.get("ma_alignment") == "bullish":
            direction_votes.append(1)
            signals.append("Bullish MA alignment")
            confidence_factors.append(0.7)
        elif ma_analysis.get("ma_alignment") == "bearish":
            direction_votes.append(-1)
            signals.append("Bearish MA alignment")
            confidence_factors.append(0.7)

        if ma_analysis.get("recent_crossover") == "golden_cross":
            direction_votes.append(1)
            signals.append("Golden cross detected")
            confidence_factors.append(0.8)
        elif ma_analysis.get("recent_crossover") == "death_cross":
            direction_votes.append(-1)
            signals.append("Death cross detected")
            confidence_factors.append(0.8)

        # RSI signals
        if rsi_analysis.get("signal") == "oversold":
            direction_votes.append(1)
            signals.append("RSI oversold - reversal expected")
            confidence_factors.append(0.6)
        elif rsi_analysis.get("signal") == "overbought":
            direction_votes.append(-1)
            signals.append("RSI overbought - pullback expected")
            confidence_factors.append(0.6)
        elif rsi_analysis.get("signal") == "bullish":
            direction_votes.append(1)
            signals.append("RSI in bullish range")
            confidence_factors.append(0.4)
        elif rsi_analysis.get("signal") == "bearish":
            direction_votes.append(-1)
            signals.append("RSI in bearish range")
            confidence_factors.append(0.4)

        # MACD signals
        if macd_analysis.get("signal") == "bullish_crossover":
            direction_votes.append(1)
            signals.append("MACD bullish crossover")
            confidence_factors.append(0.7)
        elif macd_analysis.get("signal") == "bearish_crossover":
            direction_votes.append(-1)
            signals.append("MACD bearish crossover")
            confidence_factors.append(0.7)
        elif macd_analysis.get("signal") == "bullish":
            direction_votes.append(1)
            signals.append("MACD above signal line")
            confidence_factors.append(0.3)
        elif macd_analysis.get("signal") == "bearish":
            direction_votes.append(-1)
            signals.append("MACD below signal line")
            confidence_factors.append(0.3)

        # Support/Resistance
        if sr_analysis.get("position") == "near_support":
            direction_votes.append(1)
            signals.append("Near support level - bounce expected")
            confidence_factors.append(0.6)
        elif sr_analysis.get("position") == "near_resistance":
            direction_votes.append(-1)
            signals.append("Near resistance - pullback expected")
            confidence_factors.append(0.6)

        # Volume confirmation
        if volume_analysis.get("signal") in ["high", "elevated"]:
            # High volume confirms the trend
            if direction_votes and sum(direction_votes) > 0:
                signals.append("High volume confirms upward move")
                confidence_factors.append(0.3)
            elif direction_votes and sum(direction_votes) < 0:
                signals.append("High volume confirms downward move")
                confidence_factors.append(0.3)

        # Calculate final prediction
        if not direction_votes:
            direction = 0
            confidence = 20
            predicted_change = 0
        else:
            direction = sum(direction_votes)
            confidence = min(80, sum(confidence_factors) * 20)  # Scale to percentage

            # Estimate price change magnitude
            base_change = (
                0.005 + (abs(direction) / len(direction_votes)) * 0.015
            )  # 0.5% to 2%
            predicted_change = base_change if direction > 0 else -base_change

            # Reduce magnitude for later days (less certainty)
            predicted_change *= 1 - (day - 1) * 0.2

            # Reduce confidence for later days
            confidence *= 1 - (day - 1) * 0.15

        predicted_price = current_price * (1 + predicted_change)
        final_direction = (
            "UP" if predicted_change > 0 else "DOWN" if predicted_change < 0 else "FLAT"
        )

        return {
            "day": day,
            "predicted_price": predicted_price,
            "predicted_change_pct": predicted_change * 100,
            "direction": final_direction,
            "confidence": max(20, confidence),  # Minimum 20% confidence
            "signals": signals[:3],  # Top 3 signals
            "technical_score": direction,
        }

    def print_prediction_summary(self, result: Dict):
        """Print formatted prediction summary"""
        if "error" in result:
            print(f"ERROR: {result['error']}")
            return

        print(f"\n{'='*70}")
        print(f"TECHNICAL ANALYSIS PREDICTION: {result['symbol']}")
        print(f"{'='*70}")
        print(f"Analysis Time: {result['timestamp']}")
        print(f"Current Price: ${result['current_price']:.2f}")

        # Technical Analysis Summary
        print(f"\nTECHNICAL INDICATORS:")

        ta = result["technical_analysis"]

        # Moving Averages
        ma = ta["moving_averages"]
        if ma:
            print(f"Moving Averages: {ma.get('ma_alignment', 'N/A')} alignment")
            if ma.get("recent_crossover") != "none":
                print(f"Recent Crossover: {ma['recent_crossover']}")

        # RSI
        rsi = ta["rsi"]
        if rsi.get("signal") != "no_data":
            print(f"RSI: {rsi['value']:.1f} ({rsi['signal']}, {rsi['trend']})")

        # MACD
        macd = ta["macd"]
        if macd.get("signal") != "no_data":
            print(f"MACD: {macd['signal']}")

        # Support/Resistance
        sr = ta["support_resistance"]
        print(
            f"Position: {sr['position']} (Support: ${sr['support_level']:.2f}, Resistance: ${sr['resistance_level']:.2f})"
        )

        # Volume
        vol = ta["volume"]
        print(f"Volume: {vol['signal']} ({vol['ratio']:.2f}x normal)")

        # 3-Day Prediction
        print(f"\n3-DAY TECHNICAL FORECAST:")
        overall = result["overall_summary"]
        print(f"Overall Direction: {overall['direction']}")
        print(f"Total Change: {overall['total_change_pct']:+.2f}%")
        print(f"Target Price: ${overall['final_price']:.2f}")

        # Daily Breakdown
        print(f"\nDAILY BREAKDOWN:")
        print(
            f"{'Day':<5} {'Direction':<10} {'Price':<12} {'Change':<10} {'Confidence':<12} {'Key Signals'}"
        )
        print(f"{'-'*80}")

        for pred in result["daily_predictions"]:
            signals_str = ", ".join(pred["signals"][:2]) if pred["signals"] else "None"
            print(
                f"{pred['day']:<5} "
                f"{pred['direction']:<10} "
                f"${pred['predicted_price']:<11.2f} "
                f"{pred['predicted_change_pct']:+8.2f}% "
                f"{pred['confidence']:<11.1f}% "
                f"{signals_str}"
            )


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python technical_predictor.py <SYMBOL>")
        print("Example: python technical_predictor.py AAPL")
        return

    symbol = sys.argv[1].upper()

    print("Technical Analysis Stock Predictor")
    print("==================================")
    print(f"Symbol: {symbol}")
    print("Method: Pure technical analysis (no machine learning)")

    predictor = TechnicalAnalysisPredictor()
    result = predictor.predict_next_3_days(symbol)
    predictor.print_prediction_summary(result)

    print(f"\n{'='*70}")
    print("DISCLAIMER: Technical analysis prediction for educational purposes.")
    print("Not financial advice. Markets are unpredictable.")
    print(f"{'='*70}")


# Alias for compatibility
TechnicalPredictor = TechnicalAnalysisPredictor

if __name__ == "__main__":
    main()
