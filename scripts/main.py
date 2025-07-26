#!/usr/bin/env python3
"""
Stock Advisor Agent - Main execution script
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from stock_advisor.core.advisor_agent import StockAdvisorAgent


def print_analysis_summary(analysis):
    """Print a formatted summary of the analysis"""
    print(f"\n{'='*60}")
    print(f"STOCK ANALYSIS: {analysis['symbol']}")
    print(f"{'='*60}")
    print(f"Timestamp: {analysis['timestamp']}")

    if "error" in analysis:
        print(f"ERROR: {analysis['error']}")
        return

    # Current metrics
    metrics = analysis.get("current_metrics", {})
    print(f"\nCURRENT METRICS:")
    print(f"Price: ${metrics.get('current_price', 0):.2f}")
    print(f"Change: {metrics.get('price_change_pct', 0):+.2f}%")
    print(f"Volume: {metrics.get('volume', 0):,}")
    if metrics.get("rsi"):
        print(f"RSI: {metrics['rsi']:.1f}")

    # Technical analysis
    technical = analysis.get("technical_analysis", {})
    if technical:
        print(f"\nTECHNICAL SIGNALS:")
        for signal, value in technical.items():
            print(f"{signal.replace('_', ' ').title()}: {value}")

    # News sentiment
    sentiment = analysis.get("news_sentiment", {})
    if sentiment:
        print(f"\nNEWS SENTIMENT:")
        print(f"Overall: {sentiment.get('sentiment', 'N/A').upper()}")
        print(f"Score: {sentiment.get('score', 0):.3f}")
        print(f"Positive keywords: {sentiment.get('positive_keywords', 0)}")
        print(f"Negative keywords: {sentiment.get('negative_keywords', 0)}")

    # Prediction
    prediction = analysis.get("prediction", {})
    if "ensemble_prediction" in prediction:
        ensemble = prediction["ensemble_prediction"]
        print(f"\nPREDICTION (Ensemble - Next 3 Days):")

        # Overall summary
        overall = ensemble.get("overall_summary", {})
        print(f"Overall Direction: {overall.get('overall_direction', 'N/A')}")
        print(f"Final Predicted Price: ${overall.get('final_predicted_price', 0):.2f}")
        print(f"Total Change: {overall.get('total_change', 0)*100:+.2f}%")
        print(f"Average Confidence: {overall.get('average_confidence', 0):.1f}%")

        # Daily breakdown
        daily_predictions = ensemble.get("daily_predictions", [])
        if daily_predictions:
            print(f"\nDaily Breakdown:")
            for day_pred in daily_predictions:
                print(
                    f"Day {day_pred['day']}: {day_pred['direction']} "
                    f"${day_pred['predicted_price']:.2f} "
                    f"({day_pred['predicted_change']*100:+.2f}%) "
                    f"[Conf: {day_pred['confidence']:.1f}%]"
                )

    # Recommendation
    recommendation = analysis.get("recommendation", {})
    if recommendation:
        print(f"\nRECOMMENDATION:")
        print(f"Action: {recommendation['action']}")
        print(f"Confidence: {recommendation['confidence']:.1f}%")
        print(f"Risk Level: {recommendation['risk_level']}")
        if recommendation["reasoning"]:
            print("Reasoning:")
            for reason in recommendation["reasoning"]:
                print(f"  • {reason}")


def print_watchlist_summary(results):
    """Print watchlist analysis summary"""
    print(f"\n{'='*60}")
    print(f"WATCHLIST ANALYSIS")
    print(f"{'='*60}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Stocks analyzed: {results['watchlist_size']}")

    summary = results.get("summary", {})

    print(f"\nMARKET SENTIMENT: {summary.get('market_sentiment', 'N/A')}")

    # Buy candidates
    buy_candidates = summary.get("buy_candidates", [])
    if buy_candidates:
        print(f"\nBUY CANDIDATES ({len(buy_candidates)}):")
        for stock in buy_candidates:
            print(
                f"  {stock['symbol']}: ${stock['current_price']:.2f} "
                f"({stock['price_change_pct']:+.1f}%) - Confidence: {stock['confidence']:.1f}%"
            )

    # Sell candidates
    sell_candidates = summary.get("sell_candidates", [])
    if sell_candidates:
        print(f"\nSELL CANDIDATES ({len(sell_candidates)}):")
        for stock in sell_candidates:
            print(
                f"  {stock['symbol']}: ${stock['current_price']:.2f} "
                f"({stock['price_change_pct']:+.1f}%) - Confidence: {stock['confidence']:.1f}%"
            )

    # Hold recommendations
    hold_recommendations = summary.get("hold_recommendations", [])
    if hold_recommendations:
        print(f"\nHOLD RECOMMENDATIONS ({len(hold_recommendations)}):")
        for stock in hold_recommendations[:5]:  # Show first 5
            print(
                f"  {stock['symbol']}: ${stock['current_price']:.2f} "
                f"({stock['price_change_pct']:+.1f}%)"
            )


def main():
    """Main function"""
    print("Stock Advisor Agent")
    print("==================")

    # Check for help
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            """
Usage: python main.py [OPTIONS] [SYMBOLS...]

Options:
  --demo          Use demo mode (simulated data)
  --days=N        Predict N days ahead (1-7, default: 3)
  --help, -h      Show this help message

Examples:
  python main.py AAPL                  # Analyze Apple
  python main.py --demo AAPL           # Demo mode
  python main.py --days=5 AAPL         # 5-day prediction
  python main.py AAPL GOOGL MSFT       # Multiple stocks
  python main.py                       # Interactive mode
        """
        )
        return

    # Check if demo mode is requested
    demo_mode = "--demo" in sys.argv
    if demo_mode:
        sys.argv.remove("--demo")
        print("Running in DEMO MODE (using simulated data)")

    # Check for custom prediction days
    prediction_days = 3  # Default to 3 days
    for arg in sys.argv:
        if arg.startswith("--days="):
            try:
                prediction_days = int(arg.split("=")[1])
                prediction_days = min(max(prediction_days, 1), 7)  # Limit to 1-7 days
                sys.argv.remove(arg)
                break
            except ValueError:
                print("Invalid days value, using default (3)")

    # Initialize the advisor
    advisor = StockAdvisorAgent(demo_mode=demo_mode)

    if len(sys.argv) == 1:
        # Interactive mode
        print("\nInteractive Mode")
        print("Commands:")
        print("  analyze <SYMBOL> - Analyze a single stock")
        print("  add <SYMBOL> - Add stock to watchlist")
        print("  remove <SYMBOL> - Remove stock from watchlist")
        print("  watchlist - Analyze entire watchlist")
        print("  show - Show current watchlist")
        print("  quit - Exit")

        while True:
            try:
                command = input("\n> ").strip().split()
                if not command:
                    continue

                cmd = command[0].lower()

                if cmd == "quit":
                    break
                elif cmd == "analyze" and len(command) > 1:
                    symbol = command[1].upper()
                    print(f"Analyzing {symbol}...")
                    analysis = advisor.analyze_stock(symbol)
                    print_analysis_summary(analysis)
                elif cmd == "add" and len(command) > 1:
                    symbol = command[1].upper()
                    advisor.add_to_watchlist([symbol])
                    print(f"Added {symbol} to watchlist")
                elif cmd == "remove" and len(command) > 1:
                    symbol = command[1].upper()
                    advisor.remove_from_watchlist([symbol])
                    print(f"Removed {symbol} from watchlist")
                elif cmd == "watchlist":
                    if not advisor.watchlist:
                        print("Watchlist is empty. Add some stocks first.")
                    else:
                        results = advisor.analyze_watchlist()
                        print_watchlist_summary(results)
                elif cmd == "show":
                    print(
                        f"Watchlist: {', '.join(advisor.watchlist) if advisor.watchlist else 'Empty'}"
                    )
                else:
                    print("Invalid command")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    elif len(sys.argv) == 2:
        # Single stock analysis
        symbol = sys.argv[1].upper()
        print(f"Analyzing {symbol}...")
        analysis = advisor.analyze_stock(symbol)
        print_analysis_summary(analysis)

    else:
        # Multiple stocks watchlist
        symbols = [s.upper() for s in sys.argv[1:]]
        advisor.add_to_watchlist(symbols)
        print(f"Analyzing watchlist: {', '.join(symbols)}")
        results = advisor.analyze_watchlist()
        print_watchlist_summary(results)


if __name__ == "__main__":
    main()
