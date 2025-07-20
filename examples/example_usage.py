#!/usr/bin/env python3
"""
Example usage of the Stock Advisor Agent
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from advisor_agent import StockAdvisorAgent

def example_single_stock_analysis():
    """Example: Analyze a single stock"""
    print("=== Single Stock Analysis Example ===")
    
    advisor = StockAdvisorAgent()
    
    # Analyze Apple stock
    symbol = "AAPL"
    print(f"Analyzing {symbol}...")
    
    analysis = advisor.analyze_stock(symbol)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    # Display key results
    print(f"\nResults for {symbol}:")
    print(f"Current Price: ${analysis['current_metrics']['current_price']:.2f}")
    print(f"Price Change: {analysis['current_metrics']['price_change_pct']:+.2f}%")
    
    recommendation = analysis['recommendation']
    print(f"\nRecommendation: {recommendation['action']}")
    print(f"Confidence: {recommendation['confidence']:.1f}%")
    print(f"Risk Level: {recommendation['risk_level']}")
    
    if 'ensemble_prediction' in analysis['prediction']:
        pred = analysis['prediction']['ensemble_prediction']
        print(f"\nPrediction:")
        print(f"Direction: {pred['direction']}")
        print(f"Predicted Price: ${pred['predicted_price']:.2f}")
        print(f"Confidence: {pred['confidence']:.1f}%")

def example_watchlist_analysis():
    """Example: Analyze a watchlist of stocks"""
    print("\n=== Watchlist Analysis Example ===")
    
    advisor = StockAdvisorAgent()
    
    # Add some popular stocks to watchlist
    stocks = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    advisor.add_to_watchlist(stocks)
    
    print(f"Analyzing watchlist: {', '.join(stocks)}")
    
    results = advisor.analyze_watchlist()
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        return
    
    summary = results['summary']
    
    print(f"\nMarket Sentiment: {summary['market_sentiment']}")
    
    # Display buy candidates
    if summary['buy_candidates']:
        print(f"\nBuy Candidates ({len(summary['buy_candidates'])}):")
        for stock in summary['buy_candidates']:
            print(f"  {stock['symbol']}: ${stock['current_price']:.2f} "
                  f"({stock['price_change_pct']:+.1f}%) - Confidence: {stock['confidence']:.1f}%")
    
    # Display sell candidates
    if summary['sell_candidates']:
        print(f"\nSell Candidates ({len(summary['sell_candidates'])}):")
        for stock in summary['sell_candidates']:
            print(f"  {stock['symbol']}: ${stock['current_price']:.2f} "
                  f"({stock['price_change_pct']:+.1f}%) - Confidence: {stock['confidence']:.1f}%")

def example_historical_tracking():
    """Example: Track analysis history"""
    print("\n=== Historical Tracking Example ===")
    
    advisor = StockAdvisorAgent()
    
    # Analyze a stock (this will be stored in history)
    advisor.analyze_stock("AAPL")
    
    # Get analysis history
    history = advisor.get_analysis_history(days=30)
    
    print(f"Analysis history (last 30 days): {len(history)} records")
    
    for analysis in history:
        print(f"  {analysis['symbol']} - {analysis['timestamp'][:19]} - "
              f"{analysis.get('recommendation', {}).get('action', 'N/A')}")

def example_custom_configuration():
    """Example: Custom usage patterns"""
    print("\n=== Custom Configuration Example ===")
    
    advisor = StockAdvisorAgent()
    
    # Analyze with different time periods
    print("Analyzing with different time periods...")
    
    # Short-term analysis (1 month)
    short_term = advisor.analyze_stock("AAPL", period="1mo")
    print(f"Short-term (1mo) recommendation: {short_term.get('recommendation', {}).get('action', 'N/A')}")
    
    # Medium-term analysis (3 months)
    medium_term = advisor.analyze_stock("AAPL", period="3mo")
    print(f"Medium-term (3mo) recommendation: {medium_term.get('recommendation', {}).get('action', 'N/A')}")
    
    # Save analysis to file
    filename = advisor.save_analysis(medium_term)
    print(f"Analysis saved to: {filename}")

def main():
    """Run all examples"""
    print("Stock Advisor Agent - Example Usage")
    print("===================================")
    
    try:
        example_single_stock_analysis()
        example_watchlist_analysis()
        example_historical_tracking()
        example_custom_configuration()
        
        print("\n=== Examples Complete ===")
        print("Check the individual functions for more detailed usage patterns.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have installed all dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()