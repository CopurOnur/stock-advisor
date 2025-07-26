#!/usr/bin/env python3
"""
Test script to verify stock data fetching works properly
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from stock_data import StockDataFetcher


def test_stock_data_fetch():
    """Test stock data fetching for common symbols"""
    fetcher = StockDataFetcher()

    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]

    print("Testing Stock Data Fetching")
    print("=" * 40)

    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")

        try:
            # Test basic data fetch
            data = fetcher.get_stock_data(symbol, period="1mo")

            if not data.empty:
                print(f"✅ {symbol}: Successfully fetched {len(data)} days of data")
                print(
                    f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
                )
                print(f"   Current price: ${data['Close'].iloc[-1]:.2f}")

                # Test technical indicators
                data_with_indicators = fetcher.calculate_technical_indicators(data)
                indicators = ["SMA_5", "SMA_10", "SMA_20", "RSI", "MACD"]
                available_indicators = [
                    ind for ind in indicators if ind in data_with_indicators.columns
                ]
                print(f"   Technical indicators: {', '.join(available_indicators)}")

            else:
                print(f"❌ {symbol}: No data returned")

        except Exception as e:
            print(f"❌ {symbol}: Error - {str(e)}")

    print("\n" + "=" * 40)
    print("Test completed!")


if __name__ == "__main__":
    test_stock_data_fetch()
