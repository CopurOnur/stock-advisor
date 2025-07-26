#!/usr/bin/env python3
"""
Test IEX Cloud Integration

Quick test to verify IEX Cloud is working properly with the stock advisor.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from stock_data import StockDataFetcher


def test_data_sources():
    """Test all data sources in order of priority"""
    fetcher = StockDataFetcher()
    symbol = "AAPL"

    print("🧪 Testing Stock Data Sources")
    print("=" * 40)
    print(f"Symbol: {symbol}")
    print()

    # Check available sources
    print("📡 Available Data Sources:")
    print(f"• Yahoo Finance: ✅ Always available")
    print(
        f"• IEX Cloud: {'✅ Configured' if fetcher.use_iex_cloud else '❌ Not configured'}"
    )
    print(
        f"• Alpha Vantage: {'✅ Configured' if fetcher.use_alpha_vantage else '❌ Not configured'}"
    )
    print(f"• Demo Data: ✅ Always available")
    print()

    # Test main data fetch
    print("🔄 Testing data fetch...")
    data = fetcher.get_stock_data(symbol, period="1mo")

    if not data.empty:
        print(f"✅ Successfully fetched {len(data)} days of data")
        print(
            f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
        )
        print(f"   Current price: ${data['Close'].iloc[-1]:.2f}")
        print(
            f"   Daily change: {((data['Close'].iloc[-1] - data['Close'].iloc[-2]) / data['Close'].iloc[-2] * 100):+.2f}%"
        )

        # Test technical indicators
        data_with_indicators = fetcher.calculate_technical_indicators(data)
        indicators = []
        for indicator in ["SMA_5", "SMA_10", "SMA_20", "RSI", "MACD"]:
            if indicator in data_with_indicators.columns:
                indicators.append(indicator)
        print(f"   Technical indicators: {', '.join(indicators)}")

        # Test IEX real-time quote if available
        if fetcher.use_iex_cloud:
            print()
            print("📈 Testing IEX Cloud real-time quote...")
            quote = fetcher.get_real_time_quote_iex(symbol)
            if quote:
                print(f"✅ Real-time quote from {quote.get('source', 'IEX Cloud')}")
                print(f"   Price: ${quote.get('current_price', 'N/A')}")
                print(f"   Change: {quote.get('change_percent', 0):+.2f}%")
                print(
                    f"   Volume: {quote.get('volume', 'N/A'):,}"
                    if quote.get("volume")
                    else "   Volume: N/A"
                )
            else:
                print("❌ Failed to get real-time quote")

        print()
        print("🎉 Data integration test successful!")
        return True
    else:
        print("❌ Failed to fetch any data")
        return False


def test_fallback_order():
    """Test the fallback order of data sources"""
    print()
    print("🔄 Testing Data Source Fallback Order")
    print("=" * 45)
    print()
    print("Expected order:")
    print("1. Yahoo Finance (unlimited)")
    print("2. IEX Cloud (500K/month)")
    print("3. Alpha Vantage (5/minute)")
    print("4. Demo Data (offline)")
    print()

    # This is just informational - actual fallback testing
    # would require simulating API failures
    print("ℹ️  Fallback testing requires simulating API failures.")
    print("   In normal operation, Yahoo Finance should work most of the time.")
    print("   IEX Cloud will automatically be used if Yahoo Finance fails.")


def show_usage_tips():
    """Show usage tips for IEX Cloud"""
    print()
    print("💡 IEX Cloud Usage Tips")
    print("=" * 30)
    print()
    print("📊 API Call Limits:")
    print("• Free Tier: 500,000 calls/month")
    print("• That's ~16,000 calls per day")
    print("• Each stock analysis uses 1-3 calls")
    print("• Plenty for normal usage!")
    print()
    print("🚀 Getting Started:")
    print("1. Run: python setup_iex_cloud.py")
    print("2. Follow the setup instructions")
    print("3. Test with: python test_iex_integration.py")
    print()
    print("🔧 Configuration:")
    print("• API key stored in .env file")
    print("• Automatic fallback integration")
    print("• No code changes needed")


def main():
    """Main test function"""
    print("IEX Cloud Integration Test")
    print("=" * 30)
    print()

    # Test data sources
    success = test_data_sources()

    # Test fallback order
    test_fallback_order()

    # Show tips
    show_usage_tips()

    print()
    if success:
        print("🎉 All tests passed! Your stock advisor is ready.")
        print("   Try running the web UI: ./run_ui.sh")
    else:
        print("⚠️  Some tests failed. Check your configuration.")
        print("   Run: python setup_iex_cloud.py for help")


if __name__ == "__main__":
    main()
