#!/usr/bin/env python3
"""
Test Yahoo Finance Fixes

This script tests the improved Yahoo Finance integration to verify
all the fixes are working properly.
"""

import sys
import os
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from stock_data import StockDataFetcher


def test_basic_yahoo_finance():
    """Test basic Yahoo Finance functionality"""
    print("🧪 Testing Basic Yahoo Finance")
    print("=" * 40)

    fetcher = StockDataFetcher()
    test_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "SPY"]

    results = {}

    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        start_time = time.time()

        try:
            data = fetcher.get_stock_data(symbol, period="1mo")
            elapsed = time.time() - start_time

            if not data.empty:
                results[symbol] = {
                    "status": "success",
                    "days": len(data),
                    "current_price": data["Close"].iloc[-1],
                    "date_range": f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
                    "fetch_time": elapsed,
                }
                print(
                    f"✅ {symbol}: {len(data)} days, ${data['Close'].iloc[-1]:.2f}, {elapsed:.2f}s"
                )
            else:
                results[symbol] = {"status": "failed", "fetch_time": elapsed}
                print(f"❌ {symbol}: No data, {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            results[symbol] = {
                "status": "error",
                "error": str(e),
                "fetch_time": elapsed,
            }
            print(f"❌ {symbol}: Error - {str(e)[:50]}..., {elapsed:.2f}s")

    return results


def test_different_periods():
    """Test different time periods"""
    print("\n🧪 Testing Different Time Periods")
    print("=" * 40)

    fetcher = StockDataFetcher()
    symbol = "AAPL"
    periods = ["5d", "1mo", "3mo", "6mo", "1y"]

    results = {}

    for period in periods:
        print(f"\nTesting {symbol} with period {period}...")
        start_time = time.time()

        try:
            data = fetcher.get_stock_data(symbol, period=period)
            elapsed = time.time() - start_time

            if not data.empty:
                results[period] = {
                    "status": "success",
                    "days": len(data),
                    "fetch_time": elapsed,
                }
                print(f"✅ {period}: {len(data)} days, {elapsed:.2f}s")
            else:
                results[period] = {"status": "failed", "fetch_time": elapsed}
                print(f"❌ {period}: No data, {elapsed:.2f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            results[period] = {
                "status": "error",
                "error": str(e),
                "fetch_time": elapsed,
            }
            print(f"❌ {period}: Error - {str(e)[:50]}..., {elapsed:.2f}s")

    return results


def test_invalid_symbols():
    """Test handling of invalid symbols"""
    print("\n🧪 Testing Invalid Symbol Handling")
    print("=" * 40)

    fetcher = StockDataFetcher()
    invalid_symbols = ["INVALID123", "NOTREAL", "FAKESYM"]

    for symbol in invalid_symbols:
        print(f"\nTesting invalid symbol {symbol}...")
        start_time = time.time()

        try:
            data = fetcher.get_stock_data(symbol, period="1mo")
            elapsed = time.time() - start_time

            if data.empty:
                print(f"✅ {symbol}: Correctly handled as invalid, {elapsed:.2f}s")
            else:
                print(
                    f"⚠️  {symbol}: Unexpectedly got data ({len(data)} days), {elapsed:.2f}s"
                )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✅ {symbol}: Correctly failed with error, {elapsed:.2f}s")


def test_technical_indicators():
    """Test technical indicator calculation"""
    print("\n🧪 Testing Technical Indicators")
    print("=" * 40)

    fetcher = StockDataFetcher()
    symbol = "AAPL"

    print(f"Testing technical indicators for {symbol}...")

    try:
        # Get raw data
        data = fetcher.get_stock_data(symbol, period="3mo")
        if data.empty:
            print("❌ No raw data available for technical indicators")
            return False

        print(f"✅ Got {len(data)} days of raw data")

        # Calculate technical indicators
        data_with_indicators = fetcher.calculate_technical_indicators(data)

        # Check which indicators were added
        indicators = [
            "SMA_5",
            "SMA_10",
            "SMA_20",
            "EMA_12",
            "EMA_26",
            "RSI",
            "MACD",
            "MACD_Signal",
        ]
        found_indicators = []

        for indicator in indicators:
            if indicator in data_with_indicators.columns:
                latest_value = data_with_indicators[indicator].iloc[-1]
                if not pd.isna(latest_value):
                    found_indicators.append(f"{indicator}: {latest_value:.2f}")

        if found_indicators:
            print(f"✅ Technical indicators calculated:")
            for indicator in found_indicators:
                print(f"   {indicator}")
            return True
        else:
            print("❌ No technical indicators calculated")
            return False

    except Exception as e:
        print(f"❌ Technical indicators failed: {e}")
        return False


def test_caching():
    """Test data caching functionality"""
    print("\n🧪 Testing Data Caching")
    print("=" * 40)

    fetcher = StockDataFetcher()
    symbol = "AAPL"
    period = "1mo"

    print(f"First fetch of {symbol}...")
    start_time = time.time()
    data1 = fetcher.get_stock_data(symbol, period=period)
    first_fetch_time = time.time() - start_time

    if data1.empty:
        print("❌ First fetch failed")
        return False

    print(f"✅ First fetch: {len(data1)} days, {first_fetch_time:.2f}s")

    print(f"Second fetch of {symbol} (should use cache)...")
    start_time = time.time()
    data2 = fetcher.get_stock_data(symbol, period=period)
    second_fetch_time = time.time() - start_time

    if data2.empty:
        print("❌ Second fetch failed")
        return False

    print(f"✅ Second fetch: {len(data2)} days, {second_fetch_time:.2f}s")

    # Check if caching worked (second fetch should be much faster)
    if second_fetch_time < first_fetch_time * 0.1:
        print(
            f"✅ Caching working properly (speedup: {first_fetch_time/second_fetch_time:.1f}x)"
        )
        return True
    else:
        print(
            f"⚠️  Caching may not be working (speedup: {first_fetch_time/second_fetch_time:.1f}x)"
        )
        return False


def run_comprehensive_test():
    """Run all tests and provide summary"""
    print("🚀 Yahoo Finance Comprehensive Test")
    print("=" * 50)
    print()

    # Import pandas for isna check
    import pandas as pd

    globals()["pd"] = pd

    test_results = {}

    # Run all tests
    print("Running test suite...")

    try:
        test_results["basic"] = test_basic_yahoo_finance()
        test_results["periods"] = test_different_periods()
        test_invalid_symbols()
        test_results["indicators"] = test_technical_indicators()
        test_results["caching"] = test_caching()
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    # Basic functionality
    if "basic" in test_results:
        basic_results = test_results["basic"]
        successful = sum(1 for r in basic_results.values() if r["status"] == "success")
        total = len(basic_results)
        print(f"Basic Data Fetch: {successful}/{total} symbols successful")

        if successful > 0:
            avg_time = (
                sum(
                    r["fetch_time"]
                    for r in basic_results.values()
                    if r["status"] == "success"
                )
                / successful
            )
            print(f"Average fetch time: {avg_time:.2f}s")

    # Period testing
    if "periods" in test_results:
        period_results = test_results["periods"]
        successful = sum(1 for r in period_results.values() if r["status"] == "success")
        total = len(period_results)
        print(f"Period Variations: {successful}/{total} periods successful")

    # Other tests
    indicators_ok = test_results.get("indicators", False)
    caching_ok = test_results.get("caching", False)

    print(f"Technical Indicators: {'✅ Working' if indicators_ok else '❌ Failed'}")
    print(f"Data Caching: {'✅ Working' if caching_ok else '❌ Failed'}")

    # Overall assessment
    print()
    if successful >= total * 0.8 and indicators_ok:
        print("🎉 OVERALL: Yahoo Finance fixes are working well!")
        print("   Your stock advisor should now be much more reliable.")
    elif successful >= total * 0.5:
        print("⚠️  OVERALL: Partial success. Some issues may remain.")
        print("   Consider using IEX Cloud as backup (run setup_iex_cloud.py)")
    else:
        print("❌ OVERALL: Significant issues detected.")
        print("   Recommend using demo mode or setting up IEX Cloud.")

    print()
    print("💡 Next steps:")
    print("   • Test the web UI: ./run_ui.sh")
    print("   • Set up IEX Cloud backup: python setup_iex_cloud.py")
    print("   • Use demo mode if issues persist")


if __name__ == "__main__":
    run_comprehensive_test()
