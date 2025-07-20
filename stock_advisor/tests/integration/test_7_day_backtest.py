#!/usr/bin/env python3
"""
Test 7-Day Backtest Fix

This script tests the 7-day backtest to ensure it returns predictions for all 7 days.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from backtest_simulator import BacktestSimulator

def test_7_day_backtest_comprehensive():
    """Test the 7-day backtest with detailed debugging"""
    print("🧪 Testing 7-Day Backtest Fix")
    print("=" * 50)
    
    simulator = BacktestSimulator()
    symbol = "AAPL"
    
    print(f"Testing symbol: {symbol}")
    print()
    
    # Run the backtest
    print("Running 7-day backtest...")
    results = simulator.run_7_day_backtest(symbol)
    
    if 'error' in results:
        print(f"❌ Backtest failed: {results['error']}")
        return False
    
    # Analyze results
    print("\n📊 Backtest Results Analysis:")
    print("=" * 40)
    
    methods = ['technical', 'rl', 'hybrid']
    method_names = {
        'technical': 'Technical Analysis',
        'rl': 'Reinforcement Learning',
        'hybrid': 'Hybrid'
    }
    
    all_good = True
    
    for method in methods:
        if method in results['methods']:
            daily_results = results['methods'][method]['daily_results']
            valid_results = [r for r in daily_results if 'error' not in r]
            error_results = [r for r in daily_results if 'error' in r]
            
            print(f"\n{method_names[method]}:")
            print(f"  Total predictions: {len(daily_results)}")
            print(f"  Valid predictions: {len(valid_results)}")
            print(f"  Failed predictions: {len(error_results)}")
            
            if len(valid_results) == 7:
                print("  ✅ All 7 days predicted successfully!")
            elif len(valid_results) > 0:
                print(f"  ⚠️  Only {len(valid_results)}/7 days predicted")
                all_good = False
                
                # Show which days are missing
                predicted_days = set(r['day'] for r in valid_results)
                missing_days = set(range(1, 8)) - predicted_days
                if missing_days:
                    print(f"  Missing days: {sorted(missing_days)}")
            else:
                print("  ❌ No valid predictions")
                all_good = False
            
            # Show error details if any
            if error_results:
                print(f"  Errors encountered:")
                for i, error_result in enumerate(error_results[:3]):  # Show first 3 errors
                    day = error_result.get('day', 'unknown')
                    error = error_result.get('error', 'unknown error')
                    print(f"    Day {day}: {error}")
                if len(error_results) > 3:
                    print(f"    ... and {len(error_results) - 3} more errors")
    
    # Detailed day-by-day analysis for first method that has results
    print("\n📅 Day-by-Day Analysis:")
    print("=" * 30)
    
    for method in methods:
        if method in results['methods']:
            daily_results = results['methods'][method]['daily_results']
            valid_results = [r for r in daily_results if 'error' not in r]
            
            if valid_results:
                print(f"\n{method_names[method]} - Daily Breakdown:")
                print(f"{'Day':<5} {'Test Date':<12} {'Pred Date':<12} {'Predicted':<12} {'Actual':<12} {'Correct':<8}")
                print("-" * 70)
                
                for result in sorted(valid_results, key=lambda x: x['day']):
                    day = result['day']
                    test_date = result['test_date']
                    pred_date = result.get('prediction_date', 'N/A')
                    predicted_change = result['predicted_change_pct']
                    actual_change = result['actual_change_pct']
                    correct = "✓" if result['direction_correct'] else "✗"
                    
                    print(f"{day:<5} {test_date:<12} {pred_date:<12} "
                          f"{predicted_change:+8.2f}% {actual_change:+9.2f}% {correct:<8}")
                
                break  # Only show detailed breakdown for first working method
    
    # Performance summary
    print("\n🎯 Performance Summary:")
    print("=" * 25)
    
    for method in methods:
        if method in results['methods'] and 'performance' in results['methods'][method]:
            perf = results['methods'][method]['performance']
            
            if 'error' not in perf:
                print(f"\n{method_names[method]}:")
                print(f"  Direction Accuracy: {perf['direction_accuracy']:.1f}%")
                print(f"  Correct Predictions: {perf['direction_correct_count']}/7")
                print(f"  Average Error: {perf['avg_price_error_pct']:.2f}%")
                print(f"  Prediction Correlation: {perf['prediction_correlation']:.3f}")
    
    # Final assessment
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 SUCCESS: 7-day backtest is working correctly!")
        print("   All methods are generating predictions for all 7 days.")
    else:
        print("⚠️  PARTIAL SUCCESS: Some issues detected.")
        print("   Not all methods are generating 7 days of predictions.")
    
    print("\n💡 Next steps:")
    print("   • Test in web UI: ./run_ui.sh")
    print("   • Check 'Include 7-Day Backtest' option")
    print("   • Verify all 7 days are shown in results")
    
    return all_good

def test_different_symbols():
    """Test backtest with different symbols"""
    print("\n🧪 Testing Multiple Symbols")
    print("=" * 35)
    
    simulator = BacktestSimulator()
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        
        try:
            results = simulator.run_7_day_backtest(symbol)
            
            if 'error' in results:
                print(f"  ❌ Failed: {results['error']}")
                continue
            
            # Count valid predictions across all methods
            total_valid = 0
            total_possible = 0
            
            for method in ['technical', 'rl', 'hybrid']:
                if method in results['methods']:
                    daily_results = results['methods'][method]['daily_results']
                    valid_results = [r for r in daily_results if 'error' not in r]
                    total_valid += len(valid_results)
                    total_possible += 7  # 7 days per method
            
            success_rate = (total_valid / total_possible * 100) if total_possible > 0 else 0
            print(f"  ✅ Success: {total_valid}/{total_possible} predictions ({success_rate:.1f}%)")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

def main():
    """Main test function"""
    print("7-Day Backtest Comprehensive Test")
    print("=" * 40)
    print()
    
    # Test main functionality
    success = test_7_day_backtest_comprehensive()
    
    # Test multiple symbols
    test_different_symbols()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 7-day backtest is working correctly!")
    else:
        print("⚠️  Some issues detected. Check the detailed output above.")
    
    print("\nThe backtest should now show predictions for all 7 days.")
    print("Try it in the web UI with any stock symbol!")

if __name__ == "__main__":
    main()