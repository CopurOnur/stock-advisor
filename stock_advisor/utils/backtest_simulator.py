#!/usr/bin/env python3
"""
Backtest Simulator for Stock Predictors

Simulates how well each predictor performed over the last 7 days by:
1. Using historical data up to each day
2. Making next-day predictions
3. Comparing predictions to actual results
4. Calculating accuracy metrics and performance scores
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

import sys
import os

# Add the parent directory to the path so we can import from stock_advisor
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from stock_advisor.core.stock_data import StockDataFetcher
    from stock_advisor.predictors.technical_predictor import TechnicalAnalysisPredictor
    from stock_advisor.predictors.rl_predictor import RLStockPredictor
    from stock_advisor.predictors.hybrid_predictor import HybridStockPredictor
except ImportError:
    # Fallback to relative imports if running from within the package
    from ..core.stock_data import StockDataFetcher
    from ..predictors.technical_predictor import TechnicalAnalysisPredictor
    from ..predictors.rl_predictor import RLStockPredictor
    from ..predictors.hybrid_predictor import HybridStockPredictor


class BacktestSimulator:
    """Backtest simulation for predictor performance evaluation"""
    
    def __init__(self):
        self.stock_fetcher = StockDataFetcher()
        self.technical_predictor = TechnicalAnalysisPredictor()
        self.rl_predictor = RLStockPredictor()
        self.hybrid_predictor = HybridStockPredictor()
    
    def simulate_single_day_prediction(self, symbol: str, data_until_date: pd.DataFrame, 
                                     actual_next_price: float, method: str) -> Dict:
        """
        Simulate a single day prediction and compare to actual result
        
        Args:
            symbol: Stock symbol
            data_until_date: Historical data available up to prediction date
            actual_next_price: Actual price the next day
            method: Prediction method ('technical', 'rl', 'hybrid')
        
        Returns:
            Dictionary with prediction results and accuracy metrics
        """
        current_price = data_until_date['Close'].iloc[-1]
        
        try:
            if method == 'technical':
                # Get technical analysis prediction (first day only)
                result = self.technical_predictor.predict_next_3_days(symbol)
                if 'error' in result or not result.get('daily_predictions'):
                    return {'error': 'Technical prediction failed'}
                
                prediction = result['daily_predictions'][0]  # First day
                predicted_price = prediction['predicted_price']
                predicted_direction = prediction['direction']
                confidence = prediction['confidence']
                
            elif method == 'rl':
                # Override RL predictor to use specific data
                self.rl_predictor.stock_fetcher.cache[f"{symbol}_3mo"] = data_until_date
                
                result = self.rl_predictor.predict_next_3_days(symbol, auto_train=False)
                if 'error' in result or not result.get('daily_predictions'):
                    return {'error': 'RL prediction failed'}
                
                prediction = result['daily_predictions'][0]
                predicted_price = prediction['predicted_price']
                predicted_direction = prediction['direction']
                confidence = prediction['confidence']
                
            elif method == 'hybrid':
                # Override hybrid predictor to use specific data
                self.hybrid_predictor.stock_fetcher.cache[f"{symbol}_3mo"] = data_until_date
                
                result = self.hybrid_predictor.predict_next_3_days(symbol, auto_train=False, use_news=False)
                if 'error' in result or not result.get('daily_predictions'):
                    return {'error': 'Hybrid prediction failed'}
                
                prediction = result['daily_predictions'][0]
                predicted_price = prediction['predicted_price']
                predicted_direction = prediction['direction']
                confidence = prediction['confidence']
            
            else:
                return {'error': f'Unknown method: {method}'}
            
            # Calculate actual direction and change
            actual_change_pct = (actual_next_price - current_price) / current_price * 100
            actual_direction = 'UP' if actual_change_pct > 0 else 'DOWN' if actual_change_pct < 0 else 'FLAT'
            
            # Calculate predicted change
            predicted_change_pct = (predicted_price - current_price) / current_price * 100
            
            # Calculate accuracy metrics
            direction_correct = (predicted_direction == actual_direction)
            price_error = abs(predicted_price - actual_next_price)
            price_error_pct = abs(predicted_change_pct - actual_change_pct)
            
            # Confidence-weighted accuracy
            confidence_weight = confidence / 100
            weighted_accuracy = direction_correct * confidence_weight
            
            return {
                'current_price': current_price,
                'predicted_price': predicted_price,
                'actual_price': actual_next_price,
                'predicted_change_pct': predicted_change_pct,
                'actual_change_pct': actual_change_pct,
                'predicted_direction': predicted_direction,
                'actual_direction': actual_direction,
                'direction_correct': direction_correct,
                'price_error': price_error,
                'price_error_pct': price_error_pct,
                'confidence': confidence,
                'weighted_accuracy': weighted_accuracy
            }
            
        except Exception as e:
            return {'error': f'Prediction error: {str(e)}'}
    
    def run_7_day_backtest(self, symbol: str) -> Dict:
        """
        Run 7-day backtest simulation for all prediction methods
        
        Args:
            symbol: Stock symbol to test
            
        Returns:
            Complete backtest results with performance metrics
        """
        symbol = symbol.upper()
        
        # Get extended historical data (need extra days for context)
        print(f"Fetching data for {symbol} backtest...")
        stock_data = self.stock_fetcher.get_stock_data(symbol, period="3mo")
        if stock_data.empty:
            return {'error': f'No data available for {symbol}'}
        
        stock_data = self.stock_fetcher.calculate_technical_indicators(stock_data)
        
        # Get sufficient historical data (need extra days for context + 7 test days)
        if len(stock_data) < 15:
            return {'error': 'Insufficient data for backtesting (need at least 15 days)'}
        
        # Use last 14 days (7 for context + 7 for testing)
        test_data = stock_data.tail(14).copy()
        
        # Pre-train RL models if needed (using older data)
        training_data = stock_data.iloc[:-14] if len(stock_data) > 50 else stock_data.iloc[:-10]
        if not training_data.empty:
            print("Pre-training RL models for backtesting...")
            try:
                self.rl_predictor.train_agent(symbol, episodes=30)
                self.hybrid_predictor.train_hybrid_agent(symbol, episodes=30)
            except Exception as e:
                print(f"Warning: Model training failed: {e}")
        
        # Run simulations for each of the last 7 days
        results = {
            'symbol': symbol,
            'backtest_period': '7 days',
            'methods': {
                'technical': {'daily_results': [], 'performance': {}},
                'rl': {'daily_results': [], 'performance': {}},
                'hybrid': {'daily_results': [], 'performance': {}}
            }
        }
        
        print("Running 7-day backtest simulation...")
        print(f"Total data points available: {len(test_data)}")
        
        # Start from day 7 (index 6) and predict the next 7 days
        for day in range(7):
            # Data available up to day (7 + day), predict day (8 + day)
            context_end_idx = 7 + day  # How much historical data is available
            prediction_day_idx = context_end_idx  # Which day we're predicting
            
            # Make sure we have enough data
            if prediction_day_idx >= len(test_data):
                print(f"  Day {day+1}/7: Insufficient data (need index {prediction_day_idx}, have {len(test_data)})")
                continue
            
            # Get historical data up to the prediction point
            data_until_date = test_data.iloc[:context_end_idx]
            
            # Get actual price for the day we're predicting
            actual_next_price = test_data.iloc[prediction_day_idx]['Close']
            test_date = test_data.index[prediction_day_idx].strftime('%Y-%m-%d')
            prediction_date = test_data.index[context_end_idx - 1].strftime('%Y-%m-%d')
            
            print(f"  Day {day+1}/7: Predicting {test_date} using data up to {prediction_date}")
            print(f"    Context data: {len(data_until_date)} days, Actual price: ${actual_next_price:.2f}")
            
            # Test each method
            for method in ['technical', 'rl', 'hybrid']:
                result = self.simulate_single_day_prediction(
                    symbol, data_until_date, actual_next_price, method
                )
                
                result['test_date'] = test_date
                result['prediction_date'] = prediction_date
                result['day'] = day + 1
                results['methods'][method]['daily_results'].append(result)
        
        # Calculate performance metrics for each method
        for method in ['technical', 'rl', 'hybrid']:
            daily_results = results['methods'][method]['daily_results']
            valid_results = [r for r in daily_results if 'error' not in r]
            
            if valid_results:
                performance = self._calculate_performance_metrics(valid_results)
                results['methods'][method]['performance'] = performance
            else:
                results['methods'][method]['performance'] = {'error': 'No valid predictions'}
        
        return results
    
    def _calculate_performance_metrics(self, daily_results: List[Dict]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not daily_results:
            return {'error': 'No valid results'}
        
        # Direction accuracy
        direction_correct = sum(1 for r in daily_results if r.get('direction_correct', False))
        direction_accuracy = direction_correct / len(daily_results) * 100
        
        # Price prediction errors
        price_errors = [r['price_error'] for r in daily_results if 'price_error' in r]
        price_errors_pct = [r['price_error_pct'] for r in daily_results if 'price_error_pct' in r]
        
        # Confidence metrics
        confidences = [r['confidence'] for r in daily_results if 'confidence' in r]
        weighted_accuracies = [r['weighted_accuracy'] for r in daily_results if 'weighted_accuracy' in r]
        
        # Prediction vs actual changes
        predicted_changes = [r['predicted_change_pct'] for r in daily_results if 'predicted_change_pct' in r]
        actual_changes = [r['actual_change_pct'] for r in daily_results if 'actual_change_pct' in r]
        
        return {
            'total_predictions': len(daily_results),
            'direction_accuracy': direction_accuracy,
            'direction_correct_count': direction_correct,
            'avg_price_error': np.mean(price_errors) if price_errors else 0,
            'avg_price_error_pct': np.mean(price_errors_pct) if price_errors_pct else 0,
            'max_price_error_pct': np.max(price_errors_pct) if price_errors_pct else 0,
            'min_price_error_pct': np.min(price_errors_pct) if price_errors_pct else 0,
            'avg_confidence': np.mean(confidences) if confidences else 0,
            'weighted_accuracy': np.mean(weighted_accuracies) if weighted_accuracies else 0,
            'prediction_bias': np.mean(predicted_changes) if predicted_changes else 0,
            'actual_volatility': np.std(actual_changes) if actual_changes else 0,
            'prediction_correlation': np.corrcoef(predicted_changes, actual_changes)[0,1] 
                                    if len(predicted_changes) > 1 else 0
        }
    
    def print_backtest_summary(self, results: Dict):
        """Print formatted backtest results"""
        if 'error' in results:
            print(f"ERROR: {results['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"7-DAY BACKTEST RESULTS: {results['symbol']}")
        print(f"{'='*80}")
        
        # Overall comparison table
        print(f"\nPERFORMANCE SUMMARY:")
        print(f"{'Method':<20} {'Direction Acc':<15} {'Avg Error %':<12} {'Confidence':<12} {'Correlation':<12}")
        print(f"{'-'*75}")
        
        for method_name, method_key in [('Technical Analysis', 'technical'), 
                                       ('Reinforcement Learning', 'rl'), 
                                       ('Hybrid', 'hybrid')]:
            perf = results['methods'][method_key]['performance']
            if 'error' not in perf:
                print(f"{method_name:<20} "
                      f"{perf['direction_accuracy']:<14.1f}% "
                      f"{perf['avg_price_error_pct']:<11.2f}% "
                      f"{perf['avg_confidence']:<11.1f}% "
                      f"{perf['prediction_correlation']:<11.3f}")
            else:
                print(f"{method_name:<20} {'FAILED':<14} {'-':<11} {'-':<11} {'-':<11}")
        
        # Daily breakdown for each method
        for method_name, method_key in [('Technical Analysis', 'technical'), 
                                       ('Reinforcement Learning', 'rl'), 
                                       ('Hybrid', 'hybrid')]:
            daily_results = results['methods'][method_key]['daily_results']
            valid_results = [r for r in daily_results if 'error' not in r]
            
            if valid_results:
                print(f"\n{method_name.upper()} - DAILY BREAKDOWN:")
                print(f"{'Day':<5} {'Date':<12} {'Predicted':<12} {'Actual':<12} {'Error %':<10} {'Direction':<10} {'Correct':<8}")
                print(f"{'-'*75}")
                
                for result in valid_results:
                    direction_symbol = "✓" if result['direction_correct'] else "✗"
                    print(f"{result['day']:<5} "
                          f"{result['test_date']:<12} "
                          f"{result['predicted_change_pct']:+8.2f}% "
                          f"{result['actual_change_pct']:+9.2f}% "
                          f"{result['price_error_pct']:<9.2f}% "
                          f"{result['predicted_direction']:<10} "
                          f"{direction_symbol:<8}")
            else:
                print(f"\n{method_name.upper()}: No valid predictions")
        
        # Best performer
        best_method = None
        best_score = -1
        
        for method_key in ['technical', 'rl', 'hybrid']:
            perf = results['methods'][method_key]['performance']
            if 'error' not in perf:
                # Combined score: direction accuracy + inverse error - bias penalty
                score = (perf['direction_accuracy'] - 
                        perf['avg_price_error_pct'] * 2 + 
                        perf['prediction_correlation'] * 20)
                
                if score > best_score:
                    best_score = score
                    best_method = method_key
        
        if best_method:
            method_names = {'technical': 'Technical Analysis', 'rl': 'Reinforcement Learning', 'hybrid': 'Hybrid'}
            print(f"\n🏆 BEST PERFORMER: {method_names[best_method]} (Score: {best_score:.1f})")
        
        print(f"\n{'='*80}")
        print("DISCLAIMER: Past performance does not guarantee future results.")
        print("Backtest results are for educational evaluation only.")
        print(f"{'='*80}")
    
    def save_backtest_results(self, results: Dict, filename: str = None):
        """Save backtest results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{results.get('symbol', 'unknown')}_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), 'results', filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Backtest results saved to: {filepath}")
        return filepath


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python backtest_simulator.py <SYMBOL> [--save]")
        print("Example: python backtest_simulator.py AAPL --save")
        return
    
    symbol = sys.argv[1].upper()
    save_results = '--save' in sys.argv
    
    print("7-Day Backtest Simulator")
    print("========================")
    print(f"Symbol: {symbol}")
    print("Testing: Technical Analysis, RL, and Hybrid predictors")
    print("Period: Last 7 trading days")
    print("")
    
    simulator = BacktestSimulator()
    results = simulator.run_7_day_backtest(symbol)
    
    simulator.print_backtest_summary(results)
    
    if save_results and 'error' not in results:
        simulator.save_backtest_results(results)


if __name__ == "__main__":
    main()