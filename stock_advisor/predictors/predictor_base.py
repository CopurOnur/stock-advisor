"""
Base class for all stock predictors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import os

class BasePredictor(ABC):
    """
    Abstract base class for stock predictors.
    
    All predictors should inherit from this class and implement
    the predict_next_3_days method.
    """
    
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        os.makedirs(self.model_dir, exist_ok=True)
    
    @abstractmethod
    def predict_next_3_days(self, symbol: str, **kwargs) -> Dict[str, Any]:
        """
        Predict stock movement for the next 3 days.
        
        Args:
            symbol: Stock symbol to predict
            **kwargs: Additional arguments specific to the predictor
            
        Returns:
            Dictionary containing prediction results with standardized format:
            {
                'symbol': str,
                'timestamp': str,
                'current_price': float,
                'daily_predictions': List[Dict],
                'overall_summary': Dict,
                'method': str
            }
        """
        pass
    
    def print_prediction_summary(self, result: Dict[str, Any]):
        """
        Print a formatted summary of prediction results.
        
        Args:
            result: Prediction result dictionary
        """
        if 'error' in result:
            print(f"ERROR: {result['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"PREDICTION SUMMARY: {result['symbol']}")
        print(f"Method: {result.get('method', 'Unknown')}")
        print(f"{'='*60}")
        print(f"Current Price: ${result['current_price']:.2f}")
        
        if 'overall_summary' in result:
            summary = result['overall_summary']
            print(f"Overall Direction: {summary.get('direction', 'N/A')}")
            print(f"Target Price: ${summary.get('final_price', 0):.2f}")
            print(f"Total Change: {summary.get('total_change_pct', 0):+.2f}%")
            print(f"Avg Confidence: {summary.get('avg_confidence', 0):.1f}%")
        
        if 'daily_predictions' in result:
            print(f"\nDaily Predictions:")
            for pred in result['daily_predictions']:
                print(f"Day {pred['day']}: {pred['direction']} "
                      f"${pred['predicted_price']:.2f} "
                      f"({pred['predicted_change_pct']:+.2f}%)")
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate stock symbol format.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if symbol is valid, False otherwise
        """
        if not symbol or not isinstance(symbol, str):
            return False
        
        # Basic validation - letters and numbers only, reasonable length
        symbol = symbol.strip().upper()
        if not symbol.isalnum() or len(symbol) > 10:
            return False
        
        return True
    
    def format_prediction_result(self, symbol: str, current_price: float, 
                               daily_predictions: list, method: str) -> Dict[str, Any]:
        """
        Format prediction results into standardized format.
        
        Args:
            symbol: Stock symbol
            current_price: Current stock price
            daily_predictions: List of daily prediction dictionaries
            method: Name of the prediction method
            
        Returns:
            Formatted prediction result dictionary
        """
        from datetime import datetime
        
        # Calculate overall summary
        if daily_predictions:
            final_price = daily_predictions[-1]['predicted_price']
            total_change_pct = ((final_price - current_price) / current_price) * 100
            avg_confidence = sum(p['confidence'] for p in daily_predictions) / len(daily_predictions)
            
            overall_direction = 'UP' if total_change_pct > 0 else 'DOWN' if total_change_pct < 0 else 'FLAT'
            
            overall_summary = {
                'direction': overall_direction,
                'final_price': final_price,
                'total_change_pct': total_change_pct,
                'avg_confidence': avg_confidence
            }
        else:
            overall_summary = {}
        
        return {
            'symbol': symbol.upper(),
            'timestamp': datetime.now().isoformat(),
            'current_price': current_price,
            'daily_predictions': daily_predictions,
            'overall_summary': overall_summary,
            'method': method
        }