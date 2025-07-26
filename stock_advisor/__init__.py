"""
Stock Advisor - AI-Powered Stock Analysis and Prediction System

A comprehensive stock analysis tool that combines technical analysis,
reinforcement learning, and news sentiment analysis to provide
intelligent trading recommendations.
"""

__version__ = "1.0.0"
__author__ = "Stock Advisor Team"
__email__ = "contact@stockadvisor.com"

from .core import StockDataFetcher, NewsCollector
from .predictors import (
    TechnicalPredictor,
    RLPredictor,
    EnhancedRLPredictor,
    HybridPredictor,
)
from .ui import StockAdvisorUI

__all__ = [
    "StockDataFetcher",
    "NewsCollector",
    "TechnicalPredictor",
    "RLPredictor",
    "EnhancedRLPredictor",
    "HybridPredictor",
    "StockAdvisorUI",
]
