"""
Core functionality for stock data fetching and processing.
"""

from .stock_data import StockDataFetcher
from .news_collector import NewsCollector
from .advisor_agent import StockAdvisorAgent
from .demo_data import generate_demo_stock_data

__all__ = [
    "StockDataFetcher",
    "NewsCollector",
    "StockAdvisorAgent",
    "generate_demo_stock_data",
]
