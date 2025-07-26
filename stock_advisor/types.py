"""
Common type definitions for the Stock Advisor application.

This module defines type aliases and data structures used throughout the application
to improve type safety and code clarity.
"""

from typing import Dict, List, Any, Union, Optional, Protocol, TypedDict
from datetime import datetime
import pandas as pd
from enum import Enum


# Basic type aliases
Symbol = str
Price = float
Percentage = float
Confidence = float  # 0.0 to 1.0
Volume = int
Timestamp = Union[datetime, str]


# Enums for better type safety
class PredictionDirection(Enum):
    """Prediction direction enumeration."""

    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"
    UNKNOWN = "UNKNOWN"


class ModelType(Enum):
    """Machine learning model types."""

    TECHNICAL_ANALYSIS = "technical_analysis"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENHANCED_RL = "enhanced_rl"
    HYBRID = "hybrid"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"


class DataSource(Enum):
    """Data source enumeration."""

    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alphavantage"
    IEX_CLOUD = "iex"
    DEMO = "demo"


class NewsSource(Enum):
    """News source enumeration."""

    NEWS_API = "news_api"
    RSS_FEEDS = "rss_feeds"
    REDDIT = "reddit"
    TWITTER = "twitter"


class RiskLevel(Enum):
    """Risk level enumeration."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# TypedDict definitions for structured data
class DailyPrediction(TypedDict):
    """Structure for a single day's prediction."""

    day: int
    date: str
    predicted_price: Price
    predicted_change_pct: Percentage
    direction: str
    confidence: Confidence
    volume_prediction: Optional[Volume]
    high_prediction: Optional[Price]
    low_prediction: Optional[Price]


class PredictionSummary(TypedDict):
    """Structure for overall prediction summary."""

    direction: str
    final_price: Price
    total_change_pct: Percentage
    avg_confidence: Confidence
    risk_level: str
    target_date: str
    stop_loss_price: Optional[Price]
    take_profit_price: Optional[Price]


class PredictionResult(TypedDict):
    """Complete prediction result structure."""

    symbol: Symbol
    timestamp: str
    current_price: Price
    daily_predictions: List[DailyPrediction]
    overall_summary: PredictionSummary
    method: str
    model_version: Optional[str]
    data_quality_score: Optional[float]
    error: Optional[str]


class StockData(TypedDict):
    """Structure for stock market data."""

    symbol: Symbol
    timestamp: str
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume
    adjusted_close: Optional[Price]


class TechnicalIndicators(TypedDict):
    """Structure for technical analysis indicators."""

    rsi: Optional[float]
    macd: Optional[float]
    macd_signal: Optional[float]
    macd_histogram: Optional[float]
    sma_5: Optional[float]
    sma_10: Optional[float]
    sma_20: Optional[float]
    ema_12: Optional[float]
    ema_26: Optional[float]
    bollinger_upper: Optional[float]
    bollinger_lower: Optional[float]
    bollinger_middle: Optional[float]
    volume_sma: Optional[float]


class NewsArticle(TypedDict):
    """Structure for news articles."""

    title: str
    content: str
    source: str
    publish_date: str
    url: str
    sentiment_score: Optional[float]
    relevance_score: Optional[float]
    symbols_mentioned: List[Symbol]


class SentimentAnalysis(TypedDict):
    """Structure for sentiment analysis results."""

    overall_sentiment: float  # -1.0 (very negative) to 1.0 (very positive)
    confidence: Confidence
    positive_mentions: int
    negative_mentions: int
    neutral_mentions: int
    key_themes: List[str]
    news_articles: List[NewsArticle]


class BacktestResult(TypedDict):
    """Structure for backtesting results."""

    symbol: Symbol
    start_date: str
    end_date: str
    initial_investment: float
    final_value: float
    total_return: Percentage
    annualized_return: Percentage
    max_drawdown: Percentage
    sharpe_ratio: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_duration: float
    transaction_costs: float


class ModelMetrics(TypedDict):
    """Structure for model performance metrics."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mse: Optional[float]  # Mean Squared Error for regression
    mae: Optional[float]  # Mean Absolute Error for regression
    training_time: float
    prediction_time: float
    model_size_mb: float


class TrainingData(TypedDict):
    """Structure for model training data."""

    features: pd.DataFrame
    targets: pd.DataFrame
    split_date: str
    train_size: int
    test_size: int
    validation_size: Optional[int]
    feature_names: List[str]
    target_names: List[str]


# Protocol definitions for dependency injection
class DataFetcherProtocol(Protocol):
    """Protocol for data fetchers."""

    def get_stock_data(self, symbol: Symbol, period: str) -> pd.DataFrame:
        """Fetch stock data for the given symbol and period."""
        ...

    def get_real_time_price(self, symbol: Symbol) -> Optional[Price]:
        """Get real-time price for the symbol."""
        ...


class PredictorProtocol(Protocol):
    """Protocol for prediction models."""

    def predict_next_3_days(self, symbol: Symbol, **kwargs) -> PredictionResult:
        """Generate predictions for the next 3 days."""
        ...

    def train_model(self, symbol: Symbol, data: pd.DataFrame) -> ModelMetrics:
        """Train the model on the provided data."""
        ...


class NewsCollectorProtocol(Protocol):
    """Protocol for news collectors."""

    def collect_news(self, symbol: Symbol, days_back: int) -> List[NewsArticle]:
        """Collect news articles for the symbol."""
        ...

    def analyze_sentiment(self, articles: List[NewsArticle]) -> SentimentAnalysis:
        """Analyze sentiment of news articles."""
        ...


class CacheProtocol(Protocol):
    """Protocol for caching implementations."""

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        ...

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        ...


# Configuration type definitions
ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any]]
ConfigDict = Dict[str, ConfigValue]


# Error handling types
class ErrorContext(TypedDict):
    """Context information for errors."""

    symbol: Optional[Symbol]
    operation: Optional[str]
    timestamp: str
    user_id: Optional[str]
    request_id: Optional[str]


# API response types
class APIResponse(TypedDict):
    """Standard API response structure."""

    success: bool
    data: Optional[Any]
    error: Optional[str]
    error_code: Optional[str]
    timestamp: str
    request_id: Optional[str]


# UI/Display types
class ChartData(TypedDict):
    """Structure for chart display data."""

    x_values: List[str]  # Usually dates
    y_values: List[float]  # Usually prices
    volume: Optional[List[Volume]]
    indicators: Optional[Dict[str, List[float]]]
    title: str
    x_label: str
    y_label: str


class DisplayMetrics(TypedDict):
    """Structure for displaying performance metrics."""

    accuracy: str
    precision: str
    recall: str
    f1_score: str
    additional_metrics: Dict[str, str]


# Utility type aliases
JSONSerializable = Union[
    str, int, float, bool, None, List["JSONSerializable"], Dict[str, "JSONSerializable"]
]
PathLike = Union[str, bytes]  # Simplified path type for compatibility
