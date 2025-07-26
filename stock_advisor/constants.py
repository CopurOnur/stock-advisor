"""
Constants and configuration values for the Stock Advisor application.

This module contains all magic numbers, thresholds, and configuration values
used throughout the application to improve maintainability and consistency.
"""

# No imports needed for constants

# Version
__version__ = "1.0.0"


# Time and Date
class TimeConstants:
    """Time-related constants for data fetching and analysis."""

    DEFAULT_PERIOD = "3mo"
    PREDICTION_PERIOD = "3d"
    MAX_PREDICTION_DAYS = 7
    DEFAULT_PREDICTION_DAYS = 3
    CACHE_TTL_SECONDS = 3600  # 1 hour
    DATA_FETCH_TIMEOUT = 30  # seconds


# Technical Analysis
class TechnicalIndicators:
    """Constants for technical analysis calculations."""

    # Moving Averages
    SMA_SHORT_PERIOD = 5
    SMA_MEDIUM_PERIOD = 10
    SMA_LONG_PERIOD = 20
    EMA_FAST_PERIOD = 12
    EMA_SLOW_PERIOD = 26

    # RSI
    RSI_PERIOD = 14
    RSI_OVERBOUGHT_THRESHOLD = 70
    RSI_OVERSOLD_THRESHOLD = 30

    # MACD
    MACD_FAST_PERIOD = 12
    MACD_SLOW_PERIOD = 26
    MACD_SIGNAL_PERIOD = 9

    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD_DEV = 2


# Machine Learning
class MLConstants:
    """Constants for machine learning models."""

    # Reinforcement Learning
    RL_EPISODES = 1000
    RL_LEARNING_RATE = 0.001
    RL_EPSILON = 0.1
    RL_EPSILON_DECAY = 0.995
    RL_EPSILON_MIN = 0.01
    RL_GAMMA = 0.95
    RL_MEMORY_SIZE = 2000
    RL_BATCH_SIZE = 32

    # Random Forest
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_RANDOM_STATE = 42

    # Gradient Boosting
    GB_N_ESTIMATORS = 100
    GB_LEARNING_RATE = 0.1
    GB_MAX_DEPTH = 6

    # Model Training
    TRAIN_TEST_SPLIT = 0.8
    CROSS_VALIDATION_FOLDS = 5


# Risk Management
class RiskManagement:
    """Risk management and trading constants."""

    MAX_POSITION_SIZE = 0.1  # 10% of portfolio
    DEFAULT_STOP_LOSS = 0.05  # 5%
    DEFAULT_TAKE_PROFIT = 0.15  # 15%
    TRANSACTION_COST = 0.001  # 0.1%
    MIN_CONFIDENCE_THRESHOLD = 0.6
    RISK_FREE_RATE = 0.02  # 2% annual


# Data Sources
class DataSources:
    """Configuration for data source priorities and settings."""

    PRIMARY_SOURCE = "yahoo"
    FALLBACK_SOURCES = ["iex", "alphavantage", "demo"]

    # Yahoo Finance
    YAHOO_MAX_RETRIES = 3
    YAHOO_RETRY_DELAY = 1  # seconds

    # API Rate Limits (calls per minute)
    YAHOO_RATE_LIMIT = 100
    IEX_RATE_LIMIT = 100
    ALPHAVANTAGE_RATE_LIMIT = 5
    NEWS_API_RATE_LIMIT = 60


# News and Sentiment
class NewsConstants:
    """Constants for news collection and sentiment analysis."""

    MAX_NEWS_ARTICLES = 50
    NEWS_LOOKBACK_DAYS = 7
    SENTIMENT_CONFIDENCE_THRESHOLD = 0.7
    NEWS_RELEVANCE_THRESHOLD = 0.8

    # Sentiment Scores
    SENTIMENT_VERY_NEGATIVE = -1.0
    SENTIMENT_NEGATIVE = -0.5
    SENTIMENT_NEUTRAL = 0.0
    SENTIMENT_POSITIVE = 0.5
    SENTIMENT_VERY_POSITIVE = 1.0


# File Paths and Storage
class FilePaths:
    """File paths and storage configuration."""

    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    DATA_DIR = "data"
    CONFIG_DIR = "configs"
    CACHE_DIR = ".cache"

    # File Extensions
    MODEL_EXTENSION = ".pkl"
    LOG_EXTENSION = ".log"
    CONFIG_EXTENSION = ".yaml"


# UI Configuration
class UIConstants:
    """Constants for the user interface."""

    DEFAULT_CHART_HEIGHT = 400
    DEFAULT_CHART_WIDTH = 800
    MAX_SYMBOLS_DISPLAY = 10
    REFRESH_INTERVAL_SECONDS = 60

    # Chart Colors
    BULLISH_COLOR = "#26a69a"  # Green
    BEARISH_COLOR = "#ef5350"  # Red
    NEUTRAL_COLOR = "#ffa726"  # Orange


# Validation
class ValidationRules:
    """Input validation rules and patterns."""

    MIN_SYMBOL_LENGTH = 1
    MAX_SYMBOL_LENGTH = 10
    VALID_SYMBOL_PATTERN = r"^[A-Z0-9.-]+$"

    # Supported time periods
    VALID_PERIODS = [
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "ytd",
        "max",
    ]

    # Prediction constraints
    MIN_PREDICTION_DAYS = 1
    MAX_PREDICTION_DAYS = 30


# Logging
class LoggingConstants:
    """Logging configuration constants."""

    DEFAULT_LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    MAX_LOG_SIZE_MB = 10
    MAX_LOG_FILES = 5


# Error Messages
class ErrorMessages:
    """Standard error messages used throughout the application."""

    INVALID_SYMBOL = "Invalid stock symbol format"
    SYMBOL_NOT_FOUND = "Stock symbol not found"
    DATA_FETCH_FAILED = "Failed to fetch stock data"
    PREDICTION_FAILED = "Failed to generate prediction"
    MODEL_LOAD_FAILED = "Failed to load ML model"
    MODEL_SAVE_FAILED = "Failed to save ML model"
    CONFIG_LOAD_FAILED = "Failed to load configuration"
    NETWORK_ERROR = "Network connection error"
    API_LIMIT_EXCEEDED = "API rate limit exceeded"
    INSUFFICIENT_DATA = "Insufficient data for analysis"


# Status Messages
class StatusMessages:
    """Standard status messages for user feedback."""

    FETCHING_DATA = "Fetching stock data..."
    ANALYZING_DATA = "Analyzing data..."
    GENERATING_PREDICTION = "Generating prediction..."
    TRAINING_MODEL = "Training ML model..."
    SAVING_MODEL = "Saving model..."
    LOADING_MODEL = "Loading model..."
    PROCESSING_NEWS = "Processing news sentiment..."
    CALCULATION_COMPLETE = "Analysis complete"


# Demo Data
class DemoConstants:
    """Constants for demo/test data generation."""

    DEMO_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "NVDA", "BTC-USD", "ETH-USD"]
    DEMO_PRICE_RANGE = (50, 500)
    DEMO_VOLATILITY_RANGE = (0.01, 0.05)
    DEMO_TREND_STRENGTH = 0.02
    DEMO_DATA_POINTS = 252  # Trading days in a year
