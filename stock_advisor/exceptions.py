"""
Custom exception hierarchy for the Stock Advisor application.

This module defines all custom exceptions used throughout the application
to provide better error classification and handling.
"""

from typing import Optional, Any


class StockAdvisorError(Exception):
    """
    Base exception class for all Stock Advisor related errors.

    All custom exceptions in the application should inherit from this class
    to enable consistent error handling and logging.
    """

    def __init__(
        self,
        message: str,
        details: Optional[Any] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.details = details
        self.cause = cause

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


# Data-related exceptions
class DataError(StockAdvisorError):
    """Base class for data-related errors."""

    pass


class DataFetchError(DataError):
    """Raised when stock data cannot be fetched from any source."""

    def __init__(
        self, symbol: str, source: Optional[str] = None, details: Optional[Any] = None
    ):
        self.symbol = symbol
        self.source = source
        message = f"Failed to fetch data for symbol '{symbol}'"
        if source:
            message += f" from {source}"
        super().__init__(message, details)


class InvalidSymbolError(DataError):
    """Raised when an invalid stock symbol is provided."""

    def __init__(self, symbol: str, reason: Optional[str] = None):
        self.symbol = symbol
        self.reason = reason
        message = f"Invalid stock symbol: '{symbol}'"
        if reason:
            message += f" - {reason}"
        super().__init__(message)


class InsufficientDataError(DataError):
    """Raised when there is insufficient data for analysis."""

    def __init__(self, symbol: str, required_points: int, available_points: int):
        self.symbol = symbol
        self.required_points = required_points
        self.available_points = available_points
        message = f"Insufficient data for '{symbol}': need {required_points}, got {available_points}"
        super().__init__(message)


class DataQualityError(DataError):
    """Raised when data quality issues are detected."""

    def __init__(self, symbol: str, issue: str, details: Optional[Any] = None):
        self.symbol = symbol
        self.issue = issue
        message = f"Data quality issue for '{symbol}': {issue}"
        super().__init__(message, details)


# Model and prediction-related exceptions
class ModelError(StockAdvisorError):
    """Base class for model-related errors."""

    pass


class PredictionError(ModelError):
    """Raised when prediction generation fails."""

    def __init__(self, symbol: str, model_type: str, details: Optional[Any] = None):
        self.symbol = symbol
        self.model_type = model_type
        message = f"Prediction failed for '{symbol}' using {model_type} model"
        super().__init__(message, details)


class ModelLoadError(ModelError):
    """Raised when a model cannot be loaded."""

    def __init__(self, model_path: str, model_type: str, details: Optional[Any] = None):
        self.model_path = model_path
        self.model_type = model_type
        message = f"Failed to load {model_type} model from '{model_path}'"
        super().__init__(message, details)


class ModelSaveError(ModelError):
    """Raised when a model cannot be saved."""

    def __init__(self, model_path: str, model_type: str, details: Optional[Any] = None):
        self.model_path = model_path
        self.model_type = model_type
        message = f"Failed to save {model_type} model to '{model_path}'"
        super().__init__(message, details)


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    def __init__(self, model_type: str, symbol: str, details: Optional[Any] = None):
        self.model_type = model_type
        self.symbol = symbol
        message = f"Training failed for {model_type} model on '{symbol}'"
        super().__init__(message, details)


class ModelValidationError(ModelError):
    """Raised when model validation fails."""

    def __init__(
        self, model_type: str, validation_metric: str, threshold: float, actual: float
    ):
        self.model_type = model_type
        self.validation_metric = validation_metric
        self.threshold = threshold
        self.actual = actual
        message = f"{model_type} model validation failed: {validation_metric} {actual:.3f} below threshold {threshold:.3f}"
        super().__init__(message)


# API and external service exceptions
class APIError(StockAdvisorError):
    """Base class for API-related errors."""

    pass


class RateLimitExceededError(APIError):
    """Raised when API rate limits are exceeded."""

    def __init__(self, service: str, reset_time: Optional[int] = None):
        self.service = service
        self.reset_time = reset_time
        message = f"Rate limit exceeded for {service}"
        if reset_time:
            message += f" - reset in {reset_time} seconds"
        super().__init__(message)


class APIKeyError(APIError):
    """Raised when API key is missing or invalid."""

    def __init__(self, service: str, key_type: str = "API key"):
        self.service = service
        self.key_type = key_type
        message = f"Invalid or missing {key_type} for {service}"
        super().__init__(message)


class NetworkError(APIError):
    """Raised when network connectivity issues occur."""

    def __init__(self, service: str, details: Optional[str] = None):
        self.service = service
        message = f"Network error connecting to {service}"
        if details:
            message += f": {details}"
        super().__init__(message)


class ServiceUnavailableError(APIError):
    """Raised when an external service is unavailable."""

    def __init__(self, service: str, status_code: Optional[int] = None):
        self.service = service
        self.status_code = status_code
        message = f"Service unavailable: {service}"
        if status_code:
            message += f" (HTTP {status_code})"
        super().__init__(message)


# Configuration and validation exceptions
class ConfigurationError(StockAdvisorError):
    """Base class for configuration-related errors."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid."""

    def __init__(self, config_key: str, value: Any, reason: str):
        self.config_key = config_key
        self.value = value
        self.reason = reason
        message = f"Invalid configuration for '{config_key}': {value} - {reason}"
        super().__init__(message)


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(self, config_key: str, config_file: Optional[str] = None):
        self.config_key = config_key
        self.config_file = config_file
        message = f"Missing required configuration: '{config_key}'"
        if config_file:
            message += f" in {config_file}"
        super().__init__(message)


class ValidationError(StockAdvisorError):
    """Raised when input validation fails."""

    def __init__(self, field: str, value: Any, rule: str):
        self.field = field
        self.value = value
        self.rule = rule
        message = f"Validation failed for '{field}': {value} - {rule}"
        super().__init__(message)


# News and sentiment analysis exceptions
class NewsError(StockAdvisorError):
    """Base class for news-related errors."""

    pass


class NewsCollectionError(NewsError):
    """Raised when news collection fails."""

    def __init__(self, symbol: str, source: str, details: Optional[Any] = None):
        self.symbol = symbol
        self.source = source
        message = f"Failed to collect news for '{symbol}' from {source}"
        super().__init__(message, details)


class SentimentAnalysisError(NewsError):
    """Raised when sentiment analysis fails."""

    def __init__(self, text_length: int, model: str, details: Optional[Any] = None):
        self.text_length = text_length
        self.model = model
        message = (
            f"Sentiment analysis failed for text ({text_length} chars) using {model}"
        )
        super().__init__(message, details)


# UI and interface exceptions
class UIError(StockAdvisorError):
    """Base class for UI-related errors."""

    pass


class ChartGenerationError(UIError):
    """Raised when chart generation fails."""

    def __init__(self, chart_type: str, symbol: str, details: Optional[Any] = None):
        self.chart_type = chart_type
        self.symbol = symbol
        message = f"Failed to generate {chart_type} chart for '{symbol}'"
        super().__init__(message, details)


class InterfaceError(UIError):
    """Raised when UI interface encounters an error."""

    def __init__(self, component: str, action: str, details: Optional[Any] = None):
        self.component = component
        self.action = action
        message = f"Interface error in {component} during {action}"
        super().__init__(message, details)


# Backtest and simulation exceptions
class BacktestError(StockAdvisorError):
    """Base class for backtesting-related errors."""

    pass


class SimulationError(BacktestError):
    """Raised when trading simulation fails."""

    def __init__(self, symbol: str, strategy: str, details: Optional[Any] = None):
        self.symbol = symbol
        self.strategy = strategy
        message = f"Simulation failed for '{symbol}' using {strategy} strategy"
        super().__init__(message, details)


class InsufficientFundsError(BacktestError):
    """Raised when simulation runs out of funds."""

    def __init__(self, required: float, available: float):
        self.required = required
        self.available = available
        message = f"Insufficient funds: need ${required:.2f}, have ${available:.2f}"
        super().__init__(message)


# Utility functions for exception handling
def is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an exception represents a retryable error.

    Args:
        exception: The exception to check

    Returns:
        True if the error is retryable, False otherwise
    """
    retryable_types = (
        NetworkError,
        ServiceUnavailableError,
        RateLimitExceededError,
        DataFetchError,
    )
    return isinstance(exception, retryable_types)


def get_error_category(exception: Exception) -> str:
    """
    Get the category of an error for logging and monitoring purposes.

    Args:
        exception: The exception to categorize

    Returns:
        String category name
    """
    if isinstance(exception, DataError):
        return "data"
    elif isinstance(exception, ModelError):
        return "model"
    elif isinstance(exception, APIError):
        return "api"
    elif isinstance(exception, ConfigurationError):
        return "configuration"
    elif isinstance(exception, NewsError):
        return "news"
    elif isinstance(exception, UIError):
        return "ui"
    elif isinstance(exception, BacktestError):
        return "backtest"
    elif isinstance(exception, StockAdvisorError):
        return "application"
    else:
        return "system"
