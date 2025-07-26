"""
Centralized configuration management using Pydantic for validation and type safety.

This module provides a robust configuration system that validates all settings,
handles environment variables, and provides type-safe access to configuration values.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseSettings, Field, validator
import yaml

from stock_advisor.constants import (
    TimeConstants,
    TechnicalIndicators,
    MLConstants,
    RiskManagement,
    DataSources as DataSourceConstants,
    UIConstants,
    LoggingConstants,
)


class DataSourceConfig(BaseSettings):
    """Configuration for data sources."""

    enabled: bool = True
    api_key: Optional[str] = None
    rate_limit: int = 100
    timeout: int = TimeConstants.DATA_FETCH_TIMEOUT
    max_retries: int = 3
    retry_delay: int = 1

    class Config:
        env_prefix = ""


class YahooFinanceConfig(DataSourceConfig):
    """Yahoo Finance specific configuration."""

    rate_limit: int = DataSourceConstants.YAHOO_RATE_LIMIT
    max_retries: int = DataSourceConstants.YAHOO_MAX_RETRIES
    retry_delay: int = DataSourceConstants.YAHOO_RETRY_DELAY


class AlphaVantageConfig(DataSourceConfig):
    """Alpha Vantage specific configuration."""

    enabled: bool = False
    api_key: Optional[str] = Field(None, env="ALPHAVANTAGE_API_KEY")
    rate_limit: int = DataSourceConstants.ALPHAVANTAGE_RATE_LIMIT


class IEXCloudConfig(DataSourceConfig):
    """IEX Cloud specific configuration."""

    enabled: bool = False
    api_key: Optional[str] = Field(None, env="IEX_API_KEY")
    rate_limit: int = DataSourceConstants.IEX_RATE_LIMIT


class DataSourcesConfig(BaseSettings):
    """Configuration for all data sources."""

    primary: str = Field(default="yahoo", description="Primary data source")
    fallback_sources: List[str] = Field(
        default_factory=lambda: ["iex", "alphavantage", "demo"]
    )

    yahoo: YahooFinanceConfig = Field(default_factory=YahooFinanceConfig)
    alphavantage: AlphaVantageConfig = Field(default_factory=AlphaVantageConfig)
    iex: IEXCloudConfig = Field(default_factory=IEXCloudConfig)

    @validator("primary")
    def validate_primary_source(cls, v):
        valid_sources = ["yahoo", "alphavantage", "iex", "demo"]
        if v not in valid_sources:
            raise ValueError(f"Primary source must be one of {valid_sources}")
        return v

    @validator("fallback_sources")
    def validate_fallback_sources(cls, v):
        valid_sources = ["yahoo", "alphavantage", "iex", "demo"]
        for source in v:
            if source not in valid_sources:
                raise ValueError(
                    f"Fallback source '{source}' must be one of {valid_sources}"
                )
        return v


class NewsAPIConfig(BaseSettings):
    """News API configuration."""

    enabled: bool = False
    api_key: Optional[str] = Field(None, env="NEWS_API_KEY")
    rate_limit: int = DataSourceConstants.NEWS_API_RATE_LIMIT


class RSSFeedsConfig(BaseSettings):
    """RSS feeds configuration."""

    enabled: bool = True
    sources: List[str] = Field(
        default_factory=lambda: [
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://feeds.reuters.com/news/us",
            "https://rss.cnn.com/rss/money_latest.rss",
        ]
    )


class NewsSourcesConfig(BaseSettings):
    """Configuration for news sources."""

    enabled: bool = True
    max_articles: int = 50
    lookback_days: int = 7
    sentiment_threshold: float = 0.7

    newsapi: NewsAPIConfig = Field(default_factory=NewsAPIConfig)
    rss_feeds: RSSFeedsConfig = Field(default_factory=RSSFeedsConfig)

    # LLM Configuration for sentiment analysis
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    use_local_sentiment: bool = True


class TechnicalAnalysisConfig(BaseSettings):
    """Technical analysis configuration."""

    enabled: bool = True

    # Moving averages
    sma_periods: List[int] = Field(
        default_factory=lambda: [
            TechnicalIndicators.SMA_SHORT_PERIOD,
            TechnicalIndicators.SMA_MEDIUM_PERIOD,
            TechnicalIndicators.SMA_LONG_PERIOD,
        ]
    )
    ema_periods: List[int] = Field(
        default_factory=lambda: [
            TechnicalIndicators.EMA_FAST_PERIOD,
            TechnicalIndicators.EMA_SLOW_PERIOD,
        ]
    )

    # RSI
    rsi_period: int = TechnicalIndicators.RSI_PERIOD
    rsi_overbought: int = TechnicalIndicators.RSI_OVERBOUGHT_THRESHOLD
    rsi_oversold: int = TechnicalIndicators.RSI_OVERSOLD_THRESHOLD

    # MACD
    macd_fast: int = TechnicalIndicators.MACD_FAST_PERIOD
    macd_slow: int = TechnicalIndicators.MACD_SLOW_PERIOD
    macd_signal: int = TechnicalIndicators.MACD_SIGNAL_PERIOD

    # Bollinger Bands
    bb_period: int = TechnicalIndicators.BB_PERIOD
    bb_std_dev: int = TechnicalIndicators.BB_STD_DEV


class ReinforcementLearningConfig(BaseSettings):
    """Reinforcement Learning configuration."""

    enabled: bool = True
    training_episodes: int = MLConstants.RL_EPISODES
    auto_train: bool = True

    # RL Parameters
    learning_rate: float = MLConstants.RL_LEARNING_RATE
    epsilon: float = MLConstants.RL_EPSILON
    epsilon_decay: float = MLConstants.RL_EPSILON_DECAY
    epsilon_min: float = MLConstants.RL_EPSILON_MIN
    gamma: float = MLConstants.RL_GAMMA
    memory_size: int = MLConstants.RL_MEMORY_SIZE
    batch_size: int = MLConstants.RL_BATCH_SIZE

    @validator("learning_rate", "epsilon", "epsilon_decay", "epsilon_min", "gamma")
    def validate_rate_params(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Rate parameters must be between 0 and 1")
        return v


class EnhancedRLConfig(BaseSettings):
    """Enhanced Reinforcement Learning configuration."""

    enabled: bool = True
    ensemble_size: int = 3
    training_episodes: int = 200
    use_pytorch: bool = True

    @validator("ensemble_size")
    def validate_ensemble_size(cls, v):
        if v < 1 or v > 10:
            raise ValueError("Ensemble size must be between 1 and 10")
        return v


class HybridModelConfig(BaseSettings):
    """Hybrid model configuration."""

    enabled: bool = True
    weights: Dict[str, float] = Field(
        default_factory=lambda: {"technical": 0.3, "rl": 0.3, "news": 0.4}
    )

    @validator("weights")
    def validate_weights(cls, v):
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        for key, weight in v.items():
            if not 0 <= weight <= 1:
                raise ValueError(
                    f"Weight for '{key}' must be between 0 and 1, got {weight}"
                )

        return v


class MLModelsConfig(BaseSettings):
    """Machine Learning models configuration."""

    technical_analysis: TechnicalAnalysisConfig = Field(
        default_factory=TechnicalAnalysisConfig
    )
    reinforcement_learning: ReinforcementLearningConfig = Field(
        default_factory=ReinforcementLearningConfig
    )
    enhanced_rl: EnhancedRLConfig = Field(default_factory=EnhancedRLConfig)
    hybrid: HybridModelConfig = Field(default_factory=HybridModelConfig)


class PredictionConfig(BaseSettings):
    """Prediction settings configuration."""

    default_days: int = TimeConstants.DEFAULT_PREDICTION_DAYS
    max_days: int = TimeConstants.MAX_PREDICTION_DAYS
    confidence_threshold: float = RiskManagement.MIN_CONFIDENCE_THRESHOLD

    @validator("default_days", "max_days")
    def validate_days(cls, v):
        if v < 1 or v > 30:
            raise ValueError("Prediction days must be between 1 and 30")
        return v

    @validator("confidence_threshold")
    def validate_confidence(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v


class RiskManagementConfig(BaseSettings):
    """Risk management configuration."""

    max_position_size: float = RiskManagement.MAX_POSITION_SIZE
    stop_loss: float = RiskManagement.DEFAULT_STOP_LOSS
    take_profit: float = RiskManagement.DEFAULT_TAKE_PROFIT
    transaction_cost: float = RiskManagement.TRANSACTION_COST
    risk_free_rate: float = RiskManagement.RISK_FREE_RATE

    @validator("max_position_size", "stop_loss", "take_profit", "transaction_cost")
    def validate_percentages(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Percentage values must be between 0 and 1")
        return v


class UIConfig(BaseSettings):
    """User interface configuration."""

    port: int = 8501
    theme: str = "light"
    default_demo_mode: bool = True
    chart_height: int = UIConstants.DEFAULT_CHART_HEIGHT
    chart_width: int = UIConstants.DEFAULT_CHART_WIDTH
    refresh_interval: int = UIConstants.REFRESH_INTERVAL_SECONDS

    @validator("theme")
    def validate_theme(cls, v):
        if v not in ["light", "dark"]:
            raise ValueError("Theme must be 'light' or 'dark'")
        return v

    @validator("port")
    def validate_port(cls, v):
        if not 1000 <= v <= 65535:
            raise ValueError("Port must be between 1000 and 65535")
        return v


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = LoggingConstants.DEFAULT_LOG_LEVEL
    file: str = "logs/stock_advisor.log"
    format: str = LoggingConstants.LOG_FORMAT
    date_format: str = LoggingConstants.DATE_FORMAT
    max_size_mb: int = LoggingConstants.MAX_LOG_SIZE_MB
    max_files: int = LoggingConstants.MAX_LOG_FILES

    @validator("level")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()


class CacheConfig(BaseSettings):
    """Cache configuration."""

    enabled: bool = True
    ttl: int = TimeConstants.CACHE_TTL_SECONDS
    max_size: int = 1000
    redis_url: Optional[str] = Field(None, env="REDIS_URL")

    @validator("ttl")
    def validate_ttl(cls, v):
        if v < 0:
            raise ValueError("TTL must be non-negative")
        return v

    @validator("max_size")
    def validate_max_size(cls, v):
        if v < 1:
            raise ValueError("Max size must be positive")
        return v


class ModelsConfig(BaseSettings):
    """Model storage configuration."""

    directory: str = "models"
    auto_save: bool = True
    versioning: bool = True
    compression: bool = True

    @validator("directory")
    def validate_directory(cls, v):
        # Ensure directory exists
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class StockAdvisorConfig(BaseSettings):
    """Main configuration class for the Stock Advisor application."""

    # Core configurations
    data_sources: DataSourcesConfig = Field(default_factory=DataSourcesConfig)
    news_sources: NewsSourcesConfig = Field(default_factory=NewsSourcesConfig)
    ml_models: MLModelsConfig = Field(default_factory=MLModelsConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    risk_management: RiskManagementConfig = Field(default_factory=RiskManagementConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    # Environment and deployment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @validator("environment")
    def validate_environment(cls, v):
        valid_envs = ["development", "testing", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "StockAdvisorConfig":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            StockAdvisorConfig instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        return cls(**yaml_data)

    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            output_path: Path where to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)

    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a specific service.

        Args:
            service: Service name (e.g., 'alphavantage', 'news_api', 'openai')

        Returns:
            API key if available, None otherwise
        """
        key_mapping = {
            "alphavantage": self.data_sources.alphavantage.api_key,
            "iex": self.data_sources.iex.api_key,
            "news_api": self.news_sources.newsapi.api_key,
            "openai": self.news_sources.openai_api_key,
        }
        return key_mapping.get(service)

    def is_service_enabled(self, service: str) -> bool:
        """
        Check if a service is enabled.

        Args:
            service: Service name

        Returns:
            True if service is enabled, False otherwise
        """
        service_mapping = {
            "yahoo": self.data_sources.yahoo.enabled,
            "alphavantage": self.data_sources.alphavantage.enabled,
            "iex": self.data_sources.iex.enabled,
            "news_api": self.news_sources.newsapi.enabled,
            "rss_feeds": self.news_sources.rss_feeds.enabled,
            "technical_analysis": self.ml_models.technical_analysis.enabled,
            "reinforcement_learning": self.ml_models.reinforcement_learning.enabled,
            "enhanced_rl": self.ml_models.enhanced_rl.enabled,
            "hybrid": self.ml_models.hybrid.enabled,
        }
        return service_mapping.get(service, False)


# Global configuration instance
_config_instance: Optional[StockAdvisorConfig] = None


def get_config(config_path: Optional[Union[str, Path]] = None) -> StockAdvisorConfig:
    """
    Get the global configuration instance.

    Args:
        config_path: Optional path to configuration file

    Returns:
        StockAdvisorConfig instance
    """
    global _config_instance

    if _config_instance is None:
        if config_path:
            _config_instance = StockAdvisorConfig.from_yaml(config_path)
        else:
            # Try to load from default location
            default_config_path = Path("configs/config.yaml")
            if default_config_path.exists():
                _config_instance = StockAdvisorConfig.from_yaml(default_config_path)
            else:
                _config_instance = StockAdvisorConfig()

    return _config_instance


def set_config(config: StockAdvisorConfig) -> None:
    """
    Set the global configuration instance.

    Args:
        config: StockAdvisorConfig instance to set as global
    """
    global _config_instance
    _config_instance = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None
