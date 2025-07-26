"""
Factory classes for creating and configuring Stock Advisor components.

This module provides factory classes that handle dependency injection,
configuration management, and object creation for the Stock Advisor application.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from stock_advisor.config import StockAdvisorConfig, get_config
from stock_advisor.core.stock_data import StockDataFetcher
from stock_advisor.core.news_collector import NewsCollector
from stock_advisor.core.predictor import StockPredictor
from stock_advisor.core.advisor_agent import StockAdvisorAgent
from stock_advisor.types import (
    DataFetcherProtocol,
    NewsCollectorProtocol,
    PredictorProtocol,
)
# ConfigurationError is available but not used in this module currently


class ComponentFactory:
    """
    Factory for creating Stock Advisor components with proper configuration.

    This factory handles dependency injection and ensures all components
    are created with the correct configuration settings.
    """

    def __init__(self, config: Optional[StockAdvisorConfig] = None):
        """
        Initialize the component factory.

        Args:
            config: Configuration object (optional, loads default if None)
        """
        self.config = config or get_config()

    def create_stock_fetcher(self) -> DataFetcherProtocol:
        """
        Create a configured stock data fetcher.

        Returns:
            StockDataFetcher instance with proper configuration
        """
        # TODO: Pass configuration to StockDataFetcher when it's updated
        return StockDataFetcher()

    def create_news_collector(self) -> NewsCollectorProtocol:
        """
        Create a configured news collector.

        Returns:
            NewsCollector instance with proper configuration
        """
        # TODO: Pass configuration to NewsCollector when it's updated
        return NewsCollector()

    def create_predictor(self) -> PredictorProtocol:
        """
        Create a configured predictor.

        Returns:
            StockPredictor instance with proper configuration
        """
        # TODO: Pass configuration to StockPredictor when it's updated
        return StockPredictor()

    def create_advisor_agent(
        self,
        demo_mode: Optional[bool] = None,
        stock_fetcher: Optional[DataFetcherProtocol] = None,
        news_collector: Optional[NewsCollectorProtocol] = None,
        predictor: Optional[PredictorProtocol] = None,
    ) -> StockAdvisorAgent:
        """
        Create a fully configured Stock Advisor Agent.

        Args:
            demo_mode: Whether to use demo mode (uses config default if None)
            stock_fetcher: Custom stock fetcher (creates default if None)
            news_collector: Custom news collector (creates default if None)
            predictor: Custom predictor (creates default if None)

        Returns:
            StockAdvisorAgent instance with all dependencies injected
        """
        # Use config default for demo mode if not specified
        if demo_mode is None:
            demo_mode = self.config.ui.default_demo_mode

        # Create components if not provided
        stock_fetcher = stock_fetcher or self.create_stock_fetcher()
        news_collector = news_collector or self.create_news_collector()
        predictor = predictor or self.create_predictor()

        return StockAdvisorAgent(
            demo_mode=demo_mode,
            stock_fetcher=stock_fetcher,
            news_collector=news_collector,
            predictor=predictor,
        )


class ConfigurableFactory(ComponentFactory):
    """
    Extended factory that supports runtime configuration changes.

    This factory allows for dynamic reconfiguration of components
    without requiring application restart.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the configurable factory.

        Args:
            config_path: Path to configuration file (optional)
        """
        if config_path:
            config = StockAdvisorConfig.from_yaml(config_path)
        else:
            config = get_config()

        super().__init__(config)
        self._config_path = config_path

    def reload_config(self, config_path: Optional[Path] = None) -> None:
        """
        Reload configuration from file.

        Args:
            config_path: Path to configuration file (uses stored path if None)
        """
        config_path = config_path or self._config_path
        if config_path:
            self.config = StockAdvisorConfig.from_yaml(config_path)
        else:
            # Reload default config
            from stock_advisor.config import reset_config

            reset_config()
            self.config = get_config()

    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            config_updates: Dictionary of configuration updates
        """
        # Create new config with updates
        current_config = self.config.dict()

        # Deep merge the updates
        def deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
            for key, value in update_dict.items():
                if (
                    key in base_dict
                    and isinstance(base_dict[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value

        deep_update(current_config, config_updates)

        # Create new config instance
        self.config = StockAdvisorConfig(**current_config)

    def create_test_components(self) -> Dict[str, Any]:
        """
        Create components configured for testing.

        Returns:
            Dictionary containing test-configured components
        """
        # Override config for testing
        test_config_updates = {
            "ui": {"default_demo_mode": True},
            "logging": {"level": "DEBUG"},
            "cache": {"enabled": False},
        }

        original_config = self.config.dict()
        self.update_config(test_config_updates)

        try:
            components = {
                "stock_fetcher": self.create_stock_fetcher(),
                "news_collector": self.create_news_collector(),
                "predictor": self.create_predictor(),
                "advisor_agent": self.create_advisor_agent(demo_mode=True),
            }
            return components
        finally:
            # Restore original config
            self.config = StockAdvisorConfig(**original_config)


class MockFactory(ComponentFactory):
    """
    Factory for creating mock components for testing.

    This factory creates mock implementations of all components
    that can be used for unit testing without external dependencies.
    """

    def __init__(self):
        """Initialize the mock factory with test configuration."""
        # Create minimal test configuration
        super().__init__(StockAdvisorConfig())

    def create_mock_stock_fetcher(self):
        """Create a mock stock data fetcher for testing."""
        try:
            from stock_advisor.tests.mocks import MockStockDataFetcher
            return MockStockDataFetcher()
        except ImportError:
            raise ImportError("Mock classes not available. Run tests to create them.")

    def create_mock_news_collector(self):
        """Create a mock news collector for testing."""
        try:
            from stock_advisor.tests.mocks import MockNewsCollector
            return MockNewsCollector()
        except ImportError:
            raise ImportError("Mock classes not available. Run tests to create them.")

    def create_mock_predictor(self):
        """Create a mock predictor for testing."""
        try:
            from stock_advisor.tests.mocks import MockPredictor
            return MockPredictor()
        except ImportError:
            raise ImportError("Mock classes not available. Run tests to create them.")

    def create_test_advisor_agent(self) -> StockAdvisorAgent:
        """Create an advisor agent with all mock dependencies."""
        return StockAdvisorAgent(
            demo_mode=True,
            stock_fetcher=self.create_mock_stock_fetcher(),
            news_collector=self.create_mock_news_collector(),
            predictor=self.create_mock_predictor(),
        )


# Global factory instances
_default_factory: Optional[ComponentFactory] = None
_configurable_factory: Optional[ConfigurableFactory] = None


def get_factory() -> ComponentFactory:
    """
    Get the default component factory.

    Returns:
        Default ComponentFactory instance
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = ComponentFactory()
    return _default_factory


def get_configurable_factory(config_path: Optional[Path] = None) -> ConfigurableFactory:
    """
    Get the configurable component factory.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigurableFactory instance
    """
    global _configurable_factory
    if _configurable_factory is None:
        _configurable_factory = ConfigurableFactory(config_path)
    return _configurable_factory


def create_advisor_agent(
    demo_mode: bool = False, config_path: Optional[Path] = None
) -> StockAdvisorAgent:
    """
    Convenience function to create a fully configured Stock Advisor Agent.

    Args:
        demo_mode: Whether to use demo mode
        config_path: Path to configuration file

    Returns:
        Configured StockAdvisorAgent instance
    """
    if config_path:
        factory = get_configurable_factory(config_path)
    else:
        factory = get_factory()

    return factory.create_advisor_agent(demo_mode=demo_mode)


def reset_factories() -> None:
    """Reset all global factory instances."""
    global _default_factory, _configurable_factory
    _default_factory = None
    _configurable_factory = None
