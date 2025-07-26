import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import time
import random
import requests
import os
from dotenv import load_dotenv

from stock_advisor.constants import (
    TimeConstants,
    DataSources,
    ValidationRules,
    ErrorMessages,
)
from stock_advisor.exceptions import (
    DataFetchError,
    InvalidSymbolError,
    InsufficientDataError,
    APIError,
    NetworkError,
)

load_dotenv()


class StockDataFetcher:
    def __init__(self) -> None:
        self.cache: Dict[str, pd.DataFrame] = {}
        self.alpha_vantage_key: Optional[str] = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.use_alpha_vantage: bool = bool(self.alpha_vantage_key)
        self.iex_cloud_key: Optional[str] = os.getenv("IEX_CLOUD_API_KEY")
        self.use_iex_cloud: bool = bool(self.iex_cloud_key)

    def get_stock_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """
        Fetch stock data for a given symbol

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')

        Returns:
            DataFrame with stock data including OHLCV
        """
        cache_key = f"{symbol}_{period}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Try Alpha Vantage first if API key is available and working
        if self.use_alpha_vantage and self.alpha_vantage_key:
            print(f"Trying Alpha Vantage for {symbol}...")
            try:
                data = self.get_stock_data_alpha_vantage(symbol, period)
                if not data.empty:
                    self.cache[cache_key] = data
                    return data
            except Exception as av_error:
                print(f"Alpha Vantage error: {av_error}")
                # Don't use Alpha Vantage for this session
                self.use_alpha_vantage = False

        # Yahoo Finance with enhanced error handling and robustness
        print(f"Trying Yahoo Finance for {symbol}...")

        # First, validate the symbol
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Check if ticker is valid (has basic info)
            if not info or len(info) < 3:
                print(f"Invalid ticker symbol: {symbol}")
                return self._try_alternative_yahoo_methods(symbol, period, cache_key)

        except Exception as validation_error:
            print(f"Symbol validation failed: {validation_error}")
            return self._try_alternative_yahoo_methods(symbol, period, cache_key)

        # Main Yahoo Finance fetch with multiple strategies
        max_retries = 4
        for attempt in range(max_retries):
            try:
                # Progressive delay to avoid rate limiting
                if attempt > 0:
                    delay = min(2**attempt + random.uniform(0, 1), 10)
                    print(f"Waiting {delay:.1f} seconds before retry {attempt + 1}...")
                    time.sleep(delay)

                # Strategy 1: Try original period
                data = self._fetch_yahoo_data_with_period(ticker, period, symbol)
                if not data.empty:
                    self.cache[cache_key] = data
                    return data

                # Strategy 2: Try alternative period formats
                data = self._fetch_yahoo_data_with_alternative_periods(
                    ticker, period, symbol
                )
                if not data.empty:
                    self.cache[cache_key] = data
                    return data

                # Strategy 3: Try with date ranges
                data = self._fetch_yahoo_data_with_dates(ticker, period, symbol)
                if not data.empty:
                    self.cache[cache_key] = data
                    return data

                print(f"No data returned for {symbol} (attempt {attempt + 1})")

            except Exception as e:
                error_msg = str(e).lower()
                print(f"Yahoo Finance error for {symbol} (attempt {attempt + 1}): {e}")

                # Handle specific error types
                if "delisted" in error_msg or "not found" in error_msg:
                    print(f"Symbol {symbol} appears to be delisted or invalid")
                    break
                elif "rate limit" in error_msg or "429" in error_msg:
                    print("Rate limit detected, increasing delay...")
                    time.sleep(5 + attempt * 2)
                elif "connection" in error_msg or "timeout" in error_msg:
                    print("Connection issue, retrying with longer timeout...")
                    time.sleep(3 + attempt)

        # Final comprehensive fallback attempts
        data = self._try_alternative_yahoo_methods(symbol, period, cache_key)
        if not data.empty:
            return data

        # Try IEX Cloud as final fallback
        if self.use_iex_cloud:
            print(f"Trying IEX Cloud for {symbol}...")
            try:
                data = self.get_stock_data_iex_cloud(symbol, period)
                if not data.empty:
                    self.cache[cache_key] = data
                    return data
            except Exception as iex_error:
                print(f"IEX Cloud error: {iex_error}")

        print(f"All data sources failed for {symbol}. No data available.")
        return pd.DataFrame()

    def get_stock_data_alpha_vantage(
        self, symbol: str, period: str = "1mo"
    ) -> pd.DataFrame:
        """
        Fetch stock data using Alpha Vantage API
        """
        if not self.alpha_vantage_key:
            return pd.DataFrame()

        # Map period to Alpha Vantage function
        if period in ["1d", "5d"]:
            function = "TIME_SERIES_INTRADAY"
            interval = "60min"
        else:
            function = "TIME_SERIES_DAILY"
            interval = None

        url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
            "outputsize": "full",
        }

        if interval:
            params["interval"] = interval

        try:
            response = requests.get(url, params=params)
            data = response.json()

            # Handle different response formats
            if function == "TIME_SERIES_INTRADAY":
                time_series = data.get(f"Time Series ({interval})", {})
            else:
                time_series = data.get("Time Series (Daily)", {})

            if not time_series:
                print(
                    f"No Alpha Vantage data for {symbol}: {data.get('Note', 'Unknown error')}"
                )
                return pd.DataFrame()

            # Convert to DataFrame
            df_data = []
            for date_str, values in time_series.items():
                df_data.append(
                    {
                        "Date": pd.to_datetime(date_str),
                        "Open": float(values["1. open"]),
                        "High": float(values["2. high"]),
                        "Low": float(values["3. low"]),
                        "Close": float(values["4. close"]),
                        "Volume": int(values["5. volume"]),
                    }
                )

            df = pd.DataFrame(df_data)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

            # Filter by period
            if period == "1mo":
                cutoff_date = datetime.now() - timedelta(days=30)
            elif period == "3mo":
                cutoff_date = datetime.now() - timedelta(days=90)
            elif period == "6mo":
                cutoff_date = datetime.now() - timedelta(days=180)
            elif period == "1y":
                cutoff_date = datetime.now() - timedelta(days=365)
            else:
                cutoff_date = datetime.now() - timedelta(days=30)

            df = df[df.index >= cutoff_date]

            return df

        except Exception as e:
            print(f"Alpha Vantage error for {symbol}: {e}")
            return pd.DataFrame()

    def get_stock_data_iex_cloud(
        self, symbol: str, period: str = "1mo"
    ) -> pd.DataFrame:
        """
        Fetch stock data using IEX Cloud API
        500,000 free calls per month
        """
        if not self.iex_cloud_key:
            return pd.DataFrame()

        # Map period to IEX Cloud format
        period_mapping = {
            "1d": "1d",
            "5d": "5d",
            "1mo": "1m",
            "3mo": "3m",
            "6mo": "6m",
            "1y": "1y",
            "2y": "2y",
            "5y": "5y",
        }

        iex_period = period_mapping.get(period, "3m")

        # IEX Cloud API endpoint
        url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/{iex_period}"
        params = {"token": self.iex_cloud_key, "chartCloseOnly": "false"}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if not data:
                print(f"No IEX Cloud data for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df_data = []
            for item in data:
                try:
                    df_data.append(
                        {
                            "Date": pd.to_datetime(item["date"]),
                            "Open": float(item.get("open", 0)),
                            "High": float(item.get("high", 0)),
                            "Low": float(item.get("low", 0)),
                            "Close": float(item.get("close", 0)),
                            "Volume": int(item.get("volume", 0)),
                        }
                    )
                except (ValueError, TypeError, KeyError) as e:
                    print(f"Error parsing IEX data point: {e}")
                    continue

            if not df_data:
                print(f"No valid IEX Cloud data points for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(df_data)
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)

            print(
                f"Successfully fetched {len(df)} days of data for {symbol} from IEX Cloud"
            )
            return df

        except requests.exceptions.RequestException as e:
            print(f"IEX Cloud API error for {symbol}: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing IEX Cloud data for {symbol}: {e}")
            return pd.DataFrame()

    def get_real_time_quote_iex(self, symbol: str) -> Dict:
        """
        Get real-time quote from IEX Cloud
        """
        if not self.iex_cloud_key:
            return {}

        url = f"https://cloud.iexapis.com/stable/stock/{symbol}/quote"
        params = {"token": self.iex_cloud_key}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            return {
                "symbol": data.get("symbol"),
                "current_price": data.get("latestPrice"),
                "change": data.get("change"),
                "change_percent": data.get("changePercent", 0) * 100,
                "volume": data.get("latestVolume"),
                "market_cap": data.get("marketCap"),
                "pe_ratio": data.get("peRatio"),
                "source": "IEX Cloud",
            }
        except Exception as e:
            print(f"Error getting IEX Cloud quote for {symbol}: {e}")
            return {}

    def _fetch_yahoo_data_with_period(
        self, ticker, period: str, symbol: str
    ) -> pd.DataFrame:
        """Fetch Yahoo Finance data with original period"""
        try:
            data = ticker.history(period=period, timeout=30)
            if not data.empty:
                print(
                    f"✅ Yahoo Finance: {len(data)} days for {symbol} (period: {period})"
                )
                return data
        except Exception as e:
            print(f"Period fetch failed: {e}")
        return pd.DataFrame()

    def _fetch_yahoo_data_with_alternative_periods(
        self, ticker, period: str, symbol: str
    ) -> pd.DataFrame:
        """Try alternative period formats"""
        period_alternatives = {
            "1mo": ["1mo", "30d", "1M", "4w"],
            "3mo": ["3mo", "90d", "3M", "12w"],
            "6mo": ["6mo", "180d", "6M", "26w"],
            "1y": ["1y", "365d", "1Y", "12mo"],
            "2y": ["2y", "730d", "2Y", "24mo"],
            "5y": ["5y", "5Y"],
        }

        alternatives = period_alternatives.get(period, [period])

        for alt_period in alternatives[1:]:  # Skip first (already tried)
            try:
                data = ticker.history(period=alt_period, timeout=30)
                if not data.empty:
                    print(
                        f"✅ Yahoo Finance: {len(data)} days for {symbol} (alt period: {alt_period})"
                    )
                    return data
            except Exception as e:
                print(f"Alternative period {alt_period} failed: {e}")
                continue

        return pd.DataFrame()

    def _fetch_yahoo_data_with_dates(
        self, ticker, period: str, symbol: str
    ) -> pd.DataFrame:
        """Try with explicit start/end dates"""
        try:
            from datetime import datetime, timedelta

            end_date = datetime.now()

            # Map periods to days with buffer
            period_days = {
                "1d": 2,
                "5d": 7,
                "1mo": 35,
                "3mo": 95,
                "6mo": 185,
                "1y": 370,
                "2y": 735,
                "5y": 1830,
            }

            days = period_days.get(period, 35)
            start_date = end_date - timedelta(days=days)

            data = ticker.history(start=start_date, end=end_date, timeout=30)
            if not data.empty:
                print(f"✅ Yahoo Finance: {len(data)} days for {symbol} (date range)")
                return data

        except Exception as e:
            print(f"Date range fetch failed: {e}")

        return pd.DataFrame()

    def _try_alternative_yahoo_methods(
        self, symbol: str, period: str, cache_key: str
    ) -> pd.DataFrame:
        """Comprehensive fallback methods for Yahoo Finance"""
        print(f"Trying alternative Yahoo Finance methods for {symbol}...")

        # Method 1: Different session/headers
        try:
            import yfinance as yf

            # Create new ticker instance with different settings
            ticker = yf.Ticker(symbol)

            # Try with minimal period first
            for minimal_period in ["5d", "1mo", "3mo"]:
                try:
                    data = ticker.history(
                        period=minimal_period, timeout=45, threads=False, progress=False
                    )
                    if not data.empty:
                        print(f"✅ Alternative method: {len(data)} days for {symbol}")
                        # Filter to requested period if we got more data
                        if period != minimal_period:
                            data = self._filter_data_to_period(data, period)
                        self.cache[cache_key] = data
                        return data
                except Exception as e:
                    print(f"Minimal period {minimal_period} failed: {e}")
                    continue

        except Exception as e:
            print(f"Alternative session method failed: {e}")

        # Method 2: Try with different intervals
        try:
            ticker = yf.Ticker(symbol)

            # Try daily data with shorter period first, then extend
            short_data = ticker.history(period="5d", interval="1d", timeout=30)
            if not short_data.empty:
                print(f"Got short data, trying to extend...")
                try:
                    # Try to get more data
                    extended_data = ticker.history(
                        period="2mo", interval="1d", timeout=45
                    )
                    if len(extended_data) > len(short_data):
                        data = self._filter_data_to_period(extended_data, period)
                        print(f"✅ Extended data: {len(data)} days for {symbol}")
                        self.cache[cache_key] = data
                        return data
                except:
                    pass

                # Use short data if extension failed
                print(f"✅ Using short data: {len(short_data)} days for {symbol}")
                self.cache[cache_key] = short_data
                return short_data

        except Exception as e:
            print(f"Interval method failed: {e}")

        # Method 3: Try to get any available data and check info
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if info and info.get("regularMarketPrice"):
                print(
                    f"Ticker exists (price: ${info.get('regularMarketPrice')}), trying basic fetch..."
                )

                # Try the most basic fetch possible
                data = ticker.history(
                    period="1mo", auto_adjust=True, back_adjust=False, timeout=60
                )
                if not data.empty:
                    print(f"✅ Basic fetch: {len(data)} days for {symbol}")
                    self.cache[cache_key] = data
                    return data

        except Exception as e:
            print(f"Basic info method failed: {e}")

        print(f"❌ All Yahoo Finance methods failed for {symbol}")
        return pd.DataFrame()

    def _filter_data_to_period(
        self, data: pd.DataFrame, target_period: str
    ) -> pd.DataFrame:
        """Filter data to match target period"""
        if data.empty:
            return data

        period_days = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
        }

        target_days = period_days.get(target_period, 90)

        if len(data) > target_days:
            return data.tail(target_days)

        return data

    def get_multiple_stocks(
        self, symbols: List[str], period: str = "1mo"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks

        Args:
            symbols: List of stock symbols
            period: Time period

        Returns:
            Dictionary mapping symbols to their DataFrames
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.get_stock_data(symbol, period)
        return data

    def get_stock_info(self, symbol: str) -> Dict:
        """
        Get basic information about a stock

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock information
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Add delay to avoid rate limiting
                time.sleep(random.uniform(1, 2))

                ticker = yf.Ticker(symbol)
                info = ticker.info

                if info:
                    return {
                        "symbol": symbol,
                        "company_name": info.get("longName", "N/A"),
                        "sector": info.get("sector", "N/A"),
                        "industry": info.get("industry", "N/A"),
                        "market_cap": info.get("marketCap", 0),
                        "current_price": info.get("currentPrice", 0),
                        "previous_close": info.get("previousClose", 0),
                        "volume": info.get("volume", 0),
                        "avg_volume": info.get("averageVolume", 0),
                    }

            except Exception as e:
                print(f"Error fetching info for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)

        # Return basic info if API calls fail
        return {
            "symbol": symbol,
            "company_name": "N/A",
            "sector": "N/A",
            "industry": "N/A",
            "market_cap": 0,
            "current_price": 0,
            "previous_close": 0,
            "volume": 0,
            "avg_volume": 0,
        }

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators

        Args:
            data: Stock data DataFrame

        Returns:
            DataFrame with added technical indicators
        """
        if data.empty:
            return data

        df = data.copy()

        # Simple Moving Averages
        df["SMA_5"] = df["Close"].rolling(window=5).mean()
        df["SMA_10"] = df["Close"].rolling(window=10).mean()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()

        # Exponential Moving Average
        df["EMA_12"] = df["Close"].ewm(span=12).mean()
        df["EMA_26"] = df["Close"].ewm(span=26).mean()

        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

        # RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_Middle"] = df["Close"].rolling(window=20).mean()
        bb_std = df["Close"].rolling(window=20).std()
        df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
        df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

        return df
