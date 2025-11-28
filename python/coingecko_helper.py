# python/coingecko_helper.py
"""
CoinGecko API helper for fetching real cryptocurrency data.
CoinGecko provides free API access without requiring an API key.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging

from python.retry_utils import RetryConfig, make_retriable_request

logger = logging.getLogger(__name__)


class CoinGeckoAPI:
    """
    CoinGecko API connector for free cryptocurrency data.
    
    Features:
    - No API key required
    - Historical OHLCV data
    - Multiple cryptocurrencies
    - Rate limit: 10-50 calls/minute (free tier)
    - Automatic retry with exponential backoff on failures
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Popular cryptocurrency IDs
    COIN_IDS = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'BNB': 'binancecoin',
        'SOL': 'solana',
        'ADA': 'cardano',
        'XRP': 'ripple',
        'DOT': 'polkadot',
        'DOGE': 'dogecoin',
        'AVAX': 'avalanche-2',
        'MATIC': 'matic-network',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'LTC': 'litecoin',
        'BCH': 'bitcoin-cash'
    }
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """
        Initialize CoinGecko API client.
        
        Args:
            retry_config: Optional retry configuration. Defaults to conservative
                         settings suitable for CoinGecko's rate limits.
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        })
        # Use conservative retry config by default for CoinGecko's rate limits
        self.retry_config = retry_config or RetryConfig.conservative()
    
    def _make_request(self, url: str, params: Optional[Dict] = None, timeout: int = 10) -> requests.Response:
        """
        Make an HTTP request with automatic retry on failure.
        
        Args:
            url: URL to request
            params: Optional query parameters
            timeout: Request timeout in seconds
            
        Returns:
            requests.Response object
        """
        return make_retriable_request(
            self.session,
            'GET',
            url,
            config=self.retry_config,
            params=params,
            timeout=timeout,
        )
    
    def ping(self) -> bool:
        """Test API connectivity."""
        try:
            response = self._make_request(f"{self.BASE_URL}/ping", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"CoinGecko ping failed: {e}")
            return False
    
    def get_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko coin ID."""
        symbol_upper = symbol.upper().replace('USDT', '').replace('USD', '')
        return self.COIN_IDS.get(symbol_upper, symbol.lower())
    
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """
        Get current price data for a cryptocurrency.
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        
        Returns:
            Dict with price, market_cap, volume, change_24h
        """
        coin_id = self.get_coin_id(symbol)
        
        try:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true',
                'include_last_updated_at': 'true'
            }
            
            response = self._make_request(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if coin_id in data:
                coin_data = data[coin_id]
                return {
                    'symbol': symbol,
                    'price': coin_data.get('usd', 0),
                    'market_cap': coin_data.get('usd_market_cap', 0),
                    'volume_24h': coin_data.get('usd_24h_vol', 0),
                    'change_24h': coin_data.get('usd_24h_change', 0),
                    'last_updated': datetime.fromtimestamp(coin_data.get('last_updated_at', 0))
                }
            else:
                logger.warning(f"No data for {symbol} (coin_id: {coin_id})")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch price for {symbol}: {e}")
            return None
    
    def get_historical_ohlc(
        self,
        symbol: str,
        days: int = 30,
        vs_currency: str = 'usd'
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLC data.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
            vs_currency: Quote currency (default: 'usd')
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close
        """
        coin_id = self.get_coin_id(symbol)
        
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': vs_currency,
                'days': days
            }
            
            response = self._make_request(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.warning(f"No OHLC data for {symbol}")
                return None
            
            # Parse data: [timestamp, open, high, low, close]
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Add volume estimate (not provided by free tier)
            df['volume'] = 0  # Placeholder
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLC for {symbol}: {e}")
            return None
    
    def get_market_chart(
        self,
        symbol: str,
        days: int = 30,
        vs_currency: str = 'usd',
        interval: str = 'daily'
    ) -> Optional[pd.DataFrame]:
        """
        Get market chart data with prices and volumes.
        
        Args:
            symbol: Cryptocurrency symbol
            days: Number of days (1-365 for hourly, 1+ for daily)
            vs_currency: Quote currency
            interval: 'daily' or will auto-select based on days
        
        Returns:
            DataFrame with price and volume data
        """
        coin_id = self.get_coin_id(symbol)
        
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': interval if days > 90 else 'hourly'
            }
            
            response = self._make_request(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            if 'prices' not in data:
                logger.warning(f"No market chart data for {symbol}")
                return None
            
            # Parse prices
            prices_df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
            prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'], unit='ms')
            
            # Parse volumes
            if 'total_volumes' in data:
                volumes_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
                volumes_df['timestamp'] = pd.to_datetime(volumes_df['timestamp'], unit='ms')
                
                # Merge
                df = pd.merge(prices_df, volumes_df, on='timestamp', how='left')
            else:
                df = prices_df
                df['volume'] = 0
            
            df.set_index('timestamp', inplace=True)
            
            # Calculate OHLC from close prices (approximation)
            # For better accuracy, use get_historical_ohlc
            df['open'] = df['close'].shift(1)
            df['high'] = df[['open', 'close']].max(axis=1) * 1.002  # Approximate
            df['low'] = df[['open', 'close']].min(axis=1) * 0.998   # Approximate
            df['open'] = df['open'].fillna(df['close'])
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Failed to fetch market chart for {symbol}: {e}")
            return None
    
    def get_multiple_prices(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get current prices for multiple cryptocurrencies.
        
        Args:
            symbols: List of cryptocurrency symbols
        
        Returns:
            Dict mapping symbol to price data
        """
        coin_ids = [self.get_coin_id(s) for s in symbols]
        
        try:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                'ids': ','.join(coin_ids),
                'vs_currencies': 'usd',
                'include_24hr_change': 'true',
                'include_24hr_vol': 'true',
                'include_market_cap': 'true'
            }
            
            response = self._make_request(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = {}
            for symbol, coin_id in zip(symbols, coin_ids):
                if coin_id in data:
                    coin_data = data[coin_id]
                    results[symbol] = {
                        'price': coin_data.get('usd', 0),
                        'market_cap': coin_data.get('usd_market_cap', 0),
                        'volume_24h': coin_data.get('usd_24h_vol', 0),
                        'change_24h': coin_data.get('usd_24h_change', 0)
                    }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch multiple prices: {e}")
            return {}
    
    def search_coins(self, query: str) -> List[Dict]:
        """
        Search for cryptocurrencies.
        
        Args:
            query: Search query
        
        Returns:
            List of matching coins with id, symbol, name
        """
        try:
            url = f"{self.BASE_URL}/search"
            params = {'query': query}
            
            response = self._make_request(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'coins' in data:
                return [
                    {
                        'id': coin['id'],
                        'symbol': coin['symbol'].upper(),
                        'name': coin['name']
                    }
                    for coin in data['coins'][:10]  # Top 10 results
                ]
            
            return []
            
        except Exception as e:
            logger.error(f"Search failed for '{query}': {e}")
            return []


# Convenience functions
def fetch_crypto_data(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    """
    Fetch cryptocurrency data using CoinGecko.
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH')
        days: Number of days of historical data
    
    Returns:
        DataFrame with OHLCV data
    """
    api = CoinGeckoAPI()
    
    # Try OHLC first (more accurate)
    df = api.get_historical_ohlc(symbol, days)
    
    if df is None or len(df) < 10:
        # Fallback to market chart
        df = api.get_market_chart(symbol, days)
    
    if df is not None:
        # Add returns
        df['returns'] = df['close'].pct_change()
    
    return df


def get_available_cryptos() -> List[str]:
    """Get list of available cryptocurrency symbols."""
    return list(CoinGeckoAPI.COIN_IDS.keys())


def test_coingecko_connection() -> bool:
    """Test CoinGecko API connection."""
    api = CoinGeckoAPI()
    return api.ping()
