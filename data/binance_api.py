import asyncio
import json
import websockets
import pandas as pd
import time
from datetime import datetime, timedelta
import requests
from typing import List, Dict, Any, Optional, Tuple

class BinanceAPI:
    """
    Interface to the Binance API for fetching cryptocurrency data.
    Supports both REST API for historical data and WebSocket for real-time data.
    """
    
    REST_API_BASE_URL = "https://api.binance.com/api/v3"
    WS_API_BASE_URL = "wss://stream.binance.com:9443/ws"
    
    def __init__(self):
        self.ws = None
    
    async def connect_websocket(self):
        """Connect to Binance WebSocket API."""
        if self.ws is None or self.ws.closed:
            self.ws = await websockets.connect(self.WS_API_BASE_URL)
    
    async def close_websocket(self):
        """Close the WebSocket connection."""
        if self.ws and not self.ws.closed:
            await self.ws.close()
    
    async def subscribe_kline_stream(self, symbol: str, interval: str = "1d"):
        """
        Subscribe to kline/candlestick stream for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1m", "1h", "1d")
        """
        await self.connect_websocket()
        
        # Create subscription message
        stream_name = f"{symbol.lower()}@kline_{interval}"
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": [stream_name],
            "id": 1
        }
        
        # Send subscription message
        await self.ws.send(json.dumps(subscribe_msg))
        
        # Wait for confirmation
        response = await self.ws.recv()
        return json.loads(response)
    
    async def receive_kline_updates(self):
        """Receive kline updates from the WebSocket."""
        if self.ws and not self.ws.closed:
            response = await self.ws.recv()
            return json.loads(response)
        return None
    
    def get_historical_klines(self, symbol: str, interval: str = "1d", 
                             start_time: Optional[int] = None, end_time: Optional[int] = None, 
                             limit: int = 500) -> pd.DataFrame:
        """
        Get historical klines (candlesticks) from Binance REST API.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            interval: Candlestick interval (e.g., "1m", "1h", "1d")
            start_time: Start time in milliseconds
            end_time: End time in milliseconds
            limit: Maximum number of candlesticks to return (max 1000)
        
        Returns:
            DataFrame containing historical price data
        """
        # Construct API endpoint
        endpoint = f"{self.REST_API_BASE_URL}/klines"
        
        # Prepare parameters
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        
        # Add optional parameters if provided
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        # Make the request
        response = requests.get(endpoint, params=params)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Parse the response
        klines = response.json()
        
        # Convert to DataFrame
        columns = [
            "Open time", "Open", "High", "Low", "Close", "Volume",
            "Close time", "Quote asset volume", "Number of trades",
            "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
        ]
        df = pd.DataFrame(klines, columns=columns)
        
        # Convert numeric columns
        numeric_columns = ["Open", "High", "Low", "Close", "Volume",
                          "Quote asset volume", "Taker buy base asset volume", 
                          "Taker buy quote asset volume"]
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        # Convert timestamp columns
        df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")
        df["Close time"] = pd.to_datetime(df["Close time"], unit="ms")
        
        return df
    
    def get_training_data(self, symbol: str, days: int = 365, interval: str = "1d") -> pd.DataFrame:
        """
        Get historical data for training the model.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            days: Number of days of historical data to fetch
            interval: Candlestick interval (e.g., "1m", "1h", "1d")
        
        Returns:
            DataFrame containing historical price data for training
        """
        # Calculate start time (current time - days)
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        # Fetch data
        df = self.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )
        
        # If we need more data than what a single request can provide
        if days > 1000 and interval == "1d":
            # Make multiple requests and concatenate the results
            remaining_days = days - 1000
            while remaining_days > 0:
                end_time = int(df["Open time"].min().timestamp() * 1000) - 1
                days_to_fetch = min(1000, remaining_days)
                start_time = end_time - (days_to_fetch * 24 * 60 * 60 * 1000)
                
                temp_df = self.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=start_time,
                    end_time=end_time,
                    limit=1000
                )
                
                df = pd.concat([temp_df, df])
                remaining_days -= days_to_fetch
        
        # Sort by time
        df = df.sort_values("Open time")
        
        return df
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading pairs on Binance.
        
        Returns:
            List of available trading pair symbols
        """
        endpoint = f"{self.REST_API_BASE_URL}/exchangeInfo"
        response = requests.get(endpoint)
        response.raise_for_status()
        
        exchange_info = response.json()
        symbols = [symbol["symbol"] for symbol in exchange_info["symbols"]]
        
        return symbols