import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from typing import Tuple, List, Dict, Any


class CryptoDataPreprocessor:
    """
    Preprocessor for cryptocurrency price data.
    Prepares data for training and prediction with the Phased LSTM model.
    """
    
    def __init__(self, sequence_length: int = 30):
        """
        Initialize the preprocessor.
        
        Args:
            sequence_length: Number of timesteps to use for each input sequence
        """
        self.sequence_length = sequence_length
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.time_scaler = MinMaxScaler()
    
    def process_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw data from Binance API.
        
        Args:
            df: Raw DataFrame from Binance API
        
        Returns:
            Processed DataFrame with selected features
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Select relevant columns
        df = df[["Open time", "Open", "High", "Low", "Close", "Volume"]]
        
        # Add timestamp column
        df.loc[:, "timestamp"] = df["Open time"].apply(lambda x: x.timestamp())
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Handle missing values
        df = df.dropna()
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with added technical indicators
        """
        # Copy the DataFrame to avoid modifying the original
        df = df.copy()
        
        # Calculate moving averages
        df.loc[:, "MA7"] = df["Close"].rolling(window=7).mean()
        df.loc[:, "MA14"] = df["Close"].rolling(window=14).mean()
        df.loc[:, "MA30"] = df["Close"].rolling(window=30).mean()
        
        # Calculate relative strength index (RSI)
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df.loc[:, "RSI"] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df.loc[:, "MACD"] = ema12 - ema26
        df.loc[:, "MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        
        # Calculate Bollinger Bands
        df.loc[:, "BB_middle"] = df["Close"].rolling(window=20).mean()
        df.loc[:, "BB_std"] = df["Close"].rolling(window=20).std()
        df.loc[:, "BB_upper"] = df["BB_middle"] + (df["BB_std"] * 2)
        df.loc[:, "BB_lower"] = df["BB_middle"] - (df["BB_std"] * 2)
        
        # Daily returns
        df.loc[:, "daily_return"] = df["Close"].pct_change()
        
        # Volatility (standard deviation of returns)
        df.loc[:, "volatility"] = df["daily_return"].rolling(window=7).std()
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare data for model training or prediction.
        
        Args:
            df: Processed DataFrame
        
        Returns:
            Tuple of (X, T, y) tensors
            X: Feature sequences
            T: Time sequences
            y: Target values (open and close prices for 7 days ahead)
        """
        # Select features for model input
        features = ["Open", "High", "Low", "Close", "Volume", 
                   "MA7", "MA14", "MA30", "RSI", "MACD", "MACD_signal",
                   "BB_middle", "BB_std", "daily_return", "volatility"]
        
        # Select data
        data = df[features].values
        timestamps = df["timestamp"].values.reshape(-1, 1)
        
        # Scale the data
        price_cols = [0, 1, 2, 3]  # Open, High, Low, Close
        volume_col = [4]  # Volume
        
        # Fit the scalers if they haven't been fit already
        if not hasattr(self.price_scaler, "data_min_"):
            self.price_scaler.fit(data[:, price_cols])
        if not hasattr(self.volume_scaler, "data_min_"):
            self.volume_scaler.fit(data[:, volume_col])
        if not hasattr(self.time_scaler, "data_min_"):
            self.time_scaler.fit(timestamps)
        
        # Scale the data
        data_scaled = data.copy()
        data_scaled[:, price_cols] = self.price_scaler.transform(data[:, price_cols])
        data_scaled[:, volume_col] = self.volume_scaler.transform(data[:, volume_col])
        timestamps_scaled = self.time_scaler.transform(timestamps)
        
        # Create sequences
        X, T, y = self._create_sequences(data_scaled, timestamps_scaled, df)
        
        return X, T, y
    
    def _create_sequences(self, data: np.ndarray, timestamps: np.ndarray, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create input sequences and target values.
        
        Args:
            data: Scaled feature data
            timestamps: Scaled timestamp data
            df: Original DataFrame for extracting target values
        
        Returns:
            Tuple of (X, T, y) tensors
        """
        X, T, y = [], [], []
        
        # For each possible sequence
        for i in range(len(data) - self.sequence_length - 7):
            # Input sequence
            X.append(data[i:i+self.sequence_length])
            T.append(timestamps[i:i+self.sequence_length])
            
            # Target: open and close prices for the next 7 days
            target = []
            for j in range(1, 8):
                idx = i + self.sequence_length + j - 1
                target.extend([df["Open"].iloc[idx], df["Close"].iloc[idx]])
            
            y.append(target)
        
        # Convert to tensors
        X = torch.tensor(np.array(X), dtype=torch.float32)
        T = torch.tensor(np.array(T), dtype=torch.float32)
        y = torch.tensor(np.array(y), dtype=torch.float32)
        
        return X, T, y
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform the scaled predictions back to original scale.
        
        Args:
            predictions: Predicted values (batch_size, 14)
                        representing open and close prices for 7 days
        
        Returns:
            Original scale predictions
        """
        # Reshape predictions for inverse transform
        batch_size = predictions.shape[0]
        
        # Create an array of the same shape as what the scaler expects
        price_cols = [0, 1, 2, 3]  # Open, High, Low, Close
        dummy_array = np.zeros((batch_size * 7, len(price_cols)))
        
        # Assign values to the dummy array - just open and close
        for i in range(7):
            # Extract open and close from predictions
            open_idx = i * 2  # Even indices are open prices
            close_idx = i * 2 + 1  # Odd indices are close prices
            
            # Assign to the dummy array - open in first column, close in fourth column
            dummy_array[i*batch_size:(i+1)*batch_size, 0] = predictions[:, open_idx]  # Open
            dummy_array[i*batch_size:(i+1)*batch_size, 3] = predictions[:, close_idx]  # Close
            
            # For simplicity, we'll set high equal to max(open, close) and low equal to min(open, close)
            dummy_array[i*batch_size:(i+1)*batch_size, 1] = np.maximum(predictions[:, open_idx], predictions[:, close_idx])  # High
            dummy_array[i*batch_size:(i+1)*batch_size, 2] = np.minimum(predictions[:, open_idx], predictions[:, close_idx])  # Low
        
        # Inverse transform
        original_values = self.price_scaler.inverse_transform(dummy_array)
        
        # Extract just the open and close values and reshape to match the input format
        result = np.zeros((batch_size, 14))
        for i in range(7):
            result[:, i*2] = original_values[i*batch_size:(i+1)*batch_size, 0]      # Open
            result[:, i*2+1] = original_values[i*batch_size:(i+1)*batch_size, 3]    # Close
            
        return result
    
    def prepare_single_prediction(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for a single prediction.
        
        Args:
            df: Processed DataFrame
        
        Returns:
            Tuple of (X, T) tensors for prediction
        """
        # Check if we have enough data
        if len(df) < self.sequence_length:
            raise ValueError(f"Not enough data for prediction. Need at least {self.sequence_length} data points, but got {len(df)}.")
        
        # Get the most recent data for the sequence
        recent_data = df.iloc[-self.sequence_length:].copy()
        
        # Verify recent_data has no NaN values
        if recent_data.isna().any().any():
            columns_with_nan = [col for col in recent_data.columns if recent_data[col].isna().any()]
            raise ValueError(f"NaN values found in columns: {columns_with_nan}. Cannot proceed with prediction.")
            
        # Select features for model input
        features = ["Open", "High", "Low", "Close", "Volume", 
                   "MA7", "MA14", "MA30", "RSI", "MACD", "MACD_signal",
                   "BB_middle", "BB_std", "daily_return", "volatility"]
        
        # Verify all features exist
        missing_features = [f for f in features if f not in recent_data.columns]
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")
        
        # Extract features and timestamps
        data = recent_data[features].values
        timestamps = recent_data["timestamp"].values.reshape(-1, 1)
        
        # Scale the data
        price_cols = [0, 1, 2, 3]  # Open, High, Low, Close
        volume_col = [4]  # Volume
        
        # Check if scalers have been fit
        if not hasattr(self.price_scaler, "data_min_") or not hasattr(self.volume_scaler, "data_min_") or not hasattr(self.time_scaler, "data_min_"):
            raise ValueError("Scalers have not been fit. Train a model first.")
        
        # Scale the data
        data_scaled = data.copy()
        data_scaled[:, price_cols] = self.price_scaler.transform(data[:, price_cols])
        data_scaled[:, volume_col] = self.volume_scaler.transform(data[:, volume_col])
        timestamps_scaled = self.time_scaler.transform(timestamps)
        
        # Create a single sequence
        X = torch.tensor(data_scaled.reshape(1, self.sequence_length, -1), dtype=torch.float32)
        T = torch.tensor(timestamps_scaled.reshape(1, self.sequence_length, -1), dtype=torch.float32)
        
        return X, T