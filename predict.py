import os
import torch
import numpy as np
import pandas as pd
import argparse
import pickle
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.xlstm import CryptoXLSTM

class CryptoPredictor:
    """
    Class for making cryptocurrency price predictions using trained xLSTM models.
    """
    
    def __init__(self, model_path, preprocessor_path):
        """
        Initialize the predictor with a trained model and preprocessor.
        
        Args:
            model_path: Path to the saved model checkpoint
            preprocessor_path: Path to the saved preprocessor
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the preprocessor
        with open(preprocessor_path, 'rb') as f:
            self.preprocessor = pickle.load(f)
        
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration
        config = checkpoint['config']
        
        # Initialize model with uncertainty estimation
        self.model = CryptoXLSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            uncertainty=True  # Enable uncertainty estimation
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize Binance API
        self.api = BinanceAPI()
        
        # Store model metadata
        self.model_info = {
            'epoch': checkpoint.get('epoch', 0),
            'val_loss': checkpoint.get('val_loss', 0),
            'train_loss': checkpoint.get('train_loss', 0),
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layers'],
            'sequence_length': config['sequence_length'],
            'architecture': 'xLSTM',
            'date_trained': checkpoint.get('date', 'Unknown')
        }
    
    def predict(self, symbol, days=30, num_samples=20):
        """
        Make predictions for the given cryptocurrency with uncertainty estimation.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTCUSDT")
            days: Number of days of historical data to use
            num_samples: Number of samples for Monte Carlo dropout uncertainty estimation
        
        Returns:
            DataFrame with predictions and uncertainty measures
        """
        print(f"Fetching historical data for {symbol}...")
        
        # Fetch recent data
        df = self.api.get_historical_klines(
            symbol=symbol,
            interval="1d",
            limit=days + self.preprocessor.sequence_length
        )
        
        if len(df) < self.preprocessor.sequence_length:
            raise ValueError(f"Not enough historical data. Got {len(df)} days, need at least {self.preprocessor.sequence_length}.")
            
        print(f"Processing data...")
        
        # Process the data
        df = self.preprocessor.process_raw_data(df)
        
        # Ensure we have enough data after processing
        if len(df) < self.preprocessor.sequence_length:
            raise ValueError(f"Not enough data after preprocessing. Got {len(df)} valid data points, need {self.preprocessor.sequence_length}.")
            
        # Ensure all features are available (no NaN values in important columns)
        required_features = ["Open", "High", "Low", "Close", "Volume", "MA7", "MA14", "MA30"]
        missing_features = [col for col in required_features if col in df.columns and df[col].isna().any()]
        if missing_features:
            raise ValueError(f"Missing values in features: {missing_features}. Need more historical data.")
        
        print(f"Preparing data for prediction...")
        
        # Prepare data for prediction
        X, T = self.preprocessor.prepare_single_prediction(df)
        
        # Verify tensor shapes
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError(f"Invalid tensor shape: X shape is {X.shape}. Check data preprocessing.")
        
        # Move tensors to device
        X, T = X.to(self.device), T.to(self.device)
        
        print(f"Running prediction model...")
        
        # Make predictions with uncertainty estimation
        with torch.no_grad():
            # Generate multiple samples for uncertainty estimation
            samples, mean_predictions, std_predictions = self.model.sample_predictions(X, T, num_samples)
            
            # Get the standard predictions from the mean
            predictions_np = mean_predictions.cpu().numpy()
            uncertainties_np = std_predictions.cpu().numpy()
        
        # Scale back to original values
        original_predictions = self.preprocessor.inverse_transform_predictions(predictions_np)
        
        # Scale uncertainty estimates (approximate method)
        # We scale uncertainties by the ratio of original to normalized predictions
        scaling_factors = np.divide(
            original_predictions, 
            predictions_np,
            out=np.ones_like(original_predictions),
            where=predictions_np != 0
        )
        original_uncertainties = np.abs(uncertainties_np * scaling_factors)
        
        # Reshape to [open, close] format for predictions and uncertainties
        original_predictions = original_predictions.reshape(-1, 7, 2)
        original_uncertainties = original_uncertainties.reshape(-1, 7, 2)
        
        # Create a DataFrame with predictions and uncertainties
        last_date = df["Open time"].iloc[-1]
        dates = [last_date + timedelta(days=i+1) for i in range(7)]
        
        result = []
        for i, date in enumerate(dates):
            result.append({
                "Date": date,
                "Predicted Open": original_predictions[0, i, 0],
                "Predicted Close": original_predictions[0, i, 1],
                "Open Uncertainty": original_uncertainties[0, i, 0],
                "Close Uncertainty": original_uncertainties[0, i, 1]
            })
        
        # Add confidence intervals
        confidence_df = pd.DataFrame(result)
        confidence_df["Open Lower 95%"] = confidence_df["Predicted Open"] - 1.96 * confidence_df["Open Uncertainty"]
        confidence_df["Open Upper 95%"] = confidence_df["Predicted Open"] + 1.96 * confidence_df["Open Uncertainty"]
        confidence_df["Close Lower 95%"] = confidence_df["Predicted Close"] - 1.96 * confidence_df["Close Uncertainty"]
        confidence_df["Close Upper 95%"] = confidence_df["Predicted Close"] + 1.96 * confidence_df["Close Uncertainty"]
        
        print(f"Prediction completed successfully!")
        
        return confidence_df
    
    def create_visualization(self, df_pred, symbol, output_dir="predictions"):
        """
        Create enhanced visualization of the predictions with uncertainty measures.
        
        Args:
            df_pred: DataFrame with predictions and uncertainty
            symbol: Trading pair symbol
            output_dir: Directory to save the visualization
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Fetching recent historical data for visualization...")
        
        # Fetch recent historical data for context (last 30 days)
        df_hist = self.api.get_historical_klines(
            symbol=symbol,
            interval="1d",
            limit=30
        )
        
        # Select relevant columns and convert to DataFrame format
        df_hist = df_hist[["Open time", "Open", "High", "Low", "Close"]].copy()
        df_hist.loc[:, "Open"] = pd.to_numeric(df_hist["Open"])
        df_hist.loc[:, "High"] = pd.to_numeric(df_hist["High"])
        df_hist.loc[:, "Low"] = pd.to_numeric(df_hist["Low"])
        df_hist.loc[:, "Close"] = pd.to_numeric(df_hist["Close"])
        df_hist.columns = ["Date", "Open", "High", "Low", "Close"]
        
        # Calculate percentage change from last close to predicted values
        last_close = df_hist["Close"].iloc[-1]
        df_pred.loc[:, "Change_From_Last"] = ((df_pred["Predicted Close"] - last_close) / last_close * 100)
        
        print(f"Creating enhanced visualization...")
        
        # Set up style for better visuals
        plt.style.use('dark_background' if plt.get_cmap().name == 'viridis' else 'seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1.5, 1], width_ratios=[4, 1], wspace=0.05, hspace=0.3)
        
        # Main price chart
        ax_main = fig.add_subplot(gs[0, 0])
        
        # Plot historical prices as candlesticks
        for i in range(len(df_hist)):
            # Determine if it's a bullish or bearish candle
            if df_hist['Close'].iloc[i] >= df_hist['Open'].iloc[i]:
                color = '#26a69a'  # Green
                body_bottom = df_hist['Open'].iloc[i]
                body_top = df_hist['Close'].iloc[i]
            else:
                color = '#ef5350'  # Red
                body_bottom = df_hist['Close'].iloc[i]
                body_top = df_hist['Open'].iloc[i]
            
            # Plot the candle body
            rect = plt.Rectangle((i-0.4, body_bottom), 0.8, body_top-body_bottom, 
                                color=color, alpha=0.8, zorder=2)
            ax_main.add_patch(rect)
            
            # Plot the high/low wicks
            ax_main.plot([i, i], [df_hist['Low'].iloc[i], df_hist['High'].iloc[i]], 
                        color='white', linewidth=1, zorder=1)
        
        # Calculate MA for historical data
        df_hist.loc[:, 'MA5'] = df_hist['Close'].rolling(window=5).mean()
        df_hist.loc[:, 'MA10'] = df_hist['Close'].rolling(window=10).mean()
        df_hist.loc[:, 'MA20'] = df_hist['Close'].rolling(window=20).mean()
        
        # Plot moving averages
        x_hist = range(len(df_hist))
        if len(df_hist) >= 5:
            ax_main.plot(x_hist, df_hist['MA5'].values, color='#42a5f5', linewidth=1.5, label='5-day MA')
        if len(df_hist) >= 10:
            ax_main.plot(x_hist, df_hist['MA10'].values, color='#ff9800', linewidth=1.5, label='10-day MA')
        if len(df_hist) >= 20:
            ax_main.plot(x_hist, df_hist['MA20'].values, color='#9c27b0', linewidth=1.5, label='20-day MA')
        
        # Add prediction zone shading with gradient
        hist_len = len(df_hist)
        prediction_zone = np.linspace(0, 1, len(df_pred))
        for i in range(len(df_pred) - 1):
            ax_main.axvspan(hist_len + i - 0.5, hist_len + i + 0.5, color='#2196f3', 
                          alpha=0.05 + 0.05 * prediction_zone[i], zorder=0)
        
        # Draw vertical line to separate historical data and predictions
        ax_main.axvline(x=hist_len-0.5, color="#ffffff", linestyle="--", linewidth=1.5)
        
        # Plot predicted open/close as candlesticks with enhanced design
        for i in range(len(df_pred)):
            idx = hist_len + i
            # Determine if it's a bullish or bearish candle
            if df_pred['Predicted Close'].iloc[i] >= df_pred['Predicted Open'].iloc[i]:
                color = '#26a69a'  # Green
                body_bottom = df_pred['Predicted Open'].iloc[i]
                body_top = df_pred['Predicted Close'].iloc[i]
            else:
                color = '#ef5350'  # Red
                body_bottom = df_pred['Predicted Close'].iloc[i]
                body_top = df_pred['Predicted Open'].iloc[i]
            
            # Plot the candle body with enhanced styling for predictions
            rect = plt.Rectangle((idx-0.4, body_bottom), 0.8, body_top-body_bottom, 
                                color=color, alpha=0.7, hatch='///', zorder=3, linewidth=2, edgecolor='white')
            ax_main.add_patch(rect)
            
            # Plot confidence intervals for close prices
            lower_ci = df_pred['Close Lower 95%'].iloc[i]
            upper_ci = df_pred['Close Upper 95%'].iloc[i]
            ax_main.plot([idx, idx], [lower_ci, upper_ci], color='white', linestyle='-', alpha=0.6, linewidth=2, zorder=2)
            ax_main.plot([idx-0.2, idx+0.2], [lower_ci, lower_ci], color='white', linestyle='-', alpha=0.6, linewidth=2)
            ax_main.plot([idx-0.2, idx+0.2], [upper_ci, upper_ci], color='white', linestyle='-', alpha=0.6, linewidth=2)
            
            # Connect predicted candles with spline curve
            if i > 0:
                prev_idx = hist_len + i - 1
                prev_close = df_pred['Predicted Close'].iloc[i-1]
                curr_open = df_pred['Predicted Open'].iloc[i]
                
                # Create smooth curve between points
                curve_x = np.linspace(prev_idx, idx, 100)
                
                # Apply sine-based smoothing for nice curved connection
                t = np.linspace(0, np.pi, 100)
                smooth_factor = np.sin(t)
                curve_y = prev_close + (curr_open - prev_close) * smooth_factor
                
                ax_main.plot(curve_x, curve_y, color='#2196f3', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Predicted price range
        pred_min = min(df_pred['Close Lower 95%'].min(), df_pred['Open Lower 95%'].min())
        pred_max = max(df_pred['Close Upper 95%'].max(), df_pred['Open Upper 95%'].max())
        hist_min = df_hist[['Open', 'Close', 'Low']].min().min()
        hist_max = df_hist[['Open', 'Close', 'High']].max().max()
        
        # Calculate overall price range and add padding
        y_min = min(hist_min, pred_min)
        y_max = max(hist_max, pred_max)
        y_range = y_max - y_min
        y_min -= y_range * 0.05
        y_max += y_range * 0.05
        
        # Set y-axis limits with padding
        ax_main.set_ylim(y_min, y_max)
        
        # Add price labels on the right for predicted values with improved styling
        for i, row in df_pred.iterrows():
            idx = hist_len + i
            price = row['Predicted Close']
            uncertainty = row['Close Uncertainty']
            change = row['Change_From_Last']
            change_sign = '+' if change >= 0 else ''
            
            # Determine color based on prediction
            if i == 0:  # First prediction compared to last actual close
                color = '#26a69a' if price >= last_close else '#ef5350'
            else:  # Other predictions compared to previous prediction
                color = '#26a69a' if price >= df_pred['Predicted Close'].iloc[i-1] else '#ef5350'
            
            # Add price label with enhanced styling
            ax_main.annotate(f"{price:.2f} ±{uncertainty:.2f} ({change_sign}{change:.2f}%)", 
                          xy=(idx, price),
                          xytext=(idx + 0.1, price),
                          fontsize=10,
                          fontweight='bold',
                          color=color,
                          bbox=dict(boxstyle="round,pad=0.3", fc='black', ec=color, alpha=0.8))
        
        # Add annotations to the last historical point
        ax_main.annotate(f"Last Close: {last_close:.2f}", 
                      xy=(hist_len - 1, last_close),
                      xytext=(hist_len - 1, last_close + y_range * 0.02),
                      fontsize=10,
                      fontweight='bold',
                      color='white',
                      ha='center',
                      bbox=dict(boxstyle="round,pad=0.3", fc='black', ec='white', alpha=0.8))
        
        # Format x-axis with dates
        all_dates = list(df_hist['Date']) + list(df_pred['Date'])
        ax_main.set_xticks(range(len(all_dates)))
        
        # Only show selected dates to avoid crowding
        show_dates = []
        date_labels = []
        
        # Always show first historical date
        show_dates.append(0)
        date_labels.append(all_dates[0].strftime('%Y-%m-%d'))
        
        # Show some dates in the middle
        date_step = max(1, len(df_hist) // 5)
        for i in range(date_step, len(df_hist), date_step):
            show_dates.append(i)
            date_labels.append(all_dates[i].strftime('%Y-%m-%d'))
        
        # Always show last historical date
        if hist_len - 1 not in show_dates:
            show_dates.append(hist_len - 1)
            date_labels.append(all_dates[hist_len - 1].strftime('%Y-%m-%d'))
        
        # Show all prediction dates
        for i in range(len(df_pred)):
            show_dates.append(hist_len + i)
            date_labels.append(all_dates[hist_len + i].strftime('%Y-%m-%d'))
        
        # Sort and apply labels
        show_dates, date_labels = zip(*sorted(zip(show_dates, date_labels)))
        ax_main.set_xticks(show_dates)
        ax_main.set_xticklabels(date_labels, rotation=45, ha='right')
        
        # Side panel for prediction summary
        ax_side = fig.add_subplot(gs[0, 1])
        ax_side.axis('off')
        
        # Create a summary table of predictions with uncertainty
        y_pos = 0.95
        step = 0.12
        
        ax_side.text(0.05, y_pos, "PRICE PREDICTIONS", fontsize=12, fontweight='bold', color='white')
        y_pos -= step
        
        for i, row in df_pred.iterrows():
            date = row['Date'].strftime('%Y-%m-%d')
            open_price = row['Predicted Open']
            open_uncertainty = row['Open Uncertainty']
            close_price = row['Predicted Close']
            close_uncertainty = row['Close Uncertainty']
            change = ((close_price - open_price) / open_price) * 100
            change_sign = '+' if change >= 0 else ''
            
            # Determine color
            color = '#26a69a' if change >= 0 else '#ef5350'
            
            # Draw day label
            ax_side.text(0.05, y_pos, f"Day {i+1}: {date}", fontsize=10, fontweight='bold', color='white')
            y_pos -= step/2
            
            # Draw predicted prices with uncertainty
            ax_side.text(0.1, y_pos, f"Open: {open_price:.2f} ±{open_uncertainty:.2f}", fontsize=9, color='white')
            ax_side.text(0.6, y_pos, f"Close: {close_price:.2f} ±{close_uncertainty:.2f}", fontsize=9, color=color)
            y_pos -= step/2
            
            # Draw daily change
            ax_side.text(0.1, y_pos, f"Daily change: {change_sign}{change:.2f}%", fontsize=9, color=color)
            y_pos -= step/1.5
        
        # Volume subplot
        ax_vol = fig.add_subplot(gs[1, 0], sharex=ax_main)
        
        # Plot historical volume with gradient coloring
        for i in range(len(df_hist)):
            # Color based on price direction
            if df_hist['Close'].iloc[i] >= df_hist['Open'].iloc[i]:
                color = '#26a69a'  # Green
            else:
                color = '#ef5350'  # Red
            
            # Plot volume bars with gradient
            alpha = 0.6 + 0.4 * (i / len(df_hist))
            ax_vol.bar(i, df_hist['Volume'].iloc[i] if 'Volume' in df_hist.columns else 0, 
                      color=color, alpha=alpha, width=0.8)
        
        # Add volume title
        ax_vol.set_ylabel('Volume', fontweight='bold', color='white')
        ax_vol.tick_params(axis='x', labelbottom=False)
        
        # Daily change subplot with enhanced visualization
        ax_change = fig.add_subplot(gs[2, 0])
        
        # Calculate historical daily changes
        if len(df_hist) > 1:
            df_hist.loc[:, 'Daily_Change'] = df_hist['Close'].pct_change() * 100
            
            # Plot historical daily changes with gradient fill
            for i in range(1, len(df_hist)):
                change = df_hist['Daily_Change'].iloc[i]
                color = '#26a69a' if change >= 0 else '#ef5350'
                alpha = 0.5 + 0.5 * (i / len(df_hist))
                ax_change.bar(i, change, color=color, alpha=alpha, width=0.8)
        
        # Plot predicted daily changes with enhanced styling
        for i in range(len(df_pred)):
            idx = hist_len + i
            if i == 0:
                # First day compared to last historical close
                change = ((df_pred['Predicted Close'].iloc[i] - df_hist['Close'].iloc[-1]) / 
                          df_hist['Close'].iloc[-1] * 100)
            else:
                # Other days compared to previous predicted close
                change = ((df_pred['Predicted Close'].iloc[i] - df_pred['Predicted Close'].iloc[i-1]) / 
                          df_pred['Predicted Close'].iloc[i-1] * 100)
            
            color = '#26a69a' if change >= 0 else '#ef5350'
            
            # Plot with hatched pattern for predicted values
            bar = ax_change.bar(idx, change, color=color, alpha=0.7, width=0.8, hatch='///', 
                             edgecolor='white', linewidth=1)
            
            # Add percentage labels with enhanced styling
            ax_change.annotate(f"{change:.2f}%", 
                            xy=(idx, change),
                            xytext=(idx, change + (0.5 if change >= 0 else -0.5)),
                            fontsize=9,
                            fontweight='bold',
                            ha='center',
                            va='bottom' if change >= 0 else 'top',
                            color=color)
        
        # Set daily change y-axis with enhanced styling
        max_change = max(5, 
                         df_hist['Daily_Change'].abs().max() if 'Daily_Change' in df_hist.columns else 0,
                         df_pred['Change_From_Last'].abs().max() if 'Change_From_Last' in df_pred.columns else 0)
        ax_change.set_ylim(-max_change * 1.2, max_change * 1.2)
        ax_change.set_ylabel('Daily Change %', fontweight='bold', color='white')
        
        # Add zero line with styling
        ax_change.axhline(y=0, color='white', linestyle='-', alpha=0.3, linewidth=1)
        
        # Confidence interval chart based on model uncertainty estimates
        ax_conf = fig.add_subplot(gs[3, 0])
        
        # Plot confidence intervals using model-generated uncertainty
        x_pred = range(hist_len, hist_len + len(df_pred))
        base_prices = df_pred['Predicted Close'].values
        lower_bound = df_pred['Close Lower 95%'].values
        upper_bound = df_pred['Close Upper 95%'].values
        
        # Fill confidence band
        ax_conf.fill_between(x_pred, lower_bound, upper_bound, color='#2196f3', alpha=0.3, label='95% Confidence')
        
        # Plot the prediction line
        ax_conf.plot(x_pred, base_prices, 'o-', color='#2196f3', linewidth=2, markersize=6, label='Predicted Close')
        
        # Add labels for uncertainty bounds
        for i in range(len(df_pred)):
            idx = hist_len + i
            ax_conf.plot([idx, idx], 
                      [lower_bound[i], upper_bound[i]], 
                      color='white', linestyle='-', alpha=0.6, linewidth=1)
        
        # Add uncertainty legend
        ax_conf.text(0.02, 0.05, "Uncertainty Analysis:", transform=ax_conf.transAxes, 
                   fontsize=9, fontweight='bold', color='white')
        ax_conf.text(0.02, 0.92, f"xLSTM uncertainty estimates", transform=ax_conf.transAxes, 
                   fontsize=8, color='#2196f3', fontweight='bold')
        
        # Set confidence chart formatting
        ax_conf.set_ylabel('Price with Uncertainty', fontweight='bold', color='white')
        ax_conf.set_xlim(ax_main.get_xlim())
        ax_conf.legend(loc='upper right')
        
        # Set custom ticks for clarity
        x_ticks = range(hist_len, hist_len + len(df_pred))
        ax_conf.set_xticks(x_ticks)
        ax_conf.set_xticklabels([f"Day {i+1}" for i in range(len(df_pred))], rotation=45, ha='right')
        
        # Add labels and title with enhanced styling
        ax_main.set_ylabel('Price', fontweight='bold', color='white')
        ax_main.set_title(f"{symbol} Price Prediction for Next 7 Days", fontsize=16, fontweight='bold', 
                       color='white', pad=20)
        ax_main.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        
        # Add annotations
        prediction_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        overall_change = df_pred['Change_From_Last'].iloc[-1]
        change_text = f"Predicted 7-day change: {'+' if overall_change >= 0 else ''}{overall_change:.2f}%"
        color = '#26a69a' if overall_change >= 0 else '#ef5350'
        fig.text(0.02, 0.02, f"Prediction made on: {prediction_date}\n{change_text}", 
                fontsize=10, color=color, fontweight='bold')
        
        # Add model info
        model_text = f"Model: xLSTM ({self.model_info['hidden_size']} units, {self.model_info['num_layers']} layers)"
        fig.text(0.98, 0.02, model_text, fontsize=10, color='white', 
               fontweight='bold', ha='right')
        
        # Add prediction summary with fancy box styling
        avg_pred = df_pred["Predicted Close"].mean()
        min_pred = df_pred["Predicted Close"].min()
        max_pred = df_pred["Predicted Close"].max()
        avg_uncertainty = df_pred["Close Uncertainty"].mean()
        trend = "Bullish" if df_pred["Predicted Close"].iloc[-1] > df_hist["Close"].iloc[-1] else "Bearish"
        trend_color = '#26a69a' if trend == "Bullish" else '#ef5350'
        
        summary_text = (
            f"PREDICTION SUMMARY\n\n"
            f"Current price: {df_hist['Close'].iloc[-1]:.2f}\n"
            f"Average predicted: {avg_pred:.2f}\n"
            f"Range: {min_pred:.2f} - {max_pred:.2f}\n"
            f"Avg uncertainty: ±{avg_uncertainty:.2f}\n"
            f"7-day outlook: {trend}"
        )
        
        # Add text box for summary with enhanced styling
        props = dict(boxstyle="round,pad=0.5", facecolor='black', 
                    alpha=0.9, edgecolor=trend_color, linewidth=2)
        ax_main.text(0.02, 0.98, summary_text, transform=ax_main.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props, fontweight='bold', color='white')
        
        # Add signature/watermark
        fig.text(0.5, 0.01, f"xLSTM Crypto Prediction System © {datetime.now().year}",
                fontsize=8, color='gray', ha='center', fontweight='bold', fontstyle='italic')
        
        # Adjust layout
        plt.tight_layout()
        
        print(f"Saving visualization...")
        
        # Save the plot with high resolution
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"{symbol}_prediction_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        print(f"Visualization saved to {plot_path}")
        
        return plot_path
    
    def get_prediction_summary(self, df_pred, symbol):
        """
        Generate a detailed summary of the predictions with uncertainty measures.
        
        Args:
            df_pred: DataFrame with predictions
            symbol: Trading pair symbol
        
        Returns:
            Dictionary with prediction summary
        """
        # Calculate price changes
        first_open = df_pred["Predicted Open"].iloc[0]
        last_close = df_pred["Predicted Close"].iloc[-1]
        price_change = last_close - first_open
        price_change_pct = (price_change / first_open) * 100
        
        # Calculate uncertainty statistics
        avg_open_uncertainty = df_pred["Open Uncertainty"].mean()
        avg_close_uncertainty = df_pred["Close Uncertainty"].mean()
        
        # Calculate daily changes
        daily_changes = []
        for i in range(len(df_pred)):
            open_price = df_pred["Predicted Open"].iloc[i]
            open_uncertainty = df_pred["Open Uncertainty"].iloc[i]
            close_price = df_pred["Predicted Close"].iloc[i]
            close_uncertainty = df_pred["Close Uncertainty"].iloc[i]
            
            daily_change = close_price - open_price
            daily_change_pct = (daily_change / open_price) * 100
            
            daily_changes.append({
                "date": df_pred["Date"].iloc[i].strftime("%Y-%m-%d"),
                "open": open_price,
                "open_uncertainty": open_uncertainty,
                "close": close_price,
                "close_uncertainty": close_uncertainty,
                "change": daily_change,
                "change_pct": daily_change_pct
            })
        
        # Check if trend is bullish, bearish, or sideways
        if price_change_pct > 3:
            trend = "Bullish"
            confidence = min(100, 50 + 10 * (price_change_pct / avg_close_uncertainty))
        elif price_change_pct < -3:
            trend = "Bearish"
            confidence = min(100, 50 + 10 * (abs(price_change_pct) / avg_close_uncertainty))
        else:
            trend = "Sideways"
            confidence = max(50 - 10 * (avg_close_uncertainty / abs(price_change_pct + 1e-6)), 30)
        
        # Create detailed summary
        summary = {
            "symbol": symbol,
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction_period": {
                "start": df_pred["Date"].iloc[0].strftime("%Y-%m-%d"),
                "end": df_pred["Date"].iloc[-1].strftime("%Y-%m-%d")
            },
            "overall": {
                "start_price": first_open,
                "end_price": last_close,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "trend": trend,
                "confidence": confidence,
                "avg_uncertainty": {
                    "open": avg_open_uncertainty,
                    "close": avg_close_uncertainty
                }
            },
            "daily_predictions": daily_changes,
            "model_type": "xLSTM",
            "model_metadata": self.model_info
        }
        
        return summary


def predict_crypto_prices(model_path, preprocessor_path, symbol, visualize=True, output_dir="predictions"):
    """
    Predict cryptocurrency prices using a trained model.
    
    Args:
        model_path: Path to the saved model checkpoint
        preprocessor_path: Path to the saved preprocessor
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        visualize: Whether to create visualization
        output_dir: Directory to save the results
    """
    # Create predictor
    print("\n" + "="*50)
    print(f"Initializing xLSTM predictor for {symbol}...")
    predictor = CryptoPredictor(model_path, preprocessor_path)
    
    # Make predictions
    print("\n" + "="*50)
    print(f"Generating predictions for {symbol}...")
    df_pred = predictor.predict(symbol)
    
    # Print predictions
    print("\n" + "="*50)
    print("\nPredictions for", symbol)
    print(df_pred.to_string(index=False))
    
    # Create visualization if requested
    if visualize:
        print("\n" + "="*50)
        print(f"Creating visualization for {symbol}...")
        plot_path = predictor.create_visualization(df_pred, symbol, output_dir)
    
    # Generate summary
    print("\n" + "="*50)
    print(f"Generating prediction summary for {symbol}...")
    summary = predictor.get_prediction_summary(df_pred, symbol)
    
    # Print summary
    print("\n" + "="*50)
    print("\nPrediction Summary:")
    print(f"Symbol: {summary['symbol']}")
    print(f"Prediction Period: {summary['prediction_period']['start']} to {summary['prediction_period']['end']}")
    print(f"Overall Trend: {summary['overall']['trend']} (Confidence: {summary['overall']['confidence']:.1f}%)")
    print(f"Price Change: {summary['overall']['price_change']:.2f} ({summary['overall']['price_change_pct']:.2f}%)")
    print(f"Average Uncertainty: Open ±{summary['overall']['avg_uncertainty']['open']:.2f}, Close ±{summary['overall']['avg_uncertainty']['close']:.2f}")
    
    # Save summary to JSON
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_dir, f"{symbol}_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nSummary saved to {summary_path}")
    print("="*50 + "\n")
    
    return df_pred, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict cryptocurrency prices with xLSTM')
    parser.add_argument('--model', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--preprocessor', type=str, required=True, help='Path to the preprocessor')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory')
    
    args = parser.parse_args()
    
    predict_crypto_prices(
        model_path=args.model,
        preprocessor_path=args.preprocessor,
        symbol=args.symbol,
        visualize=args.visualize,
        output_dir=args.output_dir
    )