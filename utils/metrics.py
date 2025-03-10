import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray, detailed: bool = False) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for predictions.
    
    Args:
        predictions: Predicted values (batch_size, 14)
        targets: Target values (batch_size, 14)
        detailed: Whether to include detailed metrics
    
    Returns:
        Dictionary of metrics
    """
    # Ensure predictions and targets have the same shape
    if predictions.shape != targets.shape:
        error_msg = f"Shapes of predictions {predictions.shape} and targets {targets.shape} do not match"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Mean Absolute Error
    mae = mean_absolute_error(targets, predictions)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    
    # Mean Absolute Percentage Error
    # Avoid division by zero
    mask = targets != 0
    if np.any(mask):
        mape = np.mean(np.abs((targets[mask] - predictions[mask]) / targets[mask])) * 100
    else:
        mape = 0.0
    
    # Symmetric Mean Absolute Percentage Error (handles zero and near-zero values better)
    smape = np.mean(2.0 * np.abs(predictions - targets) / 
                   (np.abs(predictions) + np.abs(targets) + 1e-8)) * 100
    
    # R-squared (coefficient of determination)
    r2 = r2_score(targets, predictions)
    
    # Directional Accuracy
    # For each day, check if the model correctly predicts the direction of price movement
    direction_correct = 0
    total_directions = 0
    
    # For each instance in the batch
    for i in range(predictions.shape[0]):
        # For each day in the prediction (7 days)
        for j in range(7):
            # Skip the first day as we need previous day's close
            if j > 0:
                # Index for the previous day's close and current day's close in the flattened array
                prev_day_close_idx = j * 2 - 1  # Previous day's close (odd indices)
                curr_day_close_idx = j * 2 + 1  # Current day's close (odd indices)
                
                # Get predicted and actual direction
                pred_direction = predictions[i, curr_day_close_idx] - predictions[i, prev_day_close_idx]
                true_direction = targets[i, curr_day_close_idx] - targets[i, prev_day_close_idx]
                
                # Check if directions match (both positive or both negative)
                if (pred_direction * true_direction) > 0:
                    direction_correct += 1
                
                total_directions += 1
    
    dir_acc = direction_correct / total_directions * 100 if total_directions > 0 else 0
    
    # Create basic metrics dictionary
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'smape': smape,
        'r2': r2,
        'dir_acc': dir_acc
    }
    
    # Add detailed metrics if requested
    if detailed:
        # Mean Error (to check for bias)
        me = np.mean(predictions - targets)
        
        # Calculate metrics for open prices
        open_indices = np.array([i for i in range(0, 14, 2)])
        mae_open = mean_absolute_error(targets[:, open_indices], predictions[:, open_indices])
        rmse_open = np.sqrt(mean_squared_error(targets[:, open_indices], predictions[:, open_indices]))
        
        # Calculate metrics for close prices
        close_indices = np.array([i for i in range(1, 14, 2)])
        mae_close = mean_absolute_error(targets[:, close_indices], predictions[:, close_indices])
        rmse_close = np.sqrt(mean_squared_error(targets[:, close_indices], predictions[:, close_indices]))
        
        # Calculate metrics by day
        day_metrics = []
        for day in range(7):
            day_indices = np.array([day*2, day*2+1])
            mae_day = mean_absolute_error(targets[:, day_indices], predictions[:, day_indices])
            rmse_day = np.sqrt(mean_squared_error(targets[:, day_indices], predictions[:, day_indices]))
            day_metrics.append({
                'day': day + 1,
                'mae': mae_day,
                'rmse': rmse_day
            })
        
        # Add to metrics dictionary
        metrics.update({
            'mean_error': me,
            'mae_open': mae_open,
            'rmse_open': rmse_open,
            'mae_close': mae_close,
            'rmse_close': rmse_close,
            'day_metrics': day_metrics
        })
        
    return metrics


def calculate_profitability(predictions: np.ndarray, targets: np.ndarray, 
                           initial_capital: float = 1000.0, 
                           transaction_fee: float = 0.001) -> Dict[str, float]:
    """
    Calculate profitability metrics based on a simple trading strategy.
    
    Args:
        predictions: Predicted values (batch_size, 14)
        targets: Target values (batch_size, 14)
        initial_capital: Initial capital for the simulation
        transaction_fee: Fee per transaction as a decimal (e.g., 0.001 for 0.1%)
    
    Returns:
        Dictionary of profitability metrics
    """
    # Reshape predictions and targets to (batch_size, 7, 2) for easier access
    batch_size = predictions.shape[0]
    pred_reshaped = predictions.reshape(batch_size, 7, 2)
    target_reshaped = targets.reshape(batch_size, 7, 2)
    
    # Initialize results arrays
    model_profits = np.zeros(batch_size)
    baseline_profits = np.zeros(batch_size)  # Buy and hold
    perfect_profits = np.zeros(batch_size)   # Perfect knowledge
    
    # For each batch instance (separate trading simulation)
    for i in range(batch_size):
        # Initial capital is the same for all strategies
        model_capital = initial_capital
        baseline_capital = initial_capital
        perfect_capital = initial_capital
        
        # Starting price is the open price of day 1
        start_price = target_reshaped[i, 0, 0]
        
        # Simulation for model predictions
        position = False  # No position to start
        for day in range(7):
            pred_open = pred_reshaped[i, day, 0]
            pred_close = pred_reshaped[i, day, 1]
            actual_open = target_reshaped[i, day, 0]
            actual_close = target_reshaped[i, day, 1]
            
            # Simple strategy: Buy if predicted close > predicted open, sell otherwise
            if pred_close > pred_open and not position:
                # Buy at actual open
                position = True
                shares = model_capital * (1 - transaction_fee) / actual_open
                model_capital = shares * actual_open  # Capital is now in crypto
            elif pred_close < pred_open and position:
                # Sell at actual open
                position = False
                model_capital = shares * actual_open * (1 - transaction_fee)
        
        # Final evaluation - if still holding, sell at the final close
        if position:
            final_close = target_reshaped[i, 6, 1]  # Day 7 close
            model_capital = shares * final_close * (1 - transaction_fee)
        
        # Buy and hold baseline
        shares = baseline_capital * (1 - transaction_fee) / start_price
        final_price = target_reshaped[i, 6, 1]  # Day 7 close
        baseline_capital = shares * final_price * (1 - transaction_fee)
        
        # Perfect knowledge (optimal trades)
        perfect_capital = initial_capital
        position = False
        for day in range(7):
            actual_open = target_reshaped[i, day, 0]
            actual_close = target_reshaped[i, day, 1]
            
            # With perfect knowledge: Buy if close > open
            if actual_close > actual_open and not position:
                position = True
                shares = perfect_capital * (1 - transaction_fee) / actual_open
                perfect_capital = shares * actual_open
            elif actual_close < actual_open and position:
                position = False
                perfect_capital = shares * actual_open * (1 - transaction_fee)
        
        # Final evaluation for perfect knowledge
        if position:
            final_close = target_reshaped[i, 6, 1]
            perfect_capital = shares * final_close * (1 - transaction_fee)
        
        # Calculate profits
        model_profits[i] = model_capital - initial_capital
        baseline_profits[i] = baseline_capital - initial_capital
        perfect_profits[i] = perfect_capital - initial_capital
    
    # Calculate profit metrics
    avg_model_profit = np.mean(model_profits)
    avg_baseline_profit = np.mean(baseline_profits)
    avg_perfect_profit = np.mean(perfect_profits)
    
    # Calculate ROI
    avg_model_roi = (avg_model_profit / initial_capital) * 100
    avg_baseline_roi = (avg_baseline_profit / initial_capital) * 100
    avg_perfect_roi = (avg_perfect_profit / initial_capital) * 100
    
    # Calculate win rate (percentage of positive profits)
    model_win_rate = np.mean(model_profits > 0) * 100
    baseline_win_rate = np.mean(baseline_profits > 0) * 100
    
    # Calculate relative performance metrics
    vs_baseline = (avg_model_roi / avg_baseline_roi) * 100 if avg_baseline_roi != 0 else 0
    vs_perfect = (avg_model_roi / avg_perfect_roi) * 100 if avg_perfect_roi != 0 else 0
    
    return {
        'model_profit': avg_model_profit,
        'model_roi': avg_model_roi,
        'model_win_rate': model_win_rate,
        'baseline_profit': avg_baseline_profit,
        'baseline_roi': avg_baseline_roi,
        'baseline_win_rate': baseline_win_rate,
        'perfect_profit': avg_perfect_profit,
        'perfect_roi': avg_perfect_roi,
        'vs_baseline': vs_baseline,
        'vs_perfect': vs_perfect,
        'initial_capital': initial_capital
    }