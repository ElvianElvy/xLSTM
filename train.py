import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from tqdm import tqdm
import json

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.xlstm import CryptoXLSTM
from utils.metrics import calculate_metrics

def train_model(symbol: str, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001,
               hidden_size: int = 256, num_layers: int = 3, sequence_length: int = 30,
               train_days: int = 365, validation_split: float = 0.2, save_dir: str = "saved_models",
               early_stopping: bool = True, mixed_precision: bool = True):
    """
    Train the xLSTM model on cryptocurrency data.
    
    Args:
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        hidden_size: Hidden size of the LSTM
        num_layers: Number of LSTM layers
        sequence_length: Number of timesteps in each input sequence
        train_days: Number of days of historical data to use for training
        validation_split: Fraction of data to use for validation
        save_dir: Directory to save the trained model
        early_stopping: Whether to use early stopping
        mixed_precision: Whether to use mixed precision training
    
    Returns:
        Trained model and training history
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Configure mixed precision training if available
    scaler = None
    if mixed_precision and torch.cuda.is_available():
        try:
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
            print("Using mixed precision training")
        except ImportError:
            print("Mixed precision training not available, using full precision")
            mixed_precision = False
    
    # Initialize components
    api = BinanceAPI()
    preprocessor = CryptoDataPreprocessor(sequence_length=sequence_length)
    
    print(f"Fetching historical data for {symbol}...")
    df = api.get_training_data(symbol=symbol, days=train_days)
    
    print(f"Processing data...")
    df = preprocessor.process_raw_data(df)
    
    print(f"Preparing sequences...")
    X, T, y = preprocessor.prepare_data(df)
    
    # Split into training and validation sets
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    T_train, T_val = T[:split_idx], T[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, T_train, y_train)
    val_dataset = TensorDataset(X_val, T_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)
    
    # Initialize model
    input_size = X.shape[2]  # Number of features
    model = CryptoXLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3,
        l2_reg=1e-5,
        uncertainty=True
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Define loss function - use negative log likelihood for uncertainty-aware model
    def uncertainty_loss(y_pred, y_true):
        mean, var = y_pred
        # Gaussian negative log likelihood with uncertainty
        nll = 0.5 * torch.log(var) + 0.5 * torch.pow(y_true - mean, 2) / var
        return torch.mean(nll)
    
    # Define optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5, betas=(0.9, 0.999))
    
    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.01
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'learning_rates': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    early_stop_patience = 20
    early_stop_counter = 0
    best_epoch = 0
    
    # Create log file
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{symbol}_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Training loop
    print("\nStarting training...\n")
    start_time = datetime.now()
    
    for epoch in range(epochs):
        epoch_start_time = datetime.now()
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rates'].append(current_lr)
        
        # Training
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for X_batch, T_batch, y_batch in progress_bar:
            # Move tensors to device
            X_batch, T_batch, y_batch = X_batch.to(device), T_batch.to(device), y_batch.to(device)
            
            # Forward pass with mixed precision if enabled
            if mixed_precision and scaler is not None:
                with autocast():
                    outputs = model(X_batch, T_batch)
                    loss = uncertainty_loss(outputs, y_batch)
                    
                    # Add L2 regularization
                    if hasattr(model, 'get_l2_regularization_loss'):
                        reg_loss = model.get_l2_regularization_loss()
                        loss = loss + reg_loss
                
                # Backward pass with gradient scaling
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                outputs = model(X_batch, T_batch)
                loss = uncertainty_loss(outputs, y_batch)
                
                # Add L2 regularization
                if hasattr(model, 'get_l2_regularization_loss'):
                    reg_loss = model.get_l2_regularization_loss()
                    loss = loss + reg_loss
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item(), "lr": current_lr})
        
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []
        all_variances = []  # Store prediction variances
        
        with torch.no_grad():
            for X_batch, T_batch, y_batch in val_loader:
                # Move tensors to device
                X_batch, T_batch, y_batch = X_batch.to(device), T_batch.to(device), y_batch.to(device)
                
                if mixed_precision and scaler is not None:
                    with autocast():
                        outputs = model(X_batch, T_batch)
                        loss = uncertainty_loss(outputs, y_batch)
                else:
                    outputs = model(X_batch, T_batch)
                    loss = uncertainty_loss(outputs, y_batch)
                
                val_loss += loss.item()
                
                # Store predictions and targets for metrics calculation
                mean_pred, var_pred = outputs
                all_predictions.append(mean_pred.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())
                all_variances.append(var_pred.cpu().numpy())
        
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Calculate validation metrics
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        all_variances = np.vstack(all_variances)
        
        # Convert predictions back to original scale
        original_predictions = preprocessor.inverse_transform_predictions(all_predictions)
        original_targets = preprocessor.inverse_transform_predictions(all_targets)
        
        # Calculate metrics
        metrics = calculate_metrics(original_predictions, original_targets)
        history['val_metrics'].append(metrics)
        
        # Calculate average uncertainty (variance)
        avg_variance = np.mean(all_variances)
        
        # Calculate epoch time
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        
        # Print progress
        progress_msg = (
            f"Epoch {epoch+1}/{epochs} "
            f"[{epoch_time:.1f}s] - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
            f"Avg Uncertainty: {avg_variance:.6f}, "
            f"VAL: [MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%, "
            f"Dir. Acc: {metrics['dir_acc']:.2f}%], LR: {current_lr:.6f}"
        )
        print(progress_msg)
        
        # Write to log file
        with open(log_path, 'a') as log_file:
            log_file.write(progress_msg + '\n')
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_epoch = epoch
            
            # Save the best model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(save_dir, f"{symbol}_xlstm_{timestamp}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': {
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'sequence_length': sequence_length,
                    'input_size': input_size
                },
                'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }, model_path)
            print(f"Model saved to {model_path}")
            
            # Save preprocessor
            preprocessor_path = os.path.join(save_dir, f"{symbol}_preprocessor_{timestamp}.pkl")
            import pickle
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            print(f"Preprocessor saved to {preprocessor_path}")
        else:
            early_stop_counter += 1
            if early_stopping and early_stop_counter >= early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Training complete
    total_time = (datetime.now() - start_time).total_seconds() / 60
    print(f"\nTraining completed in {total_time:.2f} minutes.")
    print(f"Best model saved at epoch {best_epoch+1} with validation loss {best_val_loss:.6f}")
    
    # Plot training history
    plt.figure(figsize=(15, 12))
    
    plt.subplot(3, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
    plt.yscale('log')
    plt.legend()
    
    plt.subplot(3, 2, 2)
    plt.plot(history['learning_rates'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.subplot(3, 2, 3)
    plt.plot([m['mae'] for m in history['val_metrics']], label='MAE')
    plt.plot([m['rmse'] for m in history['val_metrics']], label='RMSE')
    plt.title('Validation Error Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    plt.legend()
    
    plt.subplot(3, 2, 4)
    plt.plot([m['mape'] for m in history['val_metrics']], label='MAPE (%)')
    plt.title('Mean Absolute Percentage Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    
    plt.subplot(3, 2, 5)
    plt.plot([m['dir_acc'] for m in history['val_metrics']], label='Directional Accuracy (%)')
    plt.title('Directional Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.axvline(x=best_epoch, color='r', linestyle='--')
    
    # Plot sample predictions vs targets for the last validation batch
    plt.subplot(3, 2, 6)
    idxs = np.random.choice(len(original_predictions), min(10, len(original_predictions)), replace=False)
    for idx in idxs:
        plt.plot(original_predictions[idx, :].reshape(-1), 'b-', alpha=0.5)
        plt.plot(original_targets[idx, :].reshape(-1), 'r-', alpha=0.5)
    plt.title('Sample Predictions (blue) vs Targets (red)')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f"{symbol}_training_history_{timestamp}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Training history plot saved to {plot_path}")
    
    # Save training history as JSON
    history_json = {
        'train_loss': [float(x) for x in history['train_loss']],
        'val_loss': [float(x) for x in history['val_loss']],
        'learning_rates': [float(x) for x in history['learning_rates']],
        'val_metrics': history['val_metrics'],
        'best_epoch': best_epoch,
        'best_val_loss': float(best_val_loss),
        'training_time_minutes': float(total_time),
        'model_config': {
            'symbol': symbol,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'sequence_length': sequence_length,
            'input_size': input_size,
            'train_days': train_days,
            'architecture': 'xLSTM'
        }
    }
    
    history_path = os.path.join(save_dir, f"{symbol}_training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=4)
    
    print(f"Training history saved to {history_path}")
    
    return model, history, preprocessor, model_path, preprocessor_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train cryptocurrency price prediction model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size of LSTM')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--seq_length', type=int, default=30, help='Sequence length')
    parser.add_argument('--train_days', type=int, default=365, help='Days of historical data')
    parser.add_argument('--save_dir', type=str, default='saved_models', help='Directory to save models')
    parser.add_argument('--no_early_stopping', action='store_true', help='Disable early stopping')
    parser.add_argument('--no_mixed_precision', action='store_true', help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    model, history, preprocessor, model_path, preprocessor_path = train_model(
        symbol=args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        sequence_length=args.seq_length,
        train_days=args.train_days,
        save_dir=args.save_dir,
        early_stopping=not args.no_early_stopping,
        mixed_precision=not args.no_mixed_precision
    )
    
    # Option to immediately run prediction after training
    predict_now = input("\nDo you want to run prediction with the trained model? (y/n): ")
    if predict_now.lower() == 'y':
        try:
            from predict import predict_crypto_prices
            predict_crypto_prices(
                model_path=model_path,
                preprocessor_path=preprocessor_path,
                symbol=args.symbol,
                visualize=True,
                output_dir="predictions"
            )
        except ImportError:
            print("Could not import predict.py. Please run prediction separately.")
            print(f"Use: python predict.py --model {model_path} --preprocessor {preprocessor_path} --symbol {args.symbol} --visualize")