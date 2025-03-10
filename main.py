import os
import argparse
import json
import pickle
import time
from datetime import datetime
import sys

import torch
import pandas as pd
import matplotlib.pyplot as plt

from data.binance_api import BinanceAPI
from data.preprocessor import CryptoDataPreprocessor
from models.xlstm import CryptoXLSTM
from train import train_model
from predict import predict_crypto_prices
from config import load_config, update_config, optimize_config


def list_available_cryptos():
    """List available cryptocurrencies on Binance."""
    print("\n" + "="*60)
    print("Fetching available cryptocurrencies from Binance...")
    api = BinanceAPI()
    symbols = api.get_available_symbols()
    
    # Filter for common quote assets
    quote_assets = ["USDT", "BUSD", "BTC", "ETH"]
    filtered_symbols = []
    
    for symbol in symbols:
        for quote in quote_assets:
            if symbol.endswith(quote) and not symbol.startswith("USDT"):
                filtered_symbols.append(symbol)
                break
    
    # Group by quote asset
    grouped = {}
    for symbol in filtered_symbols:
        for quote in quote_assets:
            if symbol.endswith(quote):
                if quote not in grouped:
                    grouped[quote] = []
                grouped[quote].append(symbol)
                break
    
    # Print in organized format
    print(f"\nAvailable cryptocurrencies on Binance ({len(filtered_symbols)} pairs):")
    for quote, symbols in grouped.items():
        print(f"\n{quote} pairs ({len(symbols)}):")
        # Print in multiple columns
        col_width = 12
        cols = 6
        symbols_sorted = sorted(symbols)
        for i in range(0, len(symbols_sorted), cols):
            row = symbols_sorted[i:i+cols]
            print("  ".join(symbol.ljust(col_width) for symbol in row))
    
    print("="*60)
    return filtered_symbols


def find_latest_model(symbol, model_dir="saved_models"):
    """Find the latest trained model for a symbol."""
    if not os.path.exists(model_dir):
        return None, None
    
    # List all files in the directory
    files = os.listdir(model_dir)
    
    # Filter for model files matching the symbol
    model_files = [f for f in files if f.startswith(f"{symbol}_xlstm_") and f.endswith(".pt")]
    preprocessor_files = [f for f in files if f.startswith(f"{symbol}_preprocessor_") and f.endswith(".pkl")]
    
    if not model_files or not preprocessor_files:
        return None, None
    
    # Sort by timestamp in filename
    model_files.sort(reverse=True)
    preprocessor_files.sort(reverse=True)
    
    # Return paths
    model_path = os.path.join(model_dir, model_files[0])
    preprocessor_path = os.path.join(model_dir, preprocessor_files[0])
    
    return model_path, preprocessor_path


def list_available_models(model_dir="saved_models"):
    """List available trained models."""
    if not os.path.exists(model_dir):
        print("No models directory found.")
        return []
    
    # List all files in the directory
    files = os.listdir(model_dir)
    
    # Filter for model files
    model_files = [f for f in files if f.endswith(".pt") and "_xlstm_" in f]
    
    if not model_files:
        print("No trained models found.")
        return []
    
    # Extract symbols and group by symbol
    models_by_symbol = {}
    for model_file in model_files:
        symbol = model_file.split("_xlstm_")[0]
        if symbol not in models_by_symbol:
            models_by_symbol[symbol] = []
        models_by_symbol[symbol].append(model_file)
    
    # Sort models by timestamp (newest first)
    for symbol in models_by_symbol:
        models_by_symbol[symbol].sort(reverse=True)
    
    print("\n" + "="*60)
    print(f"Available trained models ({len(model_files)} total):")
    
    for symbol, models in sorted(models_by_symbol.items()):
        print(f"\n{symbol} ({len(models)} models):")
        for i, model in enumerate(models[:3]):  # Show only the 3 most recent models
            # Extract timestamp and format it
            timestamp = model.split("_xlstm_")[1].split(".pt")[0]
            try:
                date = datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            except:
                date = timestamp
            
            # Try to load model info
            try:
                checkpoint = torch.load(os.path.join(model_dir, model), map_location='cpu')
                epoch = checkpoint.get('epoch', 'Unknown')
                val_loss = checkpoint.get('val_loss', 'Unknown')
                config = checkpoint.get('config', {})
                hidden_size = config.get('hidden_size', 'Unknown')
                num_layers = config.get('num_layers', 'Unknown')
                model_info = f"Epoch: {epoch}, Val Loss: {val_loss:.6f}" if isinstance(val_loss, float) else f"Epoch: {epoch}"
                model_info += f", Size: {hidden_size}x{num_layers}"
            except:
                model_info = "Could not load model info"
            
            print(f"  {i+1}. {date} - {model_info}")
        
        if len(models) > 3:
            print(f"     ... and {len(models)-3} more")
    
    print("="*60)
    return list(models_by_symbol.keys())


def train_new_model(args, config):
    """Train a new model with the specified configuration."""
    print("\n" + "="*60)
    print(f"Training new xLSTM model for {args.symbol}")
    print("="*60)
    
    # Extract training parameters from config
    train_params = config["training"]
    
    # Train the model
    model, history, preprocessor, model_path, preprocessor_path = train_model(
        symbol=args.symbol,
        epochs=train_params["epochs"],
        batch_size=train_params["batch_size"],
        learning_rate=train_params["learning_rate"],
        hidden_size=train_params["hidden_size"],
        num_layers=train_params["num_layers"],
        sequence_length=train_params["sequence_length"],
        train_days=train_params["train_days"],
        validation_split=train_params["validation_split"],
        save_dir=args.model_dir,
        early_stopping=train_params.get("early_stopping", True),
        mixed_precision=train_params.get("mixed_precision", True)
    )
    
    print("\nTraining completed.")
    
    return model_path, preprocessor_path


def predict_prices(args, config, model_path=None, preprocessor_path=None):
    """Predict prices for the specified cryptocurrency."""
    print("\n" + "="*60)
    print(f"Predicting prices for {args.symbol}")
    print("="*60)
    
    # If paths are not provided, find the latest model
    if not model_path or not preprocessor_path:
        model_path, preprocessor_path = find_latest_model(args.symbol, args.model_dir)
    
    # Check if model exists
    if not model_path or not preprocessor_path:
        print(f"No trained model found for {args.symbol}. Please train a model first.")
        return None, None
    
    print(f"Using model: {os.path.basename(model_path)}")
    print(f"Using preprocessor: {os.path.basename(preprocessor_path)}")
    
    # Extract prediction parameters from config
    pred_params = config["prediction"]
    
    # Make predictions
    df_pred, summary = predict_crypto_prices(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        symbol=args.symbol,
        visualize=pred_params["create_visualization"],
        output_dir=args.output_dir
    )
    
    return df_pred, summary


def load_model_info(model_path):
    """Load and display model information."""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Extract model info
        epoch = checkpoint.get('epoch', 'Unknown')
        val_loss = checkpoint.get('val_loss', 'Unknown')
        train_loss = checkpoint.get('train_loss', 'Unknown')
        config = checkpoint.get('config', {})
        hidden_size = config.get('hidden_size', 'Unknown')
        num_layers = config.get('num_layers', 'Unknown')
        sequence_length = config.get('sequence_length', 'Unknown')
        input_size = config.get('input_size', 'Unknown')
        date_trained = checkpoint.get('date', 'Unknown')
        
        print("\n" + "="*60)
        print("Model Information:")
        print("="*60)
        print(f"Model path: {model_path}")
        print(f"Date trained: {date_trained}")
        print(f"Training epochs: {epoch}")
        print(f"Final train loss: {train_loss}")
        print(f"Final validation loss: {val_loss}")
        print(f"Architecture: xLSTM")
        print(f"Hidden size: {hidden_size}")
        print(f"Number of layers: {num_layers}")
        print(f"Sequence length: {sequence_length}")
        print(f"Input features: {input_size}")
        print(f"Total parameters: {calculate_model_parameters(hidden_size, num_layers, input_size):,}")
        print("="*60)
        
        return {
            'epoch': epoch,
            'val_loss': val_loss,
            'train_loss': train_loss,
            'config': config,
            'date_trained': date_trained
        }
    except Exception as e:
        print(f"Error loading model info: {str(e)}")
        return None


def calculate_model_parameters(hidden_size, num_layers, input_size):
    """Estimate the number of parameters in the model."""
    # This is an approximation for xLSTM model
    xlstm_cell_params = 4 * hidden_size * (input_size + hidden_size + 1)  # Standard LSTM params
    xlstm_cell_params += 3 * hidden_size  # Time gate parameters
    xlstm_cell_params += 2 * hidden_size * (input_size + hidden_size + 1)  # Extended memory
    
    # Add parameters for attention and output layers
    attention_params = 8 * hidden_size * hidden_size  # Multi-head attention
    output_params = hidden_size * hidden_size // 2 + hidden_size // 2 * 14  # Output layers
    
    # Total parameters
    total_params = num_layers * xlstm_cell_params + attention_params + output_params
    
    return total_params


def interactive_mode():
    """Run the application in interactive mode."""
    # Load configuration
    config = load_config()
    
    # Initialize Binance API
    api = BinanceAPI()
    
    print("\n" + "="*60)
    print("Cryptocurrency Price Prediction with xLSTM")
    print("="*60 + "\n")
    
    while True:
        print("\nOptions:")
        print("1. List available cryptocurrencies")
        print("2. List trained models")
        print("3. Train a new model")
        print("4. Predict prices")
        print("5. Train and predict")
        print("6. View model information")
        print("7. Optimize configuration")
        print("8. View/edit configuration")
        print("9. Exit")
        
        choice = input("\nEnter your choice (1-9): ")
        
        if choice == "1":
            list_available_cryptos()
        
        elif choice == "2":
            list_available_models(config["paths"]["model_dir"])
        
        elif choice == "3" or choice == "4" or choice == "5":
            # Get symbol
            symbol = input("\nEnter cryptocurrency symbol (e.g., BTCUSDT): ").upper()
            
            # Create parser for argument handling
            parser = argparse.ArgumentParser()
            parser.add_argument("--symbol", type=str, default=symbol)
            parser.add_argument("--model_dir", type=str, default=config["paths"]["model_dir"])
            parser.add_argument("--output_dir", type=str, default=config["paths"]["output_dir"])
            args = parser.parse_args([])
            
            if choice == "3":
                # Train a new model
                try:
                    # Ask for optimization
                    optimize = input("\nDo you want to optimize the configuration for this symbol? (y/n): ").lower()
                    if optimize == 'y':
                        target = input("Optimize for accuracy or speed? (accuracy/speed): ").lower()
                        target = "accuracy" if target != "speed" else "speed"
                        config = optimize_config(symbol, target)
                        print(f"\nConfiguration optimized for {symbol} (target: {target})")
                    
                    model_path, preprocessor_path = train_new_model(args, config)
                    print(f"\nModel trained and saved to {model_path}")
                except Exception as e:
                    print(f"Error training model: {str(e)}")
            
            elif choice == "4":
                # Predict prices
                try:
                    model_path = None
                    preprocessor_path = None
                    
                    # Check if there are multiple models
                    all_models = [f for f in os.listdir(config["paths"]["model_dir"]) 
                                 if f.startswith(f"{symbol}_xlstm_") and f.endswith(".pt")]
                    
                    if len(all_models) > 1:
                        print(f"\nMultiple models found for {symbol}:")
                        for i, model in enumerate(sorted(all_models, reverse=True)):
                            print(f"{i+1}. {model}")
                        
                        model_choice = input("\nSelect a model (number) or press Enter for the latest: ")
                        if model_choice and model_choice.isdigit() and 1 <= int(model_choice) <= len(all_models):
                            model_file = sorted(all_models, reverse=True)[int(model_choice)-1]
                            model_path = os.path.join(config["paths"]["model_dir"], model_file)
                            
                            # Find matching preprocessor
                            timestamp = model_file.split("_xlstm_")[1].split(".pt")[0]
                            preprocessor_file = f"{symbol}_preprocessor_{timestamp}.pkl"
                            preprocessor_path = os.path.join(config["paths"]["model_dir"], preprocessor_file)
                    
                    df_pred, summary = predict_prices(args, config, model_path, preprocessor_path)
                except Exception as e:
                    print(f"Error predicting prices: {str(e)}")
            
            elif choice == "5":
                # Train and predict
                try:
                    # Ask for optimization
                    optimize = input("\nDo you want to optimize the configuration for this symbol? (y/n): ").lower()
                    if optimize == 'y':
                        target = input("Optimize for accuracy or speed? (accuracy/speed): ").lower()
                        target = "accuracy" if target != "speed" else "speed"
                        config = optimize_config(symbol, target)
                        print(f"\nConfiguration optimized for {symbol} (target: {target})")
                    
                    model_path, preprocessor_path = train_new_model(args, config)
                    df_pred, summary = predict_prices(args, config, model_path, preprocessor_path)
                except Exception as e:
                    print(f"Error: {str(e)}")
        
        elif choice == "6":
            # View model information
            model_dir = config["paths"]["model_dir"]
            
            # Get list of available models
            available_symbols = list_available_models(model_dir)
            
            if not available_symbols:
                continue
            
            symbol = input("\nEnter cryptocurrency symbol to view model info: ").upper()
            if symbol not in available_symbols:
                print(f"No models found for {symbol}")
                continue
            
            # Get list of models for the selected symbol
            model_files = [f for f in os.listdir(model_dir) 
                         if f.startswith(f"{symbol}_xlstm_") and f.endswith(".pt")]
            model_files.sort(reverse=True)
            
            print(f"\nAvailable models for {symbol}:")
            for i, model in enumerate(model_files):
                print(f"{i+1}. {model}")
            
            model_choice = input("\nSelect a model (number) or press Enter for the latest: ")
            if model_choice and model_choice.isdigit() and 1 <= int(model_choice) <= len(model_files):
                model_file = model_files[int(model_choice)-1]
            else:
                model_file = model_files[0]
            
            model_path = os.path.join(model_dir, model_file)
            load_model_info(model_path)
        
        elif choice == "7":
            # Optimize configuration
            symbol = input("\nEnter cryptocurrency symbol to optimize for: ").upper()
            target = input("Optimize for accuracy or speed? (accuracy/speed): ").lower()
            target = "accuracy" if target != "speed" else "speed"
            
            try:
                new_config = optimize_config(symbol, target)
                
                # Display the optimized configuration
                print("\nOptimized Configuration:")
                print(json.dumps(new_config["training"], indent=2))
                
                # Ask to save
                save = input("\nDo you want to save this configuration? (y/n): ").lower()
                if save == 'y':
                    update_config(new_config)
                    print("Configuration saved.")
                    config = new_config
            except Exception as e:
                print(f"Error optimizing configuration: {str(e)}")
        
        elif choice == "8":
            # View/edit configuration
            print("\nCurrent Configuration:")
            print(json.dumps(config, indent=2))
            
            edit = input("\nDo you want to edit this configuration? (y/n): ").lower()
            if edit == 'y':
                # Allow editing key parameters
                print("\nEnter new values (press Enter to keep current value):")
                
                # Training parameters
                train = config["training"]
                epochs = input(f"Epochs [{train['epochs']}]: ")
                if epochs and epochs.isdigit():
                    train['epochs'] = int(epochs)
                
                batch_size = input(f"Batch size [{train['batch_size']}]: ")
                if batch_size and batch_size.isdigit():
                    train['batch_size'] = int(batch_size)
                    
                lr = input(f"Learning rate [{train['learning_rate']}]: ")
                if lr and lr.replace('.', '', 1).isdigit():
                    train['learning_rate'] = float(lr)
                
                hidden_size = input(f"Hidden size [{train['hidden_size']}]: ")
                if hidden_size and hidden_size.isdigit():
                    train['hidden_size'] = int(hidden_size)
                
                num_layers = input(f"Number of layers [{train['num_layers']}]: ")
                if num_layers and num_layers.isdigit():
                    train['num_layers'] = int(num_layers)
                
                seq_length = input(f"Sequence length [{train['sequence_length']}]: ")
                if seq_length and seq_length.isdigit():
                    train['sequence_length'] = int(seq_length)
                
                train_days = input(f"Training days [{train['train_days']}]: ")
                if train_days and train_days.isdigit():
                    train['train_days'] = int(train_days)
                
                # Update config
                config["training"] = train
                update_config(config)
                print("Configuration updated.")
        
        elif choice == "9":
            print("\nExiting application. Goodbye!")
            break
        
        else:
            print("\nInvalid choice. Please enter a number from 1 to 9.")


def main():
    """Main entry point for the application."""
    # Load configuration
    config = load_config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction with xLSTM")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--list", action="store_true", help="List available cryptocurrencies")
    parser.add_argument("--models", action="store_true", help="List available trained models")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--predict", action="store_true", help="Predict prices")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Cryptocurrency symbol")
    parser.add_argument("--model_dir", type=str, default=config["paths"]["model_dir"], help="Directory for saved models")
    parser.add_argument("--output_dir", type=str, default=config["paths"]["output_dir"], help="Directory for prediction outputs")
    parser.add_argument("--optimize", type=str, choices=["accuracy", "speed"], help="Optimize configuration for target")
    parser.add_argument("--model_info", action="store_true", help="Display information about the latest model")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run in interactive mode if requested
    if args.interactive:
        interactive_mode()
        return
    
    # List available cryptocurrencies if requested
    if args.list:
        list_available_cryptos()
        return
    
    # List available models if requested
    if args.models:
        list_available_models(args.model_dir)
        return
    
    # Display model info if requested
    if args.model_info:
        model_path, _ = find_latest_model(args.symbol, args.model_dir)
        if model_path:
            load_model_info(model_path)
        else:
            print(f"No models found for {args.symbol}")
        return
    
    # Optimize configuration if requested
    if args.optimize:
        print(f"Optimizing configuration for {args.symbol} (target: {args.optimize})...")
        new_config = optimize_config(args.symbol, args.optimize)
        update_config(new_config)
        print("Configuration optimized and saved.")
        config = new_config
    
    # Train a new model if requested
    if args.train:
        try:
            model_path, preprocessor_path = train_new_model(args, config)
            print(f"\nModel trained and saved to {model_path}")
        except Exception as e:
            print(f"Error training model: {str(e)}")
    
    # Predict prices if requested
    if args.predict:
        try:
            df_pred, summary = predict_prices(args, config)
            if df_pred is None:
                print(f"Could not make predictions for {args.symbol}. Please train a model first.")
        except Exception as e:
            print(f"Error predicting prices: {str(e)}")


if __name__ == "__main__":
    # Set higher recursion limit for complex model
    sys.setrecursionlimit(10000)
    main()