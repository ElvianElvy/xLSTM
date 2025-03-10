import os
import json
import argparse
from typing import Dict, Any
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "paths": {
        "model_dir": "saved_models",
        "output_dir": "predictions",
        "log_dir": "logs"
    },
    "training": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.0005,
        "hidden_size": 256,
        "num_layers": 3,
        "sequence_length": 30,
        "train_days": 365,
        "validation_split": 0.2,
        "early_stopping": True,
        "mixed_precision": True
    },
    "prediction": {
        "create_visualization": True,
        "prediction_days": 7,
        "num_samples": 20  # Number of samples for uncertainty estimation
    },
    "model": {
        "dropout": 0.3,
        "l2_reg": 1e-5,
        "use_attention": True,
        "attention_heads": 8,
        "uncertainty": True
    },
    "binance_api": {
        "use_websocket": True,
        "rate_limit": 1200,  # Maximum requests per minute
        "retry_attempts": 3, # Number of retries for failed API calls
        "timeout": 30        # Timeout in seconds for API calls
    }
}

CONFIG_FILE = "config.json"


def create_default_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Create a default configuration file if it doesn't exist.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    os.makedirs(os.path.dirname(os.path.abspath(config_path)) if os.path.dirname(config_path) else '.', exist_ok=True)
    
    # Write the default config to the file
    with open(config_path, 'w') as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    
    logger.info(f"Created default configuration file: {config_path}")
    return DEFAULT_CONFIG


def load_config(config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Load configuration from file or create default if it doesn't exist.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    # Check if file exists and has content
    if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
        logger.info(f"Configuration file not found or empty. Creating default configuration.")
        return create_default_config(config_path)
    
    # Try to load the config file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check if all required sections are present, add defaults if missing
        for section, defaults in DEFAULT_CONFIG.items():
            if section not in config:
                logger.warning(f"Missing section '{section}' in config. Adding default values.")
                config[section] = defaults
            else:
                # Check if all keys in the section are present
                for key, value in defaults.items():
                    if key not in config[section]:
                        logger.warning(f"Missing key '{key}' in section '{section}'. Adding default value.")
                        config[section][key] = value
        
        return config
    except json.JSONDecodeError as e:
        # If there's an error decoding JSON, create a default config
        logger.error(f"Error parsing {config_path}: {str(e)}. Creating default configuration.")
        return create_default_config(config_path)


def update_config(updates: Dict[str, Any], config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        updates: Dictionary of updates
        config_path: Path to the configuration file
    
    Returns:
        Updated configuration dictionary
    """
    # Load current config
    config = load_config(config_path)
    
    # Recursive function to update nested dictionaries
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                update_dict(d[k], v)
            else:
                d[k] = v
    
    # Update config
    update_dict(config, updates)
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Configuration updated and saved to {config_path}")
    return config


def optimize_config(symbol: str, optimize_for: str = "accuracy") -> Dict[str, Any]:
    """
    Optimize configuration for a specific cryptocurrency.
    
    Args:
        symbol: Trading pair symbol
        optimize_for: Optimization target ("accuracy" or "speed")
    
    Returns:
        Optimized configuration dictionary
    """
    # Load default config
    config = load_config()
    
    logger.info(f"Optimizing configuration for {symbol} (target: {optimize_for})")
    
    # Optimize based on target
    if optimize_for == "accuracy":
        config["training"]["epochs"] = 150
        config["training"]["batch_size"] = 32
        config["training"]["learning_rate"] = 0.0003
        config["training"]["hidden_size"] = 384
        config["training"]["num_layers"] = 3
        config["training"]["sequence_length"] = 45
        config["training"]["train_days"] = 730  # 2 years
        config["model"]["dropout"] = 0.3
        config["model"]["l2_reg"] = 1e-5
        config["model"]["attention_heads"] = 8
    
    elif optimize_for == "speed":
        config["training"]["epochs"] = 50
        config["training"]["batch_size"] = 64
        config["training"]["learning_rate"] = 0.001
        config["training"]["hidden_size"] = 128
        config["training"]["num_layers"] = 2
        config["training"]["sequence_length"] = 20
        config["training"]["train_days"] = 365  # 1 year
        config["model"]["dropout"] = 0.2
        config["model"]["l2_reg"] = 1e-6
        config["model"]["attention_heads"] = 4
    
    # Symbol-specific optimizations based on market characteristics
    if symbol == "BTCUSDT" or symbol == "ETHUSDT":
        # Major cryptocurrencies might benefit from more data and complexity
        config["training"]["train_days"] = 1095  # 3 years
        config["training"]["hidden_size"] = 512 if optimize_for == "accuracy" else 256
        config["training"]["sequence_length"] = 60 if optimize_for == "accuracy" else 30
        config["model"]["attention_heads"] = 12 if optimize_for == "accuracy" else 6
    
    elif symbol.endswith("BTC"):
        # Pairs traded against BTC might have different patterns
        config["training"]["sequence_length"] = 40 if optimize_for == "accuracy" else 25
        config["model"]["dropout"] = 0.35 if optimize_for == "accuracy" else 0.25
    
    elif "USD" in symbol:
        # Stablecoin pairs might need different settings
        config["training"]["learning_rate"] = 0.00025 if optimize_for == "accuracy" else 0.0008
        config["model"]["l2_reg"] = 2e-5 if optimize_for == "accuracy" else 5e-6
    
    logger.info(f"Configuration optimized for {symbol}")
    return config


def export_config(config: Dict[str, Any], export_path: str) -> None:
    """
    Export configuration to a file.
    
    Args:
        config: Configuration dictionary
        export_path: Path to export the configuration
    """
    with open(export_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Configuration exported to {export_path}")


def import_config(import_path: str, config_path: str = CONFIG_FILE) -> Dict[str, Any]:
    """
    Import configuration from a file.
    
    Args:
        import_path: Path to import the configuration from
        config_path: Path to save the imported configuration
    
    Returns:
        Imported configuration dictionary
    """
    try:
        with open(import_path, 'r') as f:
            config = json.load(f)
        
        # Validate imported config
        for section, defaults in DEFAULT_CONFIG.items():
            if section not in config:
                logger.warning(f"Missing section '{section}' in imported config. Adding default values.")
                config[section] = defaults
            else:
                # Check if all keys in the section are present
                for key, value in defaults.items():
                    if key not in config[section]:
                        logger.warning(f"Missing key '{key}' in section '{section}'. Adding default value.")
                        config[section][key] = value
        
        # Save imported config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Configuration imported from {import_path} and saved to {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error importing configuration: {str(e)}")
        return load_config(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cryptocurrency Price Prediction Configuration Manager")
    parser.add_argument("--create", action="store_true", help="Create default configuration file")
    parser.add_argument("--view", action="store_true", help="View current configuration")
    parser.add_argument("--optimize", type=str, help="Optimize configuration for symbol")
    parser.add_argument("--target", type=str, choices=["accuracy", "speed"], default="accuracy",
                        help="Optimization target")
    parser.add_argument("--export", type=str, help="Export configuration to file")
    parser.add_argument("--import", dest="import_path", type=str, help="Import configuration from file")
    
    args = parser.parse_args()
    
    if args.create:
        create_default_config()
        print("Default configuration created.")
    
    if args.view:
        config = load_config()
        print(json.dumps(config, indent=4))
    
    if args.optimize:
        config = optimize_config(args.optimize, args.target)
        print(f"Optimized configuration for {args.optimize} (target: {args.target}):")
        print(json.dumps(config, indent=4))
        
        save = input("\nDo you want to save this configuration? (y/n): ").lower()
        if save == 'y':
            update_config(config)
            print("Configuration saved.")
    
    if args.export:
        config = load_config()
        export_config(config, args.export)
        print(f"Configuration exported to {args.export}")
    
    if args.import_path:
        config = import_config(args.import_path)
        print(f"Configuration imported from {args.import_path}")
        print(json.dumps(config, indent=4))