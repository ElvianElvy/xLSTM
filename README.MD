# Cryptocurrency Price Prediction with xLSTM

A comprehensive PyTorch-based system for predicting cryptocurrency prices using an advanced xLSTM model. This application fetches data from the Binance WebSocket API, trains a sophisticated deep learning model with uncertainty estimation, and generates daily open and close price predictions with confidence intervals for any cryptocurrency available on Binance for the next 7 days.

## Features

- **Advanced xLSTM Architecture**: Implementation of the extended LSTM model with multi-head attention, hierarchical memory representation, and adaptive gating mechanisms
- **Uncertainty Estimation**: Provides confidence intervals for predictions using probabilistic forecasting techniques
- **Real-time Data**: Fetches data from Binance WebSocket API for the most up-to-date information
- **Multi-feature Analysis**: Incorporates price, volume, and technical indicators for prediction
- **Enhanced Visualization**: Generates intuitive visualizations with uncertainty bands, price trends, and performance metrics
- **Flexible Configuration**: Easily configurable parameters for different trading pairs and strategies
- **Interactive Mode**: User-friendly command-line interface for interacting with the system
- **Easy to Maintain**: Modular code structure designed for a single engineer to maintain

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Internet connection for accessing Binance API

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/crypto-xlstm.git
   cd crypto-xlstm
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Project Structure

```
crypto-xlstm/
├── config.py               # Configuration management
├── main.py                 # Main application entry point
├── predict.py              # Prediction functionality
├── train.py                # Model training functionality
├── requirements.txt        # Project dependencies
├── config.json             # Configuration file
├── data/
│   ├── binance_api.py      # Binance API interface
│   └── preprocessor.py     # Data preprocessing
├── models/
│   └── xlstm.py            # xLSTM model implementation
├── utils/
│   └── metrics.py          # Evaluation metrics
├── saved_models/           # Directory for saved models
└── predictions/            # Directory for prediction outputs
```

## Usage

### Interactive Mode

Run the application in interactive mode:

```bash
python main.py --interactive
```

This will present a menu with options to:
1. List available cryptocurrencies
2. List trained models
3. Train a new model
4. Predict prices
5. Train and predict
6. View model information
7. Optimize configuration
8. View/edit configuration
9. Exit

### Command Line Options

Train a new model:

```bash
python main.py --train --symbol BTCUSDT
```

Make predictions with an existing model:

```bash
python main.py --predict --symbol BTCUSDT
```

List available cryptocurrencies:

```bash
python main.py --list
```

Optimize configuration for a specific cryptocurrency:

```bash
python main.py --optimize accuracy --symbol BTCUSDT
```

View model information:

```bash
python main.py --model_info --symbol BTCUSDT
```

### Configuration

The default configuration is stored in `config.json`. You can modify this file to change parameters like:
- Training epochs, batch size, learning rate
- Model architecture parameters
- Data preprocessing settings
- Output paths

You can also optimize the configuration for specific cryptocurrencies:

```bash
python config.py --optimize BTCUSDT --target accuracy
```

## Model Architecture

The architecture employs an enhanced xLSTM model with several advanced features:

1. **Extended Memory Cells**: Extends standard LSTM with additional memory representations and hierarchical update mechanisms
2. **Multi-Head Temporal Attention**: Focuses on the most important timesteps in the sequence with multiple attention heads
3. **Uncertainty Estimation**: Provides confidence intervals for predictions using probabilistic forecasting
4. **Adaptive Gating Mechanisms**: Dynamic update mechanisms based on time-aware oscillations
5. **Residual Connections**: Allows for better gradient flow during training
6. **Layer Normalization**: Stabilizes training and improves convergence
7. **Advanced Regularization**: Multiple regularization techniques to prevent overfitting

The model takes in multiple features including:
- Price data (Open, High, Low, Close)
- Volume data
- Technical indicators (Moving Averages, RSI, MACD, Bollinger Bands)

## Performance Optimization

The model includes several optimizations to prevent underfitting and overfitting:

1. **Early Stopping**: Prevents overfitting by stopping training when validation loss stops improving
2. **Mixed Precision Training**: Accelerates training on compatible GPUs
3. **Learning Rate Scheduling**: Cosine annealing with warm restarts for better convergence
4. **Gradient Clipping**: Prevents exploding gradients
5. **L2 Regularization**: Penalizes large weights to prevent overfitting
6. **Variational Dropout**: Applied at multiple layers with different rates
7. **Stochastic Depth**: Randomly drops layers during training for better generalization
8. **Attention Mechanisms**: Helps the model focus on relevant parts of the input sequence

## Evaluation Metrics

The system evaluates predictions using multiple metrics:

1. **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual prices
2. **Root Mean Squared Error (RMSE)**: Square root of the average squared differences
3. **Mean Absolute Percentage Error (MAPE)**: Percentage difference between predicted and actual prices
4. **Symmetric MAPE (SMAPE)**: Better handling of near-zero values
5. **Directional Accuracy**: Percentage of correct predictions of price movement direction
6. **R-squared**: Coefficient of determination for regression quality
7. **Uncertainty Calibration**: Evaluates the accuracy of confidence intervals

## Example Output

When making predictions, the system generates:

1. A DataFrame with predicted open and close prices with uncertainty estimates for the next 7 days
2. An enhanced visualization showing historical prices, predictions, and confidence intervals
3. A JSON summary with prediction details, trend analysis, and uncertainty metrics

## Uncertainty Estimation

The xLSTM model provides uncertainty estimates for all predictions:

1. **Confidence Intervals**: 95% confidence bands around predictions
2. **Monte Carlo Dropout**: Multiple stochastic forward passes for uncertainty estimation
3. **Uncertainty Calibration**: Ensures confidence intervals are well-calibrated

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Extended LSTM concept based on research in time series forecasting
- PyTorch team for the excellent deep learning framework
- Binance for providing the WebSocket API