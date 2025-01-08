# Bitcoin Price Prediction

This repository contains a machine learning model for predicting Bitcoin prices using LSTM (Long Short-Term Memory) neural networks. The model uses historical Bitcoin price data to predict future price movements.

## Features

- Fetches real-time Bitcoin price data using yfinance
- Implements LSTM neural network for time series prediction
- Includes data preprocessing and feature engineering
- Provides model evaluation metrics and visualization
- Easy-to-use Python interface

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/chandratejatiriveedhi/btc-price-prediction.git
cd btc-price-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the example:
```bash
python example_usage.py
```

## Model Architecture

The model uses a stacked LSTM architecture with the following layers:
- Input LSTM layer with 50 units
- Dropout layer (0.2)
- LSTM layer with 50 units
- Dropout layer (0.2)
- LSTM layer with 50 units
- Dropout layer (0.2)
- Dense output layer

## Performance Metrics

The model's performance is evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R2) Score

## Contributing

Feel free to open issues and pull requests to improve the model or add new features.