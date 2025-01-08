import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

class BTCPricePredictor:
    def __init__(self, start_date='2015-01-01'):
        self.start_date = start_date
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch Bitcoin historical data from Yahoo Finance"""
        btc = yf.download('BTC-USD', start=self.start_date)
        self.data = btc[['Close']].copy()
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=30).std()
        self.data.dropna(inplace=True)
        return self.data
    
    def prepare_sequences(self, sequence_length=60):
        """Prepare sequences for LSTM model"""
        scaled_data = self.scaler.fit_transform(self.data[['Close']])
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y
    
    def build_model(self, sequence_length=60):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_model(self, X, y, epochs=50, batch_size=32, validation_split=0.2):
        """Train the LSTM model"""
        self.model = self.build_model(sequence_length=X.shape[1])
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def make_prediction(self, X_test):
        """Make predictions using trained model"""
        predictions = self.model.predict(X_test)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))
        return predictions
    
    def evaluate_model(self, y_true, y_pred):
        """Evaluate model performance"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
    
    def plot_results(self, y_true, y_pred, title='Bitcoin Price Prediction'):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='Actual Price')
        plt.plot(y_pred, label='Predicted Price')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.show()