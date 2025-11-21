"""
LSTM Model for Bitcoin Price Forecasting
Deep learning baseline for time series prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_train_data():
    """Load training data"""
    data_dir = Path(__file__).parent.parent / "data"
    return pd.read_csv(data_dir / "train.csv")


def load_test_data():
    """Load test data"""
    data_dir = Path(__file__).parent.parent / "data"
    return pd.read_csv(data_dir / "test.csv")


def create_sequences(data, lookback=60):
    """
    Create sequences for LSTM training
    
    Args:
        data: Scaled price data
        lookback: Number of previous days to use for prediction
    
    Returns:
        X, y: Sequences and targets
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


def build_lstm_model(lookback=60):
    """
    Build LSTM architecture
    
    Architecture:
    - LSTM layer (50 units) with dropout
    - LSTM layer (50 units) with dropout
    - Dense output layer
    
    Args:
        lookback: Sequence length
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model


def train_lstm_model(train_df, lookback=60, epochs=50):
    """
    Train LSTM model on Bitcoin prices
    
    Args:
        train_df: Training DataFrame
        lookback: Number of days to look back
        epochs: Number of training epochs
    
    Returns:
        model: Trained Keras model
        scaler: Fitted MinMaxScaler
    """
    print("\n" + "="*80)
    print("LSTM MODEL TRAINING")
    print("="*80)
    
    print(f"\nPreparing data...")
    print(f"   Data points: {len(train_df)}")
    print(f"   Lookback window: {lookback} days")
    
    # Extract closing prices
    prices = train_df['close'].values.reshape(-1, 1)
    
    # Scale data to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Create sequences
    X_train, y_train = create_sequences(scaled_prices, lookback)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    print(f"   Training sequences: {X_train.shape[0]}")
    print(f"   Sequence shape: {X_train.shape}")
    
    # Build model
    print(f"\nBuilding LSTM model...")
    model = build_lstm_model(lookback)
    
    print(f"\nModel Architecture:")
    model.summary()
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    print(f"\nTraining model ({epochs} epochs)...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )
    
    final_loss = history.history['loss'][-1]
    final_mae = history.history['mae'][-1]
    
    print(f"   Training complete")
    print(f"   Final loss: {final_loss:.6f}")
    print(f"   Final MAE: {final_mae:.6f}")
    
    return model, scaler


def forecast_lstm(model, scaler, train_df, test_df, lookback=60):
    """
    Generate LSTM forecasts for test period
    
    Args:
        model: Trained LSTM model
        scaler: Fitted MinMaxScaler
        train_df: Training DataFrame
        test_df: Test DataFrame
        lookback: Lookback window
    
    Returns:
        forecast: Array of predictions
    """
    print("\n" + "="*80)
    print("GENERATING FORECASTS")
    print("="*80)
    
    # Use last 'lookback' days from training data
    prices = train_df['close'].values.reshape(-1, 1)
    scaled_prices = scaler.transform(prices)
    
    # Initialize with last lookback days
    current_sequence = scaled_prices[-lookback:].reshape(1, lookback, 1)
    
    forecast = []
    
    print(f"\nForecasting {len(test_df)} days...")
    
    for i in range(len(test_df)):
        # Predict next day
        pred_scaled = model.predict(current_sequence, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0, 0]
        forecast.append(pred_price)
        
        # Update sequence: remove first, add prediction
        current_sequence = np.append(current_sequence[:, 1:, :], pred_scaled.reshape(1, 1, 1), axis=1)
    
    forecast = np.array(forecast)
    
    print(f"   Forecast generated")
    print(f"   First forecast: ${forecast[0]:,.2f}")
    print(f"   Last forecast: ${forecast[-1]:,.2f}")
    print(f"   Mean forecast: ${forecast.mean():,.2f}")
    
    return forecast


def evaluate_lstm(test_df, forecast):
    """Evaluate LSTM forecasts"""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    actual = test_df['close'].values
    
    # Calculate metrics
    mae = np.mean(np.abs(actual - forecast))
    rmse = np.sqrt(np.mean((actual - forecast) ** 2))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    # Direction accuracy
    actual_direction = np.diff(actual) > 0
    forecast_direction = np.diff(forecast) > 0
    direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
    
    print(f"\nLSTM Performance Metrics:")
    print(f"   MAE: ${mae:,.2f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Direction Accuracy: {direction_accuracy:.2f}%")
    
    metrics = {
        'model': 'LSTM',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }
    
    return metrics


def plot_results(train_df, test_df, forecast):
    """Plot LSTM forecast results"""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full timeline
    ax1.plot(train_df['date'], train_df['close'], label='Training Data',
             color='#2E86DE', alpha=0.7, linewidth=1)
    ax1.plot(test_df['date'], test_df['close'], label='Actual (Test)',
             color='#10AC84', linewidth=2.5)
    ax1.plot(test_df['date'], forecast, label='LSTM Forecast',
             color='#EE5A6F', linewidth=2.5, linestyle='--')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('BTC Price (USD)', fontsize=11)
    ax1.set_title('LSTM Bitcoin Price Forecast - Full Timeline',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Zoomed view
    zoom_start_idx = max(0, len(train_df) - 60)
    zoom_train = train_df.iloc[zoom_start_idx:]
    
    ax2.plot(zoom_train['date'], zoom_train['close'], label='Recent Training Data',
             color='#2E86DE', alpha=0.7, linewidth=1.5)
    ax2.plot(test_df['date'], test_df['close'], label='Actual (Test)',
             color='#10AC84', linewidth=3, marker='o', markersize=4)
    ax2.plot(test_df['date'], forecast, label='LSTM Forecast',
             color='#EE5A6F', linewidth=3, linestyle='--', marker='s', markersize=4)
    ax2.axvline(x=test_df['date'].iloc[0], color='gray', linestyle=':',
                linewidth=2, label='Forecast Start', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('BTC Price (USD)', fontsize=11)
    ax2.set_title('LSTM Forecast - Detailed View (Last 60 Days + 30-Day Forecast)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add metrics
    mae = np.mean(np.abs(test_df['close'].values - forecast))
    mape = np.mean(np.abs((test_df['close'].values - forecast) / test_df['close'].values)) * 100
    textstr = f'MAE: ${mae:,.0f}\nMAPE: {mape:.2f}%\nDeep Learning'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'lstm_forecast.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {results_dir / 'lstm_forecast.png'}")
    plt.close()


def main():
    """Main execution"""
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load data
    print("\nLoading training data...")
    train_df = load_train_data()
    
    print("Loading test data...")
    test_df = load_test_data()
    
    # Train LSTM
    model, scaler = train_lstm_model(train_df, lookback=60, epochs=50)
    
    # Generate forecasts
    forecast = forecast_lstm(model, scaler, train_df, test_df, lookback=60)
    
    # Evaluate
    metrics = evaluate_lstm(test_df, forecast)
    
    # Plot results
    plot_results(train_df, test_df, forecast)
    
    # Save model and metrics
    results_dir = Path(__file__).parent.parent / "results"
    model.save(results_dir / 'lstm_model.keras')
    print(f"Saved model: {results_dir / 'lstm_model.keras'}")
    
    joblib.dump(scaler, results_dir / 'lstm_scaler.pkl')
    print(f"Saved scaler: {results_dir / 'lstm_scaler.pkl'}")
    
    pd.DataFrame([metrics]).to_csv(results_dir / 'lstm_metrics.csv', index=False)
    print(f"Saved metrics: {results_dir / 'lstm_metrics.csv'}")
    
    print("\n" + "="*80)
    print("LSTM MODEL COMPLETE")
    print("="*80)
    print("\nNext step: Run 'python src/evaluate_models.py' to compare all models")


if __name__ == "__main__":
    main()
