"""
ARIMA Model for Bitcoin Price Forecasting
Auto ARIMA for automatic parameter selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_train_data():
    """Load training data"""
    data_dir = Path(__file__).parent.parent / "data"
    train = pd.read_csv(data_dir / "train.csv", parse_dates=['date'])
    return train


def find_best_arima_order(train_series, seasonal=False):
    """
    Use auto_arima to find best (p,d,q) parameters
    
    Args:
        train_series: Time series data for training
        seasonal: Whether to use seasonal ARIMA
    
    Returns:
        Best ARIMA order (p,d,q)
    """
    print("\n🔍 Finding best ARIMA parameters with auto_arima...")
    print("   This may take a few minutes...")
    
    auto_model = auto_arima(
        train_series,
        start_p=0, start_q=0,
        max_p=5, max_q=5,
        d=None,  # Let auto_arima determine d
        seasonal=seasonal,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )
    
    order = auto_model.order
    print(f"   ✅ Best ARIMA order: {order}")
    print(f"   AIC: {auto_model.aic():.2f}")
    
    return order


def train_arima_model(train_df, order=None):
    """
    Train ARIMA model on Bitcoin prices
    
    Args:
        train_df: Training data
        order: ARIMA order (p,d,q), if None will auto-select
    
    Returns:
        Fitted ARIMA model
    """
    print("\n" + "="*80)
    print("ARIMA MODEL TRAINING")
    print("="*80)
    
    # Use close price for forecasting
    train_series = train_df['close'].values
    
    # Find best order if not provided
    if order is None:
        order = find_best_arima_order(train_series)
    
    print(f"\n🏋️  Training ARIMA{order} model...")
    
    # Train ARIMA model
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    
    print(f"   ✅ Model trained successfully")
    print(f"   AIC: {fitted_model.aic:.2f}")
    print(f"   BIC: {fitted_model.bic:.2f}")
    
    return fitted_model


def forecast_future(fitted_model, steps=30):
    """
    Forecast future prices
    
    Args:
        fitted_model: Fitted ARIMA model
        steps: Number of steps to forecast
    
    Returns:
        forecast: Predicted values
        conf_int: Confidence intervals
    """
    print(f"\n📈 Forecasting next {steps} days...")
    
    # Get forecast with confidence intervals
    forecast_result = fitted_model.get_forecast(steps=steps)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)  # 95% confidence interval
    
    print(f"   ✅ Forecast generated")
    print(f"   First forecast: ${forecast[0]:.2f}")
    print(f"   Last forecast: ${forecast[-1]:.2f}")
    print(f"   Mean forecast: ${forecast.mean():.2f}")
    
    return forecast, conf_int


def evaluate_model(fitted_model, test_df):
    """
    Evaluate model on test set
    
    Args:
        fitted_model: Fitted ARIMA model
        test_df: Test data
    
    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Get actual values
    actual = test_df['close'].values
    
    # Forecast
    forecast, conf_int = forecast_future(fitted_model, steps=len(test_df))
    
    # Calculate metrics
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mape = np.mean(np.abs((actual - forecast) / actual)) * 100
    
    # Direction accuracy (did we predict up/down correctly?)
    actual_direction = np.sign(np.diff(actual))
    forecast_direction = np.sign(np.diff(forecast))
    direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
    
    print(f"\n📊 ARIMA Performance Metrics:")
    print(f"   MAE: ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Direction Accuracy: {direction_accuracy:.2f}%")
    
    metrics = {
        'model': 'ARIMA',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }
    
    return metrics, forecast, conf_int


def plot_results(train_df, test_df, forecast, conf_int):
    """Plot actual vs forecast"""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    
    # Plot training data
    plt.plot(train_df['date'], train_df['close'], label='Training Data', color='blue', alpha=0.6)
    
    # Plot test data (actual)
    plt.plot(test_df['date'], test_df['close'], label='Actual Test Data', color='green', linewidth=2)
    
    # Plot forecast
    plt.plot(test_df['date'], forecast, label='ARIMA Forecast', color='red', linewidth=2, linestyle='--')
    
    # Plot confidence interval
    plt.fill_between(test_df['date'], conf_int[:, 0], conf_int[:, 1], 
                     color='red', alpha=0.2, label='95% Confidence Interval')
    
    plt.xlabel('Date')
    plt.ylabel('BTC Price (USD)')
    plt.title('ARIMA Bitcoin Price Forecast', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(results_dir / 'arima_forecast.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved: {results_dir / 'arima_forecast.png'}")
    plt.close()


def main():
    """Main execution"""
    # Load data
    print("\n📂 Loading training data...")
    train_df = load_train_data()
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "data"
    test_df = pd.read_csv(data_dir / "test.csv", parse_dates=['date'])
    
    # Train ARIMA model (auto-select best order)
    fitted_model = train_arima_model(train_df)
    
    # Evaluate on test set
    metrics, forecast, conf_int = evaluate_model(fitted_model, test_df)
    
    # Plot results
    plot_results(train_df, test_df, forecast, conf_int)
    
    # Save model
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    joblib.dump(fitted_model, results_dir / 'arima_model.pkl')
    print(f"💾 Saved model: {results_dir / 'arima_model.pkl'}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / 'arima_metrics.csv', index=False)
    print(f"💾 Saved metrics: {results_dir / 'arima_metrics.csv'}")
    
    print("\n" + "="*80)
    print("✅ ARIMA MODEL COMPLETE")
    print("="*80)
    print("\nNext step: Run 'python src/sarima_model.py' to train SARIMA model")


if __name__ == "__main__":
    main()
