"""
SARIMA Model for Bitcoin Price Forecasting
Seasonal ARIMA with automatic parameter selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def find_best_sarima_order(train_series, seasonal_period=7):
    """
    Use auto_arima to find best SARIMA parameters
    
    Args:
        train_series: Time series data
        seasonal_period: Seasonal period (7 for weekly pattern in daily data)
    
    Returns:
        order (p,d,q) and seasonal_order (P,D,Q,s)
    """
    print(f"\n🔍 Finding best SARIMA parameters...")
    print(f"   Seasonal period: {seasonal_period} days (weekly pattern)")
    print("   This may take several minutes...")
    
    auto_model = auto_arima(
        train_series,
        start_p=0, start_q=0,
        max_p=3, max_q=3,
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None, D=None,  # Let auto_arima determine differencing
        seasonal=True,
        m=seasonal_period,  # Seasonal period
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )
    
    order = auto_model.order
    seasonal_order = auto_model.seasonal_order
    
    print(f"   ✅ Best SARIMA order: {order}")
    print(f"   Seasonal order: {seasonal_order}")
    print(f"   AIC: {auto_model.aic():.2f}")
    
    return order, seasonal_order


def train_sarima_model(train_df, order=None, seasonal_order=None):
    """
    Train SARIMA model on Bitcoin prices
    
    Args:
        train_df: Training data
        order: ARIMA order (p,d,q)
        seasonal_order: Seasonal order (P,D,Q,s)
    
    Returns:
        Fitted SARIMA model
    """
    print("\n" + "="*80)
    print("SARIMA MODEL TRAINING")
    print("="*80)
    
    # Use close price
    train_series = train_df['close'].values
    
    # Find best parameters if not provided
    if order is None or seasonal_order is None:
        order, seasonal_order = find_best_sarima_order(train_series, seasonal_period=7)
    
    print(f"\n🏋️  Training SARIMA{order}x{seasonal_order} model...")
    
    # Train SARIMA model
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted_model = model.fit(disp=False)
    
    print(f"   ✅ Model trained successfully")
    print(f"   AIC: {fitted_model.aic:.2f}")
    print(f"   BIC: {fitted_model.bic:.2f}")
    
    return fitted_model


def forecast_future(fitted_model, steps=30):
    """
    Forecast future prices
    
    Args:
        fitted_model: Fitted SARIMA model
        steps: Number of steps to forecast
    
    Returns:
        forecast, conf_int
    """
    print(f"\n📈 Forecasting next {steps} days...")
    
    # Get forecast with confidence intervals
    forecast_result = fitted_model.get_forecast(steps=steps)
    forecast = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)
    
    print(f"   ✅ Forecast generated")
    
    # Convert to numpy array if it's a Series
    if hasattr(forecast, 'values'):
        forecast_array = forecast.values
        conf_int_array = conf_int.values
    else:
        forecast_array = forecast
        conf_int_array = conf_int
    
    print(f"   First forecast: ${forecast_array[0]:.2f}")
    print(f"   Last forecast: ${forecast_array[-1]:.2f}")
    print(f"   Mean forecast: ${forecast_array.mean():.2f}")
    
    return forecast_array, conf_int_array


def evaluate_model(fitted_model, test_df):
    """
    Evaluate model on test set
    
    Args:
        fitted_model: Fitted SARIMA model
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
    
    # Direction accuracy
    actual_direction = np.sign(np.diff(actual))
    forecast_direction = np.sign(np.diff(forecast))
    direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
    
    print(f"\n📊 SARIMA Performance Metrics:")
    print(f"   MAE: ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Direction Accuracy: {direction_accuracy:.2f}%")
    
    metrics = {
        'model': 'SARIMA',
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
    plt.plot(test_df['date'], forecast, label='SARIMA Forecast', color='red', linewidth=2, linestyle='--')
    
    # Plot confidence interval
    plt.fill_between(test_df['date'], conf_int[:, 0], conf_int[:, 1], 
                     color='red', alpha=0.2, label='95% Confidence Interval')
    
    plt.xlabel('Date')
    plt.ylabel('BTC Price (USD)')
    plt.title('SARIMA Bitcoin Price Forecast (with Weekly Seasonality)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(results_dir / 'sarima_forecast.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved: {results_dir / 'sarima_forecast.png'}")
    plt.close()


def main():
    """Main execution"""
    # Load data
    print("\n📂 Loading training data...")
    train_df = load_train_data()
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "data"
    test_df = pd.read_csv(data_dir / "test.csv", parse_dates=['date'])
    
    # Train SARIMA model
    fitted_model = train_sarima_model(train_df)
    
    # Evaluate on test set
    metrics, forecast, conf_int = evaluate_model(fitted_model, test_df)
    
    # Plot results
    plot_results(train_df, test_df, forecast, conf_int)
    
    # Save model
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    joblib.dump(fitted_model, results_dir / 'sarima_model.pkl')
    print(f"💾 Saved model: {results_dir / 'sarima_model.pkl'}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / 'sarima_metrics.csv', index=False)
    print(f"💾 Saved metrics: {results_dir / 'sarima_metrics.csv'}")
    
    print("\n" + "="*80)
    print("✅ SARIMA MODEL COMPLETE")
    print("="*80)
    print("\nNext step: Run 'python src/prophet_model.py' to train Prophet model")


if __name__ == "__main__":
    main()
