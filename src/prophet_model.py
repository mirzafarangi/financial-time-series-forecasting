"""
Prophet Model for Bitcoin Price Forecasting
Facebook Prophet for trend and seasonality detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_train_data():
    """Load training data"""
    data_dir = Path(__file__).parent.parent / "data"
    train = pd.read_csv(data_dir / "train.csv", parse_dates=['date'])
    return train


def prepare_prophet_data(df):
    """
    Prepare data for Prophet (requires 'ds' and 'y' columns)
    
    Args:
        df: DataFrame with 'date' and 'close' columns
    
    Returns:
        DataFrame formatted for Prophet
    """
    prophet_df = df[['date', 'close']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df


def train_prophet_model(train_df):
    """
    Train Prophet model on Bitcoin prices
    
    Args:
        train_df: Training data
    
    Returns:
        Fitted Prophet model
    """
    print("\n" + "="*80)
    print("PROPHET MODEL TRAINING")
    print("="*80)
    
    # Prepare data for Prophet
    prophet_train = prepare_prophet_data(train_df)
    
    print(f"\n🏋️  Training Prophet model...")
    print(f"   Data points: {len(prophet_train)}")
    print(f"   Date range: {prophet_train['ds'].min().date()} to {prophet_train['ds'].max().date()}")
    
    # Initialize Prophet with reasonable parameters for crypto
    model = Prophet(
        changepoint_prior_scale=0.05,  # Flexibility for trend changes
        seasonality_prior_scale=10,     # Strength of seasonality
        daily_seasonality=False,        # No daily seasonality in daily data
        weekly_seasonality=True,        # Weekly patterns in crypto
        yearly_seasonality=True,        # Yearly trends
        interval_width=0.95             # 95% confidence intervals
    )
    
    # Fit model
    model.fit(prophet_train)
    
    print(f"   ✅ Model trained successfully")
    print(f"   Components: Trend + Weekly + Yearly seasonality")
    
    return model


def forecast_future(model, periods=30):
    """
    Forecast future prices with Prophet
    
    Args:
        model: Fitted Prophet model
        periods: Number of days to forecast
    
    Returns:
        forecast DataFrame
    """
    print(f"\n📈 Forecasting next {periods} days...")
    
    # Create future dates
    future = model.make_future_dataframe(periods=periods)
    
    # Generate forecast
    forecast = model.predict(future)
    
    # Get only the future predictions
    forecast_future = forecast.tail(periods)
    
    print(f"   ✅ Forecast generated")
    print(f"   First forecast: ${forecast_future['yhat'].iloc[0]:.2f}")
    print(f"   Last forecast: ${forecast_future['yhat'].iloc[-1]:.2f}")
    print(f"   Mean forecast: ${forecast_future['yhat'].mean():.2f}")
    
    return forecast


def evaluate_model(model, test_df):
    """
    Evaluate Prophet model on test set
    
    Args:
        model: Fitted Prophet model
        test_df: Test data
    
    Returns:
        Dictionary of metrics
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Prepare test data
    prophet_test = prepare_prophet_data(test_df)
    
    # Predict on test dates
    forecast = model.predict(prophet_test)
    
    # Get actual and predicted values
    actual = prophet_test['y'].values
    predicted = forecast['yhat'].values
    
    # Calculate metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Direction accuracy
    actual_direction = np.sign(np.diff(actual))
    forecast_direction = np.sign(np.diff(predicted))
    direction_accuracy = np.mean(actual_direction == forecast_direction) * 100
    
    print(f"\n📊 Prophet Performance Metrics:")
    print(f"   MAE: ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Direction Accuracy: {direction_accuracy:.2f}%")
    
    metrics = {
        'model': 'Prophet',
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'direction_accuracy': direction_accuracy
    }
    
    return metrics, forecast


def plot_results(model, train_df, test_df, forecast):
    """Plot actual vs forecast"""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Main forecast plot
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Plot training data
    ax.plot(train_df['date'], train_df['close'], label='Training Data', color='blue', alpha=0.6)
    
    # Plot test data (actual)
    ax.plot(test_df['date'], test_df['close'], label='Actual Test Data', color='green', linewidth=2)
    
    # Plot forecast for test period
    test_forecast = forecast[forecast['ds'].isin(test_df['date'])]
    ax.plot(test_forecast['ds'], test_forecast['yhat'], 
            label='Prophet Forecast', color='red', linewidth=2, linestyle='--')
    
    # Plot confidence interval
    ax.fill_between(test_forecast['ds'], 
                    test_forecast['yhat_lower'], 
                    test_forecast['yhat_upper'], 
                    color='red', alpha=0.2, label='95% Confidence Interval')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('BTC Price (USD)')
    ax.set_title('Prophet Bitcoin Price Forecast', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(results_dir / 'prophet_forecast.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved: {results_dir / 'prophet_forecast.png'}")
    plt.close()
    
    # Component plots (trend, weekly, yearly)
    fig = model.plot_components(forecast)
    fig.set_size_inches(12, 8)
    fig.savefig(results_dir / 'prophet_components.png', dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {results_dir / 'prophet_components.png'}")
    plt.close()


def main():
    """Main execution"""
    # Load data
    print("\n📂 Loading training data...")
    train_df = load_train_data()
    
    # Load test data
    data_dir = Path(__file__).parent.parent / "data"
    test_df = pd.read_csv(data_dir / "test.csv", parse_dates=['date'])
    
    # Train Prophet model
    model = train_prophet_model(train_df)
    
    # Evaluate on test set
    metrics, forecast = evaluate_model(model, test_df)
    
    # Plot results
    plot_results(model, train_df, test_df, forecast)
    
    # Save model
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    joblib.dump(model, results_dir / 'prophet_model.pkl')
    print(f"💾 Saved model: {results_dir / 'prophet_model.pkl'}")
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(results_dir / 'prophet_metrics.csv', index=False)
    print(f"💾 Saved metrics: {results_dir / 'prophet_metrics.csv'}")
    
    print("\n" + "="*80)
    print("✅ PROPHET MODEL COMPLETE")
    print("="*80)
    print("\nNext step: Run 'python src/evaluate_models.py' to compare all models")


if __name__ == "__main__":
    main()
