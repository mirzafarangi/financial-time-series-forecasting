"""
GARCH Model for Bitcoin Volatility Forecasting
Industry-standard approach for financial volatility modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from arch import arch_model
import joblib


def load_train_data():
    """Load training data"""
    data_dir = Path(__file__).parent.parent / "data"
    return pd.read_csv(data_dir / "train.csv")


def load_test_data():
    """Load test data"""
    data_dir = Path(__file__).parent.parent / "data"
    return pd.read_csv(data_dir / "test.csv")


def train_garch_model(train_df):
    """
    Train GARCH(1,1) model on Bitcoin returns
    
    GARCH is the industry standard for volatility forecasting in finance.
    It models:
    - Volatility clustering (high volatility follows high volatility)
    - Leverage effects (negative returns increase volatility more)
    - Time-varying variance
    
    Args:
        train_df: Training DataFrame with returns
    
    Returns:
        Fitted GARCH model
    """
    print("\n" + "="*80)
    print("GARCH MODEL TRAINING")
    print("="*80)
    
    # Use returns (already calculated in preprocessing)
    # Scale returns to percentage for better numerical stability
    returns = train_df['returns'].dropna() * 100
    
    print(f"\nTraining GARCH(1,1) model...")
    print(f"   Data points: {len(returns)}")
    print(f"   Mean return: {returns.mean():.4f}%")
    print(f"   Return std: {returns.std():.4f}%")
    
    # Fit GARCH(1,1) - standard specification
    # p=1 (ARCH term), q=1 (GARCH term)
    model = arch_model(
        returns,
        vol='Garch',  # GARCH volatility
        p=1,          # ARCH order
        q=1,          # GARCH order
        dist='normal' # Normal distribution
    )
    
    # Fit model
    result = model.fit(disp='off')
    
    print(f"   Model trained successfully")
    print(f"\nModel Parameters:")
    print(f"   omega (const): {result.params['omega']:.6f}")
    print(f"   alpha (ARCH): {result.params['alpha[1]']:.6f}")
    print(f"   beta (GARCH): {result.params['beta[1]']:.6f}")
    print(f"   Persistence: {result.params['alpha[1]'] + result.params['beta[1]']:.6f}")
    print(f"\nModel Fit:")
    print(f"   AIC: {result.aic:.2f}")
    print(f"   BIC: {result.bic:.2f}")
    print(f"   Log-Likelihood: {result.loglikelihood:.2f}")
    
    return result


def forecast_volatility(garch_result, test_df):
    """
    Forecast volatility for test period
    
    Args:
        garch_result: Fitted GARCH model
        test_df: Test DataFrame
    
    Returns:
        volatility_forecast: Array of volatility forecasts
    """
    print("\n" + "="*80)
    print("VOLATILITY FORECASTING")
    print("="*80)
    
    horizon = len(test_df)
    print(f"\nForecasting volatility for {horizon} days...")
    
    # Forecast volatility
    forecast = garch_result.forecast(horizon=horizon, reindex=False)
    volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
    
    print(f"   Forecast generated")
    print(f"   Mean volatility: {volatility_forecast.mean():.4f}%")
    print(f"   Min volatility: {volatility_forecast.min():.4f}%")
    print(f"   Max volatility: {volatility_forecast.max():.4f}%")
    
    return volatility_forecast


def evaluate_garch(test_df, volatility_forecast):
    """
    Evaluate GARCH volatility forecasts
    
    Uses realized volatility (absolute returns) as proxy
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Calculate realized volatility (absolute returns as proxy)
    realized_vol = np.abs(test_df['returns'].values * 100)
    
    # Remove NaN values
    mask = ~np.isnan(realized_vol)
    realized_vol = realized_vol[mask]
    vol_forecast = volatility_forecast[mask]
    
    # Calculate metrics
    mae = np.mean(np.abs(realized_vol - vol_forecast))
    rmse = np.sqrt(np.mean((realized_vol - vol_forecast) ** 2))
    
    # Calculate directional accuracy for volatility changes
    realized_changes = np.diff(realized_vol) > 0
    forecast_changes = np.diff(vol_forecast) > 0
    direction_accuracy = np.mean(realized_changes == forecast_changes) * 100
    
    print(f"\nVolatility Forecast Performance:")
    print(f"   MAE: {mae:.4f}%")
    print(f"   RMSE: {rmse:.4f}%")
    print(f"   Direction Accuracy: {direction_accuracy:.2f}%")
    
    metrics = {
        'model': 'GARCH',
        'mae': mae,
        'rmse': rmse,
        'direction_accuracy': direction_accuracy
    }
    
    return metrics, realized_vol, vol_forecast


def plot_results(train_df, test_df, garch_result, volatility_forecast, realized_vol):
    """Plot GARCH results and conditional volatility"""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot 1: Returns with conditional volatility
    returns = train_df['returns'].dropna() * 100
    dates = train_df['date'].iloc[1:]  # Skip first NaN return
    
    ax1.plot(dates, returns, label='Daily Returns', color='#2E86DE', alpha=0.6, linewidth=0.8)
    ax1.fill_between(dates, -garch_result.conditional_volatility, 
                     garch_result.conditional_volatility,
                     color='#EE5A6F', alpha=0.2, label='Conditional Volatility')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Returns (%)', fontsize=11)
    ax1.set_title('Bitcoin Returns with GARCH Conditional Volatility', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Volatility forecast vs realized
    test_dates = test_df['date'].values
    # Align arrays
    min_len = min(len(test_dates), len(volatility_forecast), len(realized_vol))
    test_dates = test_dates[:min_len]
    vol_forecast_plot = volatility_forecast[:min_len]
    realized_vol_plot = realized_vol[:min_len]
    
    ax2.plot(test_dates, realized_vol_plot, label='Realized Volatility', 
             color='#10AC84', linewidth=2.5, marker='o', markersize=4)
    ax2.plot(test_dates, vol_forecast_plot, label='GARCH Forecast', 
             color='#EE5A6F', linewidth=2.5, linestyle='--', marker='s', markersize=4)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Volatility (%)', fontsize=11)
    ax2.set_title('GARCH Volatility Forecast vs Realized Volatility', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: QQ plot for standardized residuals
    from scipy import stats
    std_resid = garch_result.std_resid
    stats.probplot(std_resid, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot: Standardized Residuals', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'garch_volatility.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {results_dir / 'garch_volatility.png'}")
    plt.close()


def main():
    """Main execution"""
    # Load data
    print("\nLoading training data...")
    train_df = load_train_data()
    
    print("Loading test data...")
    test_df = load_test_data()
    
    # Train GARCH model
    garch_result = train_garch_model(train_df)
    
    # Forecast volatility
    volatility_forecast = forecast_volatility(garch_result, test_df)
    
    # Evaluate
    metrics, realized_vol, vol_forecast = evaluate_garch(test_df, volatility_forecast)
    
    # Plot results
    plot_results(train_df, test_df, garch_result, volatility_forecast, realized_vol)
    
    # Save model and metrics
    results_dir = Path(__file__).parent.parent / "results"
    joblib.dump(garch_result, results_dir / 'garch_model.pkl')
    print(f"Saved model: {results_dir / 'garch_model.pkl'}")
    
    pd.DataFrame([metrics]).to_csv(results_dir / 'garch_metrics.csv', index=False)
    print(f"Saved metrics: {results_dir / 'garch_metrics.csv'}")
    
    print("\n" + "="*80)
    print("GARCH MODEL COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
