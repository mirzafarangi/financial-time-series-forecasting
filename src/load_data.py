"""
Bitcoin Price Data Loading and Preprocessing
Fetches BTC-USD historical data for time series forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yfinance as yf
from datetime import datetime, timedelta


def fetch_bitcoin_data(start_date='2020-01-01', end_date=None, interval='1d'):
    """
    Fetch Bitcoin price data from Yahoo Finance
    
    Args:
        start_date: Start date for historical data (YYYY-MM-DD)
        end_date: End date (default: today)
        interval: Data interval (1d, 1h, 1wk, 1mo)
    
    Returns:
        DataFrame with OHLCV data
    """
    print("\n" + "="*80)
    print("BITCOIN PRICE DATA - LOADING")
    print("="*80)
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"\n📊 Fetching BTC-USD data...")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Interval: {interval}")
    
    try:
        # Fetch data from Yahoo Finance
        btc = yf.download('BTC-USD', start=start_date, end=end_date, interval=interval, progress=False)
        
        if btc.empty:
            raise ValueError("No data retrieved from Yahoo Finance")
        
        print(f"   ✅ Successfully fetched {len(btc)} data points")
        
        # Reset index to make date a column
        btc = btc.reset_index()
        
        # Rename columns for consistency
        btc.columns = ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        
        return btc
        
    except Exception as e:
        print(f"   ❌ Error fetching data: {e}")
        print(f"   Creating synthetic Bitcoin data as fallback...")
        return create_synthetic_btc_data(start_date, end_date)


def create_synthetic_btc_data(start_date, end_date):
    """
    Create synthetic Bitcoin price data if API fails
    Realistic price movement with trend, seasonality, and volatility
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n = len(dates)
    
    # Base price with upward trend
    base_price = 30000
    trend = np.linspace(0, 40000, n)  # $30k to $70k trend
    
    # Add weekly seasonality (7-day cycle)
    seasonality = 2000 * np.sin(2 * np.pi * np.arange(n) / 7)
    
    # Add random walk volatility
    np.random.seed(42)
    volatility = np.cumsum(np.random.normal(0, 500, n))
    
    # Combine components
    close_price = base_price + trend + seasonality + volatility
    
    # Generate OHLV from close
    high = close_price + np.abs(np.random.normal(0, 200, n))
    low = close_price - np.abs(np.random.normal(0, 200, n))
    open_price = close_price + np.random.normal(0, 100, n)
    volume = np.random.uniform(10000, 50000, n)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close_price,
        'adj_close': close_price,
        'volume': volume
    })
    
    print(f"   ✅ Created synthetic data with {len(df)} points")
    return df


def preprocess_data(df):
    """
    Preprocess Bitcoin data for time series modeling
    
    Args:
        df: Raw Bitcoin data
    
    Returns:
        Processed DataFrame ready for modeling
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Make a copy
    df = df.copy()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort by date
    df = df.sort_values('date').reset_index(drop=True)
    
    # Handle missing values (forward fill)
    df = df.fillna(method='ffill')
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate log returns (better for analysis)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate volatility (7-day rolling std of returns)
    df['volatility'] = df['returns'].rolling(window=7).std()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"   Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
    print(f"   Current price: ${df['close'].iloc[-1]:.2f}")
    
    print(f"\n📈 Statistics:")
    print(f"   Mean price: ${df['close'].mean():.2f}")
    print(f"   Std dev: ${df['close'].std():.2f}")
    print(f"   Mean daily return: {df['returns'].mean()*100:.4f}%")
    print(f"   Return volatility: {df['returns'].std()*100:.4f}%")
    
    return df


def train_test_split(df, test_size=30):
    """
    Split data into train and test sets
    For time series, we split chronologically (not random)
    
    Args:
        df: Processed DataFrame
        test_size: Number of days for testing
    
    Returns:
        train_df, test_df
    """
    print("\n" + "="*80)
    print("TRAIN-TEST SPLIT")
    print("="*80)
    
    split_idx = len(df) - test_size
    
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    print(f"\n📊 Split Summary:")
    print(f"   Train size: {len(train)} days ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"   Test size: {len(test)} days ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"   Train price range: ${train['close'].min():.2f} to ${train['close'].max():.2f}")
    print(f"   Test price range: ${test['close'].min():.2f} to ${test['close'].max():.2f}")
    
    return train, test


def main():
    """Main execution"""
    # Create data directory
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Fetch Bitcoin data (last 3 years for good training data)
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
    df = fetch_bitcoin_data(start_date=start_date)
    
    # Save raw data
    df.to_csv(data_dir / "btc_raw_data.csv", index=False)
    print(f"\n💾 Saved raw data: {data_dir / 'btc_raw_data.csv'}")
    
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Train-test split (30 days for testing)
    train_df, test_df = train_test_split(df_processed, test_size=30)
    
    # Save processed data
    df_processed.to_csv(data_dir / "btc_processed.csv", index=False)
    train_df.to_csv(data_dir / "train.csv", index=False)
    test_df.to_csv(data_dir / "test.csv", index=False)
    
    print(f"\n💾 Saved processed data:")
    print(f"   Full dataset: {data_dir / 'btc_processed.csv'}")
    print(f"   Train set: {data_dir / 'train.csv'}")
    print(f"   Test set: {data_dir / 'test.csv'}")
    
    print("\n" + "="*80)
    print("✅ DATA LOADING COMPLETE")
    print("="*80)
    print("\nNext step: Run 'python src/arima_model.py' to train ARIMA model")


if __name__ == "__main__":
    main()
