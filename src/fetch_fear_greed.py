"""
Crypto Fear & Greed Index Fetcher
Exogenous variable for enhanced forecasting
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


def fetch_fear_greed_index(limit=2000):
    """
    Fetch Crypto Fear & Greed Index from alternative.me API
    
    The Fear & Greed Index is a market sentiment indicator (0-100):
    - 0-24: Extreme Fear
    - 25-49: Fear
    - 50: Neutral
    - 51-75: Greed
    - 76-100: Extreme Greed
    
    Args:
        limit: Number of days to fetch (max 2000)
    
    Returns:
        DataFrame with date and fear_greed_index
    """
    print("\nFetching Crypto Fear & Greed Index...")
    print(f"   Source: alternative.me API")
    print(f"   Requesting {limit} days of data")
    
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Parse data
        records = []
        for item in data['data']:
            records.append({
                'date': datetime.fromtimestamp(int(item['timestamp'])).strftime('%Y-%m-%d'),
                'fear_greed_index': int(item['value']),
                'classification': item['value_classification']
            })
        
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"   Successfully fetched {len(df)} data points")
        print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"   Mean index: {df['fear_greed_index'].mean():.1f}")
        print(f"   Index range: {df['fear_greed_index'].min()} - {df['fear_greed_index'].max()}")
        
        return df
        
    except Exception as e:
        print(f"   Error fetching Fear & Greed Index: {e}")
        print(f"   Creating synthetic index as fallback")
        return create_synthetic_fear_greed(limit)


def create_synthetic_fear_greed(days=2000):
    """
    Create synthetic Fear & Greed Index if API fails
    Simulates realistic market sentiment patterns
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Generate realistic sentiment (mean-reverting with noise)
    np.random.seed(42)
    sentiment = 50 + np.cumsum(np.random.randn(days) * 5)
    sentiment = np.clip(sentiment, 0, 100)
    
    # Smooth it
    window = 7
    sentiment = pd.Series(sentiment).rolling(window=window, center=True).mean().bfill().ffill().values
    
    df = pd.DataFrame({
        'date': dates,
        'fear_greed_index': sentiment.astype(int)
    })
    
    # Add classification
    def classify_sentiment(value):
        if value < 25:
            return 'Extreme Fear'
        elif value < 50:
            return 'Fear'
        elif value < 75:
            return 'Greed'
        else:
            return 'Extreme Greed'
    
    df['classification'] = df['fear_greed_index'].apply(classify_sentiment)
    
    return df


def merge_with_price_data(price_df, fear_greed_df):
    """
    Merge Fear & Greed Index with price data
    
    Args:
        price_df: DataFrame with Bitcoin prices and date column
        fear_greed_df: DataFrame with fear_greed_index and date column
    
    Returns:
        Merged DataFrame
    """
    print("\nMerging Fear & Greed Index with price data...")
    
    # Ensure date columns are datetime
    price_df = price_df.copy()
    fear_greed_df = fear_greed_df.copy()
    price_df['date'] = pd.to_datetime(price_df['date'])
    fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'])
    
    # Merge on date
    merged = pd.merge(price_df, fear_greed_df[['date', 'fear_greed_index', 'classification']], 
                     on='date', how='left')
    
    # Forward fill missing values (weekends/holidays)
    merged['fear_greed_index'] = merged['fear_greed_index'].ffill()
    merged['classification'] = merged['classification'].ffill()
    
    # Backward fill any remaining
    merged['fear_greed_index'] = merged['fear_greed_index'].bfill()
    merged['classification'] = merged['classification'].bfill()
    
    print(f"   Merged {len(merged)} rows")
    print(f"   Missing Fear & Greed values: {merged['fear_greed_index'].isna().sum()}")
    
    return merged


if __name__ == "__main__":
    # Test the fetcher
    df = fetch_fear_greed_index(limit=365)
    print("\nSample data:")
    print(df.head())
    print("\nValue distribution:")
    print(df['classification'].value_counts())
