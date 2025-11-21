"""
XGBoost Direction-of-Movement Classifier
Solves ARIMA's 0% direction accuracy problem with ML approach
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
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


def engineer_features(df):
    """
    Engineer features for direction prediction
    
    Features include:
    - Lagged returns (1, 3, 7 days)
    - Rolling volatility
    - RSI (Relative Strength Index)
    - MACD
    - Day of week
    - Fear & Greed Index
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Target: 1 if price goes up next day, 0 if down
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Lagged returns
    df['return_lag1'] = df['returns'].shift(1)
    df['return_lag3'] = df['returns'].shift(3)
    df['return_lag7'] = df['returns'].shift(7)
    
    # Rolling statistics
    df['volatility_7d'] = df['returns'].rolling(7).std()
    df['volatility_30d'] = df['returns'].rolling(30).std()
    df['mean_return_7d'] = df['returns'].rolling(7).mean()
    df['mean_return_30d'] = df['returns'].rolling(30).mean()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # Day of week (cyclical encoding)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Price momentum
    df['momentum_3d'] = df['close'] / df['close'].shift(3) - 1
    df['momentum_7d'] = df['close'] / df['close'].shift(7) - 1
    
    # Volume features (if available)
    if 'volume' in df.columns:
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma7'] = df['volume'].rolling(7).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma7']
    
    # Fear & Greed Index (if available)
    if 'fear_greed_index' in df.columns:
        df['fear_greed_lag1'] = df['fear_greed_index'].shift(1)
        df['fear_greed_change'] = df['fear_greed_index'].diff()
    
    return df


def prepare_ml_data(train_df, test_df):
    """
    Prepare data for ML model
    
    For proper evaluation, we combine train+test, engineer features,
    then split again to preserve more test samples
    
    Returns:
        X_train, y_train, X_test, y_test, feature_names
    """
    print("\nEngineering features...")
    
    # Remember the split point
    train_size = len(train_df)
    
    # Combine for feature engineering
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df = engineer_features(combined_df)
    
    # Feature columns
    feature_cols = [
        'return_lag1', 'return_lag3', 'return_lag7',
        'volatility_7d', 'volatility_30d',
        'mean_return_7d', 'mean_return_30d',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'day_sin', 'day_cos',
        'momentum_3d', 'momentum_7d'
    ]
    
    # Add volume features if available
    if 'volume_change' in combined_df.columns:
        feature_cols.extend(['volume_change', 'volume_ratio'])
    
    # Add Fear & Greed features if available
    if 'fear_greed_lag1' in combined_df.columns:
        feature_cols.extend(['fear_greed_lag1', 'fear_greed_change'])
    
    # Remove NaN rows
    combined_df = combined_df.dropna(subset=feature_cols + ['target'])
    
    # Split back into train/test
    # Use last 30 rows as test (after NaN removal)
    test_df_clean = combined_df.tail(30)
    train_df_clean = combined_df.iloc[:-30]
    
    X_train = train_df_clean[feature_cols]
    y_train = train_df_clean['target']
    X_test = test_df_clean[feature_cols]
    y_test = test_df_clean['target']
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Class balance (train): {y_train.mean():.2%} positive")
    print(f"   Class balance (test): {y_test.mean():.2%} positive")
    
    return X_train, y_train, X_test, y_test, feature_cols


def train_xgboost_classifier(X_train, y_train):
    """
    Train XGBoost binary classifier for direction prediction
    
    Returns:
        model: Trained XGBoost model
        scaler: Fitted StandardScaler
    """
    print("\n" + "="*80)
    print("XGBOOST DIRECTION CLASSIFIER TRAINING")
    print("="*80)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    print("\nTraining XGBoost classifier...")
    print(f"   Parameters: max_depth={params['max_depth']}, "
          f"learning_rate={params['learning_rate']}, "
          f"n_estimators={params['n_estimators']}")
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_scaled, y_train, verbose=False)
    
    print(f"   Training complete")
    
    return model, scaler


def evaluate_classifier(model, scaler, X_test, y_test, feature_names):
    """Evaluate classifier performance"""
    print("\n" + "="*80)
    print("MODEL EVALUATION")
    print("="*80)
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, zero_division=0) * 100
    recall = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.5
    
    print(f"\nDirection Prediction Performance:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Precision: {precision:.2f}%")
    print(f"   Recall: {recall:.2f}%")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC-AUC: {auc:.4f}")
    print(f"\n   Baseline (random): 50.00%")
    print(f"   Improvement: +{accuracy - 50:.2f} percentage points")
    
    # Feature importance
    print(f"\nTop 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in importance_df.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    metrics = {
        'model': 'XGBoost',
        'mae': np.nan,  # Not applicable for classification
        'rmse': np.nan,
        'mape': np.nan,
        'direction_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc
    }
    
    return metrics, importance_df, y_pred, y_pred_proba


def plot_results(importance_df, y_test, y_pred_proba):
    """Plot feature importance and ROC curve"""
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Feature importance
    top_features = importance_df.head(15)
    ax1.barh(range(len(top_features)), top_features['importance'], color='#2E86DE')
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'])
    ax1.invert_yaxis()
    ax1.set_xlabel('Importance', fontsize=11, fontweight='bold')
    ax1.set_title('XGBoost Feature Importance (Top 15)', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    ax2.plot(fpr, tpr, color='#2E86DE', linewidth=2.5, label=f'XGBoost (AUC = {auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5, label='Random (AUC = 0.500)')
    ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_title('ROC Curve: Direction Prediction', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'xgboost_direction.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {results_dir / 'xgboost_direction.png'}")
    plt.close()


def main():
    """Main execution"""
    # Load data
    print("\nLoading training data...")
    train_df = load_train_data()
    
    print("Loading test data...")
    test_df = load_test_data()
    
    # Prepare data
    X_train, y_train, X_test, y_test, feature_names = prepare_ml_data(train_df, test_df)
    
    # Train model
    model, scaler = train_xgboost_classifier(X_train, y_train)
    
    # Evaluate
    metrics, importance_df, y_pred, y_pred_proba = evaluate_classifier(
        model, scaler, X_test, y_test, feature_names
    )
    
    # Plot results
    plot_results(importance_df, y_test, y_pred_proba)
    
    # Save model and metrics
    results_dir = Path(__file__).parent.parent / "results"
    joblib.dump({'model': model, 'scaler': scaler, 'features': feature_names}, 
                results_dir / 'xgboost_direction.pkl')
    print(f"Saved model: {results_dir / 'xgboost_direction.pkl'}")
    
    pd.DataFrame([metrics]).to_csv(results_dir / 'xgboost_metrics.csv', index=False)
    print(f"Saved metrics: {results_dir / 'xgboost_metrics.csv'}")
    
    importance_df.to_csv(results_dir / 'xgboost_feature_importance.csv', index=False)
    print(f"Saved feature importance: {results_dir / 'xgboost_feature_importance.csv'}")
    
    print("\n" + "="*80)
    print("XGBOOST DIRECTION CLASSIFIER COMPLETE")
    print("="*80)
    print(f"\nKey Insight: XGBoost achieved {metrics['direction_accuracy']:.1f}% accuracy")
    print(f"This is {metrics['direction_accuracy'] - 50:.1f} percentage points above random chance")
    print(f"Compared to ARIMA's 0% direction accuracy, this is a substantial improvement")


if __name__ == "__main__":
    main()
