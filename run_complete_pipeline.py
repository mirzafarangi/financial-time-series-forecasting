"""
Complete Pipeline for Bitcoin Price Forecasting
Runs all models (ARIMA, SARIMA, Prophet) and generates comparison
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from load_data import main as load_data_main
from arima_model import main as arima_main
from sarima_model import main as sarima_main
from prophet_model import main as prophet_main
from garch_model import main as garch_main
from xgboost_direction import main as xgb_main
from evaluate_models import main as evaluate_main


def main():
    """Run complete forecasting pipeline"""
    print("\n" + "="*80)
    print("BITCOIN PRICE FORECASTING - COMPLETE PIPELINE")
    print("="*80)
    print("\nModels: ARIMA, SARIMA, Prophet, GARCH, XGBoost")
    print("Features: Real Binance data + Fear & Greed Index + Technical Indicators")
    print("Approach: Statistical + Volatility + ML Classification")
    print("\nNote: LSTM available in src/lstm_model.py but excluded for speed")
    
    print("\nThis will run:")
    print("1. Data loading and preprocessing")
    print("2. ARIMA model training")
    print("3. SARIMA model training")
    print("4. Prophet model training")
    print("5. GARCH model training")
    print("6. XGBoost direction classifier")
    print("7. Model comparison and evaluation")
    
    print("\n" + "="*80)
    print()
    
    # Step 1: Load and preprocess data
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("="*80)
    try:
        load_data_main()
    except Exception as e:
        print(f"ERROR in data loading: {e}")
        return
    
    # Step 2: Train ARIMA model
    print("\n" + "="*80)
    print("STEP 2: ARIMA MODEL")
    print("="*80)
    try:
        arima_main()
    except Exception as e:
        print(f"ERROR in ARIMA: {e}")
        return
    
    # Step 3: Train SARIMA model
    print("\n" + "="*80)
    print("STEP 3: SARIMA MODEL (Seasonal)")
    print("="*80)
    try:
        sarima_main()
    except Exception as e:
        print(f"ERROR in SARIMA: {e}")
        return
    
    # Step 4: Train Prophet model
    print("\n" + "="*80)
    print("STEP 4: PROPHET MODEL")
    print("="*80)
    try:
        prophet_main()
    except Exception as e:
        print(f"ERROR in Prophet: {e}")
        return
    
    # Step 5: Train GARCH model (volatility)
    print("\n" + "="*80)
    print("STEP 5: GARCH MODEL (Volatility Forecasting)")
    print("="*80)
    try:
        garch_main()
    except Exception as e:
        print(f"ERROR in GARCH: {e}")
        return
    
    # Step 6: Train LSTM model (SKIPPED - optional, computationally expensive)
    # print("\n" + "="*80)
    # print("STEP 6: LSTM MODEL (Deep Learning)")
    # print("="*80)
    # try:
    #     from lstm_model import main as lstm_main
    #     lstm_main()
    # except Exception as e:
    #     print(f"ERROR in LSTM: {e}")
    #     return
    
    # Step 6: Train XGBoost direction classifier
    print("\n" + "="*80)
    print("STEP 6: XGBOOST DIRECTION CLASSIFIER")
    print("="*80)
    try:
        xgb_main()
    except Exception as e:
        print(f"ERROR in XGBoost: {e}")
        return
    
    # Step 7: Compare all models
    print("\n" + "="*80)
    print("STEP 7: MODEL COMPARISON & EVALUATION")
    print("="*80)
    try:
        evaluate_main()
    except Exception as e:
        print(f"ERROR in evaluation: {e}")
        return
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\nResults saved in 'results/' folder:")
    print("   - 7 forecast visualizations (5 models)")
    print("   - Model comparison charts")
    print("   - Feature importance plots")
    print("   - Performance metrics (CSV)")
    print("   - Summary report (TXT)")
    print("   - Saved models (PKL)")
    print("\nModels trained (5 total):")
    print("   1. ARIMA - Classical statistical forecasting")
    print("   2. SARIMA - Seasonal ARIMA with weekly patterns")
    print("   3. Prophet - Facebook's production forecaster")
    print("   4. GARCH - Industry-standard volatility modeling")
    print("   5. XGBoost - Direction-of-movement classifier")
    print("\nApproaches covered:")
    print("   - Statistical time series (ARIMA/SARIMA/Prophet)")
    print("   - Volatility modeling (GARCH - fintech standard)")
    print("   - Machine learning classification (XGBoost)")
    print("   - External data integration (Fear & Greed Index)")
    print("\nOptional: Run 'python src/lstm_model.py' for deep learning baseline")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        raise
