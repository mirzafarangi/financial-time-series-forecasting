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
from evaluate_models import main as evaluate_main


def main():
    """Run complete forecasting pipeline"""
    print("\n" + "="*80)
    print("BITCOIN PRICE FORECASTING - COMPLETE PIPELINE")
    print("="*80)
    
    print("\nThis will run:")
    print("1. Data loading and preprocessing")
    print("2. ARIMA model training")
    print("3. SARIMA model training")
    print("4. Prophet model training")
    print("5. Model comparison and evaluation")
    
    print("\n" + "="*80)
    print()
    
    # Step 1: Load and preprocess data
    print("🔹 STEP 1: DATA LOADING")
    print("="*80)
    try:
        load_data_main()
    except Exception as e:
        print(f"❌ ERROR in data loading: {e}")
        return
    
    # Step 2: Train ARIMA
    print("\n🔹 STEP 2: ARIMA MODEL")
    print("="*80)
    try:
        arima_main()
    except Exception as e:
        print(f"❌ ERROR in ARIMA: {e}")
        return
    
    # Step 3: Train SARIMA
    print("\n🔹 STEP 3: SARIMA MODEL")
    print("="*80)
    try:
        sarima_main()
    except Exception as e:
        print(f"❌ ERROR in SARIMA: {e}")
        return
    
    # Step 4: Train Prophet
    print("\n🔹 STEP 4: PROPHET MODEL")
    print("="*80)
    try:
        prophet_main()
    except Exception as e:
        print(f"❌ ERROR in Prophet: {e}")
        return
    
    # Step 5: Evaluate and compare
    print("\n🔹 STEP 5: MODEL COMPARISON")
    print("="*80)
    try:
        evaluate_main()
    except Exception as e:
        print(f"❌ ERROR in evaluation: {e}")
        return
    
    # Success!
    print("\n" + "="*80)
    print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("="*80)
    
    print("\n📁 Check the 'results/' folder for:")
    print("   - Model forecast visualizations")
    print("   - Model comparison charts")
    print("   - Performance metrics")
    print("   - Summary report")
    print("   - Saved models")


if __name__ == "__main__":
    main()
