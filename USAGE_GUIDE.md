# Usage Guide - Bitcoin Price Forecasting

Complete guide for running, understanding, and presenting this time series forecasting project.

## 🚀 Quick Start

### 1. Installation (2 minutes)

```bash
cd /Users/ashimashi/Desktop/Jobs/n26/financial-time-series-forecasting

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline (5-10 minutes)

```bash
# Run everything with one command
python run_complete_pipeline.py
```

**What happens:**
1. Fetches Bitcoin data from Yahoo Finance (last 3 years)
2. Preprocesses data and creates train/test split
3. Trains ARIMA model with auto parameter selection
4. Trains SARIMA model with weekly seasonality
5. Trains Prophet model with trend+seasonality
6. Compares all models and generates visualizations
7. Creates summary report

**Output:** All results saved in `results/` folder

### 3. View Results

```bash
# View all generated files
ls results/

# View comparison charts
open results/model_comparison.png
open results/sarima_forecast.png

# Read summary report
cat results/summary_report.txt
```

---

## 📂 Project Components

### Data Loading (`src/load_data.py`)

**What it does:**
- Fetches BTC-USD data from Yahoo Finance API
- Preprocesses: calculates returns, volatility
- Chronological train-test split (no data leakage)

**Run individually:**
```bash
python src/load_data.py
```

**Output:**
- `data/btc_raw_data.csv` - Raw OHLCV data
- `data/btc_processed.csv` - With returns and volatility
- `data/train.csv` - Training set (1066 days)
- `data/test.csv` - Test set (30 days)

**Key Learnings:**
- Time series requires **chronological split** (not random)
- Last 30 days held out for testing (realistic forecasting scenario)
- Features include returns, log returns, rolling volatility

---

### ARIMA Model (`src/arima_model.py`)

**What it does:**
- Auto-selects best (p,d,q) parameters
- Trains ARIMA model on training data
- Forecasts 30 days ahead
- Evaluates on test set

**Run individually:**
```bash
python src/arima_model.py
```

**Output:**
- `results/arima_forecast.png` - Forecast with confidence intervals
- `results/arima_model.pkl` - Saved model
- `results/arima_metrics.csv` - Performance metrics

**Actual Results:**
- MAE: $1,546.73
- MAPE: 1.77%
- Direction Accuracy: 96.55%
- Best ARIMA order: (5,1,1)

**When to use ARIMA:**
- Simple, interpretable forecasts
- No clear seasonality in data
- Baseline model for comparison

---

### SARIMA Model (`src/sarima_model.py`)

**What it does:**
- Adds **seasonal component** to ARIMA
- Auto-selects best parameters including seasonality
- Models weekly patterns (7-day cycle)
- Forecasts with seasonal adjustments

**Run individually:**
```bash
python src/sarima_model.py
```

**Output:**
- `results/sarima_forecast.png` - Forecast with seasonality
- `results/sarima_model.pkl` - Saved model
- `results/sarima_metrics.csv` - Performance metrics

**Actual Results:**
- MAE: $932.45 ⭐ **BEST**
- MAPE: 1.07% ⭐ **BEST**
- Direction Accuracy: 89.66%
- Best SARIMA order: (2,1,1)×(2,0,0,7)

**When to use SARIMA:**
- Data has seasonal patterns (weekly, monthly, yearly)
- Crypto markets (weekly trading patterns)
- Better accuracy than plain ARIMA

**Why it's best here:**
- Captures 7-day weekly patterns in crypto markets
- 40% lower error than ARIMA
- Most accurate for short-term forecasting

---

### Prophet Model (`src/prophet_model.py`)

**What it does:**
- Facebook's Prophet: trend + seasonality + holidays
- Automatically detects changepoints in trend
- Models weekly AND yearly seasonality
- Provides interpretable components

**Run individually:**
```bash
python src/prophet_model.py
```

**Output:**
- `results/prophet_forecast.png` - Forecast
- `results/prophet_components.png` - Trend/seasonality breakdown
- `results/prophet_model.pkl` - Saved model
- `results/prophet_metrics.csv` - Performance metrics

**Actual Results:**
- MAE: $2,148.22
- MAPE: 2.51%
- Direction Accuracy: 96.55%

**When to use Prophet:**
- Need interpretable trend and seasonality
- Multiple seasonal patterns (daily, weekly, yearly)
- Irregular events/holidays to model
- Business presentations (clear visualizations)

**Trade-off:**
- Easier to interpret (trend, weekly, yearly components)
- Slightly lower accuracy than SARIMA for this use case

---

### Model Comparison (`src/evaluate_models.py`)

**What it does:**
- Loads metrics from all models
- Creates side-by-side comparison visualizations
- Generates summary report
- Identifies best model for each metric

**Run individually:**
```bash
python src/evaluate_models.py
```

**Output:**
- `results/model_comparison.png` - Bar charts comparing all models
- `results/summary_report.txt` - Detailed findings
- `results/all_metrics.csv` - Combined metrics

---

## 📊 Understanding the Results

### Performance Metrics Explained

**MAE (Mean Absolute Error):**
- Average dollar difference between forecast and actual
- Lower is better
- **SARIMA: $932** vs **ARIMA: $1,547** vs **Prophet: $2,148**
- Interpretation: SARIMA's forecasts are off by ~$900 on average

**RMSE (Root Mean Squared Error):**
- Similar to MAE but penalizes large errors more
- Lower is better
- SARIMA: $1,161 (best)

**MAPE (Mean Absolute Percentage Error):**
- **Industry standard** for forecasting
- Error as percentage of actual value
- Lower is better
- **SARIMA: 1.07%** (excellent for crypto)
- Interpretation: Forecasts are within ~1% of actual price

**Direction Accuracy:**
- How often the model correctly predicts price will go up/down
- Higher is better (>50% = better than random)
- **ARIMA/Prophet: 96.55%** (best)
- **SARIMA: 89.66%**
- Interpretation: Critical for trading decisions

---

## 🎯 For N26 Application & Interviews

### How to Talk About This Project

**Elevator Pitch (30 seconds):**
> "I built a time series forecasting system for Bitcoin prices using ARIMA, SARIMA, and Prophet. The key challenge was capturing weekly seasonality in crypto markets. I used auto_arima for parameter selection and proper chronological train-test splitting to prevent data leakage. SARIMA achieved 1.07% MAPE on 30-day forecasts, outperforming baseline ARIMA by 40%. This demonstrates my ability to apply statistical methods to financial time series, which is directly relevant to N26's revenue forecasting and risk management."

### Interview Deep Dive (5 minutes)

**Q: "Tell me about your time series forecasting project."**

**Your Answer:**

"I built this project to demonstrate my understanding of time series models for financial applications. Here's how I approached it:

**1. Data & Challenge:**
   - Bitcoin daily prices over 3 years
   - Challenge: Crypto markets are volatile with weekly trading patterns
   - Need accurate forecasts for 30 days ahead

**2. Proper Data Splitting:**
   - Used chronological split: first 1066 days training, last 30 days testing
   - This is critical - random splitting would leak future info and give misleading results
   - It's like backtesting a trading strategy: you can only use past data

**3. Three Models Compared:**
   
   a) **ARIMA** (Baseline):
   - Auto-selected (5,1,1) parameters with pmdarima
   - Good direction accuracy (96.55%) but higher error
   - Doesn't model seasonality
   
   b) **SARIMA** (Best):
   - Added seasonal component: (2,1,1)×(2,0,0,7)
   - The '7' captures weekly patterns in crypto markets
   - **40% lower error** than ARIMA (MAPE: 1.07% vs 1.77%)
   - Best for short-term forecasting
   
   c) **Prophet** (Interpretable):
   - Facebook's tool with trend + weekly + yearly seasonality
   - Easier to explain to non-technical stakeholders
   - Slightly lower accuracy but great visualizations

**4. Evaluation:**
   - Avoided R² (misleading for time series)
   - Used MAPE (industry standard), MAE, Direction Accuracy
   - All models beat random (50%) in direction prediction

**5. Why This Matters for N26:**
   - Revenue forecasting: same techniques apply to transaction volumes
   - Risk management: volatility forecasting for VaR
   - Seasonality: capturing weekly/monthly patterns in customer behavior
   - Proper validation: prevents overfitting in production models

**Key Takeaway:** SARIMA with weekly seasonality reduced forecast error to just 1% for a volatile asset like Bitcoin, showing I can handle financial time series with appropriate statistical methods."

---

### Specific N26 Use Cases

**1. Transaction Volume Forecasting:**
```python
# Same approach as Bitcoin forecasting
# Data: Daily transaction counts
# SARIMA: Model weekly patterns (weekday vs weekend)
# Output: 30-day forecast with confidence intervals
```

**2. Cash Flow Prediction:**
```python
# Net inflow/outflow forecasting
# Seasonality: Monthly salary payments, end-of-month bills
# Use Prophet for holiday effects
```

**3. Fraud Rate Forecasting:**
```python
# Baseline fraud rate over time
# Anomalies: Deviations from forecast
# SARIMA: Weekly patterns in fraud attempts
```

---

## 🔑 Key Technical Points to Emphasize

### 1. **Data Leakage Prevention**
```python
# ❌ WRONG: Random split
X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
# This leaks future information into training!

# ✅ CORRECT: Chronological split
split_idx = len(df) - test_size
train = df.iloc[:split_idx]  # Only past data
test = df.iloc[split_idx:]   # Future data
```

### 2. **Automated Parameter Selection**
```python
# Manual tuning is tedious and suboptimal
# Auto ARIMA finds best (p,d,q) automatically
auto_model = auto_arima(train_data, seasonal=True, m=7)
```

### 3. **Seasonality Detection**
```python
# SARIMA captures weekly patterns
# seasonal_order = (P, D, Q, s) = (2, 0, 0, 7)
# s=7 means weekly cycle
# Result: 40% error reduction vs non-seasonal ARIMA
```

### 4. **Business Metrics**
```python
# MAPE: 1.07% - within 1% of actual price
# Direction: 89.66% - critical for trading decisions
# Confidence intervals: Risk quantification
```

---

## 📈 Extending This Project

### For Deeper Technical Interviews:

**1. Add Exogenous Variables:**
```python
# SARIMAX (ARIMA with eXogenous variables)
# Add macro indicators, on-chain data
exog_vars = ['interest_rate', 'stock_market_index', 'bitcoin_hash_rate']
```

**2. Volatility Forecasting:**
```python
# GARCH models for volatility
# Critical for risk management (VaR, CVaR)
from arch import arch_model
garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
```

**3. Multi-step Forecasting:**
```python
# Forecast 7, 14, 30 days ahead
# Compare accuracy degradation over horizon
for horizon in [7, 14, 30]:
    forecast = model.forecast(steps=horizon)
```

**4. Walk-Forward Validation:**
```python
# More robust than single train-test split
# Retrain model at each step
for i in range(n_splits):
    train = data[:split_idx + i]
    test = data[split_idx + i]
    model.fit(train)
    forecast = model.predict(test)
```

---

## ✅ Pre-Application Checklist

Before pushing to GitHub and applying to N26:

- [x] Project runs successfully end-to-end
- [x] All visualizations generated
- [x] README has actual results (not placeholder numbers)
- [x] Code is clean and documented
- [x] requirements.txt includes all dependencies
- [x] .gitignore excludes unnecessary files
- [x] No personal notes or internal documents

---

## 🚀 Ready for GitHub & N26!

**Your project demonstrates:**
- ✅ Time series forecasting (ARIMA, SARIMA, Prophet - all in job description)
- ✅ Financial domain knowledge (crypto markets, volatility, seasonality)
- ✅ Proper ML practices (no data leakage, proper validation)
- ✅ Business focus (MAPE, direction accuracy, confidence intervals)
- ✅ Production-ready code (modular, documented, runnable)

**Next steps:**
1. Push to GitHub
2. Add to CV and cover letter
3. Apply to N26!
