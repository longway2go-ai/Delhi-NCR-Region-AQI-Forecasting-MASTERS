# 🌫️ Air Quality Index (AQI) Forecasting — Delhi/NCR Region
### A Comprehensive Time Series Analysis Using SARIMA, SARIMAX, VECM, GARCH & Multivariate LSTM

## 📌 Project Overview

This project performs end-to-end time series forecasting of Air Quality Index (AQI) across **5 monitoring stations in the Delhi/NCR region** (India) using classical statistical models, multivariate econometric models, and deep learning — with rigorous diagnostic testing at every stage.

The project covers the full pipeline from raw data preprocessing and stationarity testing through to model evaluation and a final comparative forecast for **December 2025**.

### Stations Analysed
| Station | City | % Censored (Original) |
|---------|------|----------------------|
| NSIT Dwarka, Delhi | Delhi | 16.58% |
| Mandir Marg, Delhi | Delhi | 17.36% |
| Dwarka Sec 8, Delhi | Delhi | 17.71% |
| Greater Noida | Noida | 17.80% |
| Siri Fort, Delhi | Delhi | 18.29% |

---

## 🏗️ Project Architecture

```
AQI Forecasting Pipeline
│
├── 1. Data Preprocessing
│   ├── Censoring Detection & Treatment (AQI capped at 500)
│   ├── Skewness Analysis & Log Transform Decision
│   ├── PCA for Multicollinear Pollutant Features
│   └── Feature Selection (Correlation + VIF + Granger Causality)
│
├── 2. Stationarity & Diagnostic Testing
│   ├── ADF Test (Augmented Dickey-Fuller)
│   ├── KPSS Test
│   ├── Ljung-Box Test (Autocorrelation)
│   ├── ARCH Test (Volatility Clustering)
│   └── Breusch-Pagan Test (Heteroskedasticity)
│
├── 3. Univariate Modelling
│   ├── ARIMA Grid Search
│   ├── SARIMA Grid Search (seasonal period s=4)
│   └── SARIMA + GARCH(2,2)-skewt
│
├── 4. Multivariate Modelling
│   ├── SARIMAX + GARCH (with exogenous features)
│   ├── Johansen Cointegration Test
│   ├── VECM (Vector Error Correction Model)
│   └── VECM + GARCH(2,2)-skewt
│
├── 5. Deep Learning
│   └── Multivariate LSTM (2-layer, with Dropout)
│
└── 6. Evaluation & Comparison
    ├── Rolling Window Validation (4-day horizon, 182 windows)
    ├── MAE, RMSE, MAPE, SMAPE, R², CI Coverage
    └── December 2025 Final Comparison Plot
```

---

## 📊 Key Results

### Rolling Window Evaluation (January 2024 – December 2025)

| Model | MAE | RMSE | MAPE | R² | CI Coverage |
|-------|-----|------|------|-----|-------------|
| SARIMA + GARCH | 34.57 | 52.66 | 21.08% | 0.9213 | 92.75% ✅ |
| VECM + GARCH | 35.85 | 52.11 | 22.78% | **0.9230** | 92.27% ✅ |
| SARIMAX + GARCH | 41.33 | 86.53 | 18.94% | 0.7876 | 89.59% ⚠️ |

### December 2025 Final Forecast Performance

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|----|
| **LSTM** ⭐ | **67.05** | **77.19** | **14.18%** | **0.2537** |
| VECM + GARCH | 85.94 | 99.76 | 20.61% | -0.2465 |
| SARIMA + GARCH | 90.85 | 105.55 | 21.71% | -0.3953 |
| SARIMAX + GARCH | 187.04 | 204.12 | 38.21% | -4.2187 |

> **Key Finding:** LSTM outperformed all statistical models on the extreme winter smog period (December 2025) by capturing non-linear multivariate relationships. Statistical models excelled during typical conditions (R²=0.92) but struggled with the structurally different winter regime.

---

## 📁 Repository Structure

```
aqi-forecasting/
│
├── data/
│   └── aqi_delhi_ncr.csv           # Raw dataset (2020–2026, 4 readings/day)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb      # Data cleaning, censoring treatment, PCA
│   ├── 02_diagnostics.ipynb        # Stationarity, heteroskedasticity tests
│   ├── 03_arima_sarima.ipynb       # ARIMA/SARIMA grid search and fitting
│   ├── 04_garch.ipynb              # GARCH grid search and fitting
│   ├── 05_sarimax.ipynb            # SARIMAX with feature selection
│   ├── 06_vecm.ipynb               # Cointegration tests and VECM
│   └── 07_lstm.ipynb               # Multivariate LSTM training and evaluation
│
├── src/
│   ├── preprocessing.py            # Data cleaning and transformation functions
│   ├── diagnostics.py              # All statistical tests
│   ├── models/
│   │   ├── sarima_garch.py         # SARIMA + GARCH functions
│   │   ├── sarimax_garch.py        # SARIMAX + GARCH functions
│   │   ├── vecm_garch.py           # VECM + GARCH functions
│   │   └── lstm.py                 # LSTM model and training
│   └── evaluation.py               # Rolling validation and metrics
│
├── outputs/
│   └── december_2025_comparison.png  # Final comparison plot
│
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Data Preprocessing

**Censoring Problem:** The raw AQI data was censored at 500 (the index ceiling) — up to 30% of readings across stations were recorded as exactly 500, with consecutive runs of up to 189 readings (47 days) during winter smog seasons. Interpolation was not viable for runs this long.

**Solution:** Censored values (AQI=500) were replaced with random values drawn from U(550, 600) using a fixed seed (seed=42) for reproducibility, reflecting that true pollution exceeded the measurable threshold during these periods.

**Feature Engineering:**
- PCA applied to correlated pollutants (PM2.5, PM10, CO, NO₂, SO₂) — PC1 captured **97% of variance**
- Feature selection using correlation (|r| > 0.3), VIF (< 10), and Granger causality (p < 0.05)
- Final exogenous features: `pollution_pc1`, `visibility`, `o3`, `wind_speed`

### 2. Stationarity Testing

All variables tested using ADF and KPSS tests. AQI was borderline stationary (ADF p=0.04), confirming `d=0` for ARIMA/SARIMA. Temperature was non-stationary and differenced once before use in VECM.

### 3. SARIMA + GARCH

- Best SARIMA order: **(2, 0, 2)(2, 0, 2, 4)** — seasonal period s=4 captures intraday cycle
- GARCH residual diagnostics confirmed ARCH effects (p=0.0000) — volatility clustering present
- Best GARCH: **(2, 2) with Skewed Student-t distribution** (kurtosis ~8, skewness ~0.3)
- Post-GARCH diagnostics: ARCH p=1.00, Ljung-Box p=1.00 — residuals fully clean

### 4. Cointegration & VECM

Johansen test confirmed **6 cointegrating relationships** among 7 variables — making VECM the statistically correct multivariate model over VAR. Optimal lag order selected by AIC: k=8.

### 5. Multivariate LSTM

```
Architecture:
Input → LSTM(128, return_sequences=True) → Dropout(0.2)
      → LSTM(64) → Dropout(0.2)
      → Dense(32, relu) → Dense(1)

Optimizer : Adam
Loss      : Huber (robust to outliers)
Lookback  : 32 steps (8 days)
Features  : visibility, pollution_pc1, o3, wind_speed, humidity, temperature
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/aqi-forecasting-delhi.git
cd aqi-forecasting-delhi

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
arch>=6.2.0
tensorflow>=2.13.0
scipy>=1.11.0
```

---

## 🚀 Quick Start

```python
import pandas as pd
from src.preprocessing import prepare_data, apply_pca
from src.models.sarima_garch import sarima_grid_search, fit_sarima_and_test_arch
from src.models.lstm import train_evaluate_lstm

# Load data
df = pd.read_csv('data/aqi_delhi_ncr.csv', parse_dates=['datetime'], index_col='datetime')

# Preprocess — fix censoring
df.loc[df['aqi'] == 500, 'aqi'] = np.random.RandomState(42).randint(550, 601,
                                    size=(df['aqi'] == 500).sum())

# Apply PCA for pollutant features
df = apply_pca(df, cols=['pm25', 'pm10', 'co', 'no2', 'so2'])

# Run SARIMA grid search
results = sarima_grid_search(
    df,
    station_name   = 'NSIT Dwarka, Delhi',
    p_range        = [1], d_range = [0], q_range = [1],
    P_range        = range(0, 3), D_range = [0], Q_range = range(0, 3),
    s              = 4
)

# Train LSTM
lstm_result = train_evaluate_lstm(
    df,
    station_name = 'NSIT Dwarka, Delhi',
    feature_cols = ['visibility', 'pollution_pc1', 'o3', 'wind_speed', 'humidity', 'temperature'],
    split_date   = '2024-01-01'
)
```

---

## 📈 Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle (originally scraped from CPCB) |
| Time Period | January 2020 – December 2025 |
| Frequency | 4 readings/day (06:00, 12:00, 18:00, 23:00) |
| Stations | 5 Delhi/NCR stations |
| Features | AQI, PM2.5, PM10, NO₂, SO₂, CO, O₃, Temperature, Humidity, Wind Speed, Visibility |
| Total Observations | 8,768 per station (43,840 total) |

---

## 🔍 Key Findings

**1. Censoring is a fundamental challenge in AQI data**
Raw AQI data is capped at 500 by the index definition. Up to 30% of Delhi readings hit this ceiling during winter smog, causing models trained on raw data to produce oscillating forecasts.

**2. Log transform was not needed**
Despite widespread heteroskedasticity (Breusch-Pagan p≈0), AQI skewness was near zero across all stations (max 0.31). The heteroskedasticity was structural — driven by seasonal pollution cycles — and required GARCH rather than log transformation.

**3. SARIMA significantly outperforms ARIMA**
The intraday seasonal cycle (s=4) is strong enough that plain ARIMA accidentally captures it through near-unit-root AR coefficients, causing unrealistic oscillating forecasts. SARIMA reduced AIC by ~116 points.

**4. GARCH improves uncertainty quantification, not point forecasts**
GARCH does not change MAE/RMSE — it improves CI coverage from NaN (no intervals) to 92.3% by modelling volatility clustering. This matters for public health decisions: "AQI will be 350 ± 20 today" vs. "AQI will be 350 ± 180 during smog season."

**5. VECM was required over VAR**
Johansen cointegration test showed 6 cointegrating relationships among 7 variables — ignoring these (as VAR does) would yield biased estimates. VECM's error correction coefficient for AQI (α = -0.0845) implies 8.45% of any disequilibrium corrects per 6-hour period (~3 days for full correction).

**6. LSTM wins on extreme events**
During December 2025 (intense smog season), LSTM (MAE=67) outperformed all statistical models (best: VECM+GARCH MAE=86) by learning non-linear, regime-switching behaviour from multivariate features that linear models cannot capture.

---

## 📉 Limitations

- AQI censoring at 500 was addressed by random imputation — not statistically ideal; Tobit regression would be more rigorous
- SARIMAX underperformed due to the need for future exogenous values; lag-4 features partially addressed this
- LSTM performance is sensitive to hyperparameters and may not generalise across all stations without retuning
- December 2025 represents an out-of-distribution period for statistical models trained primarily on pre-2024 data

---

## 🔭 Future Work

- [ ] Apply Tobit regression to properly handle censored AQI values
- [ ] Extend to all 23 Delhi/NCR stations
- [ ] Implement SARIMA-LSTM hybrid model
- [ ] Add real-time forecasting pipeline using OpenAQ API
- [ ] Deploy as a web dashboard with Streamlit
- [ ] Incorporate meteorological forecast data (IMD) as exogenous inputs for SARIMAX

---

## 📚 References

- Box, G.E.P., Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control*
- Engle, R.F. (1982). Autoregressive Conditional Heteroskedasticity. *Econometrica*
- Johansen, S. (1991). Estimation and Hypothesis Testing of Cointegration Vectors. *Econometrica*
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*
- Hyndman, R.J., Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.)
- Central Pollution Control Board (CPCB), India — AQI Methodology

---

## 👤 Author

**Koushik**
- Project: AQI Forecasting — Delhi/NCR
- Institution: [Your Institution]
- Contact: [your.email@example.com]

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built as part of a research project on urban air quality forecasting in Delhi/NCR, India.</i>
</p>
