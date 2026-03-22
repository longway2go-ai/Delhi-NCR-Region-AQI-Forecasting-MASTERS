# 🌫️ Air Quality Index (AQI) Forecasting — Delhi/NCR Region
### A Comprehensive Time Series Analysis Using SARIMA, SARIMAX, VECM, GARCH & Multivariate LSTM

## 📌 Project Overview

This project performs end-to-end time series forecasting of Air Quality Index (AQI) across **5 monitoring stations in the Delhi/NCR region** (India) using classical statistical models, multivariate econometric models, and deep learning — with rigorous diagnostic testing at every stage.

The project covers the full pipeline from raw data preprocessing and stationarity testing through to model evaluation and a final comparative forecast for **December 2025**.

### Stations Analysed

| Station | City |
|---------|------|
| NSIT Dwarka, Delhi | Delhi |
| Mandir Marg, Delhi | Delhi |
| Dwarka Sec 8, Delhi | Delhi |
| Greater Noida | Noida |
| Siri Fort, Delhi | Delhi |

---

## 🏗️ Project Architecture

```
AQI Forecasting Pipeline
│
├── 1. Data Preprocessing
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
    ├── Rolling Window Validation (4-day horizon)
    ├── MAE, RMSE, MAPE, SMAPE, R², CI Coverage
    └── December 2025 Final Comparison Plot
```

---

## 📁 Repository Structure

```
aqi-forecasting/
│
├── data/
│   └── aqi_delhi_ncr.csv           # Raw dataset (2020–2026, 4 readings/day)
│
├── notebooks/
│   ├── S1-NSIT Dwarka_Delhi.ipynb  
│   ├── S2-Mandir Marg_Delhi.ipynb     
│   ├── S3-Greater_Noida.ipynb     
│   ├── S4-Dwarka_Sec8_Delhi.ipynb           
│   ├── EDA.ipynb            
│
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Data Preprocessing

**Feature Engineering:**
- PCA applied to correlated pollutants (PM2.5, PM10, CO, NO₂, SO₂) — PC1 captured **97% of combined variance**, resolving severe multicollinearity (VIF up to 197)
- Feature selection using three criteria jointly: correlation with AQI (|r| > 0.3), VIF (< 10), and Granger causality (p < 0.05)
- Final exogenous features selected: `pollution_pc1`, `visibility`, `o3`, `wind_speed`

### 2. Stationarity Testing

All variables tested using ADF and KPSS tests jointly. AQI was borderline stationary (ADF p=0.04), confirming `d=0` for ARIMA/SARIMA. Temperature was non-stationary and first-differenced before use in VECM.

### 3. SARIMA + GARCH

- Best SARIMA order: **(2, 0, 2)(2, 0, 2, 4)** — seasonal period s=4 captures the intraday cycle (4 readings/day)
- GARCH residual diagnostics confirmed strong ARCH effects (p=0.0000) — volatility clustering present in residuals
- Best GARCH: **(2, 2) with Skewed Student-t distribution** justified by kurtosis ~8 and slight positive skew in residuals
- Post-GARCH diagnostics: ARCH p=1.00, Ljung-Box p=1.00 — residuals fully cleaned

### 4. Cointegration & VECM

Johansen cointegration test confirmed **6 cointegrating relationships** among 7 variables — making VECM the statistically correct multivariate model over plain VAR. Optimal lag order selected by AIC: k=8.

### 5. Multivariate LSTM

```
Architecture:
Input → LSTM(128, return_sequences=True) → Dropout(0.2)
      → LSTM(64) → Dropout(0.2)
      → Dense(32, relu) → Dense(1)

Optimizer  : Adam
Loss       : Huber (robust to outliers)
Lookback   : 32 steps (8 days)
Features   : visibility, pollution_pc1, o3, wind_speed, humidity, temperature
Train/Test : Jan 2020 – Dec 2023 / Jan 2024 – Dec 2025
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
import numpy as np
from src.preprocessing import apply_pca
from src.models.sarima_garch import sarima_grid_search
from src.models.lstm import train_evaluate_lstm

# Load data
df = pd.read_csv('data/aqi_delhi_ncr.csv',
                 parse_dates=['datetime'],
                 index_col='datetime')

# Apply PCA to compress correlated pollutants
df = apply_pca(df, cols=['pm25', 'pm10', 'co', 'no2', 'so2'])

# Run SARIMA grid search
results = sarima_grid_search(
    df,
    station_name   = 'NSIT Dwarka, Delhi',
    p_range        = [1], d_range = [0], q_range = [1],
    P_range        = range(0, 3), D_range = [0], Q_range = range(0, 3),
    s              = 4
)

# Train Multivariate LSTM
lstm_result = train_evaluate_lstm(
    df,
    station_name = 'NSIT Dwarka, Delhi',
    feature_cols = ['visibility', 'pollution_pc1', 'o3',
                    'wind_speed', 'humidity', 'temperature'],
    split_date   = '2024-01-01'
)
```

---

## 📈 Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle |
| Time Period | January 2020 – December 2025 |
| Frequency | 4 readings/day (06:00, 12:00, 18:00, 23:00) |
| Stations | 5 Delhi/NCR stations |
| Features | AQI, PM2.5, PM10, NO₂, SO₂, CO, O₃, Temperature, Humidity, Wind Speed, Visibility |
| Total Observations | 8,768 per station (43,840 total) |

---

## 🔍 Key Findings

**1. Log transform was not needed**
Despite widespread heteroskedasticity (Breusch-Pagan p≈0), AQI skewness was near zero across all stations (max 0.31). The heteroskedasticity was structural — driven by seasonal pollution cycles — and required GARCH rather than log transformation.

**2. SARIMA significantly outperforms ARIMA**
The intraday seasonal cycle (s=4) is strong enough that plain ARIMA captures it accidentally through near-unit-root AR coefficients, causing unrealistic oscillating forecasts. SARIMA explicitly models this seasonal structure, improving AIC by ~116 points and producing cleaner forecasts.

**3. GARCH improves uncertainty quantification, not point forecasts**
GARCH does not change MAE/RMSE — it improves CI coverage by modelling volatility clustering. This is critical for public health decisions: models can now provide narrow confidence bands during calm summer periods and wide bands during volatile winter smog episodes.

**4. VECM was required over VAR**
Johansen cointegration test revealed 6 cointegrating relationships among 7 variables — ignoring these (as VAR does) yields biased and inconsistent estimates. VECM's error correction coefficient for AQI (α = -0.0845) implies 8.45% of any disequilibrium corrects per 6-hour period, with full correction in approximately 3 days.

**5. LSTM captures non-linear regime behaviour**
During the extreme winter smog period (December 2025), LSTM outperformed all statistical models by learning non-linear, regime-switching relationships from multivariate features — behaviour that linear models like SARIMA and VECM inherently cannot capture.

---

## 📉 Limitations

- SARIMAX underperformed relative to SARIMA due to the challenge of obtaining accurate future exogenous values; lag-4 features partially addressed this
- LSTM performance is sensitive to hyperparameters and may require retuning for different stations or time periods
- Statistical models (SARIMA, VECM) trained on typical conditions may underperform during structurally different extreme pollution episodes

---

## 🔭 Future Work

- [ ] Extend analysis to all 23 Delhi/NCR monitoring stations
- [ ] Implement SARIMA-LSTM hybrid model for improved accuracy
- [ ] Add real-time forecasting pipeline using OpenAQ API
- [ ] Deploy as a web dashboard with Streamlit
- [ ] Incorporate meteorological forecast data (IMD) as exogenous inputs for SARIMAX
- [ ] Explore Transformer-based architectures for long-horizon forecasting

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
- Institution: [Banaras Hindu University]
- Contact: [arnab.bhu.stcomp@gmail.com]

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built as part of a research project on urban air quality forecasting in Delhi/NCR, India.</i>
</p>
