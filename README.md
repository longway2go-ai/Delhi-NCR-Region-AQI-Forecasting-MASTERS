# 🌫️ Air Quality Index (AQI) Forecasting — Delhi/NCR Region
### A Comprehensive Time Series Analysis Using SARIMA, SARIMAX, VAR, ARDL, GARCH & Multivariate LSTM

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
│   ├── ARIMA Grid Search (manual p, d, q via AIC/BIC heatmaps)
│   ├── SARIMA Grid Search (seasonal period s=4)
│   └── SARIMA + GARCH(2,2)-skewt
│
├── 4. Multivariate Modelling
│   ├── SARIMAX + GARCH (with exogenous features)
│   ├── VAR — Vector Autoregression
│   │   ├── Temperature first-differenced to I(0)
│   │   ├── Lag selection via AIC / BIC / HQIC grid
│   │   ├── Impulse Response Functions (IRF)
│   │   └── Forecast Error Variance Decomposition (FEVD)
│   ├── ARDL — Autoregressive Distributed Lag (Pesaran et al. 2001)
│   │   ├── Mixed I(0)/I(1) variables — no pre-classification needed
│   │   ├── Lag selection via AIC / BIC grid (manual p heatmap)
│   │   ├── PSS Bounds Test for long-run relationship
│   │   └── Error Correction Model (ECM) if bounds test passes
│   └── ARCH Test + GARCH(2,2)-skewt on VAR & ARDL residuals
│
├── 5. Deep Learning
│   └── Multivariate LSTM (2-layer, with Dropout)
│
└── 6. Evaluation & Comparison
    ├── Rolling Window Validation (4-step horizon)
    ├── MAE, RMSE, MAPE, SMAPE, R², CI Coverage
    ├── Diebold-Mariano Test (pairwise forecast accuracy)
    └── December 2025 Final Comparison Plot
```

---

## 📁 Repository Structure

```
aqi-forecasting/
│
├── data/
│   └── aqi_delhi_ncr.csv                # Raw dataset (2020–2026, 4 readings/day)
│
├── notebooks/
│   ├── EDA.ipynb                        # Exploratory data analysis
│   ├── S1-NSIT_Dwarka_Delhi.ipynb
│   ├── S2-Mandir_Marg_Delhi.ipynb
│   ├── S3-Greater_Noida.ipynb
│   ├── S4-Dwarka_Sec8_Delhi.ipynb
│   └── S5-Siri_Fort_Delhi.ipynb
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
- Final exogenous features selected: `pollution_pc1`, `visibility`, `o3`, `wind_speed`, `temperature`, `humidity`

### 2. Stationarity Testing

All variables tested using ADF and KPSS jointly. AQI was borderline stationary (ADF p = 0.048), confirming `d = 0` for ARIMA/SARIMA. Temperature was I(1) and handled differently per model — first-differenced before VAR, used at levels in ARDL (which accepts mixed integration orders).

### 3. SARIMA + GARCH

- Parameters selected **manually** via AIC/BIC grid search — all candidate (p, d, q) and (P, D, Q) combinations printed as tables and visualised as heatmaps; the order where both AIC and BIC are simultaneously minimised is selected
- Best SARIMA order: **(2, 0, 2)(2, 0, 2, 4)** — seasonal period s=4 captures the intraday cycle (4 readings/day)
- GARCH residual diagnostics confirmed strong ARCH effects (p = 0.0000) — volatility clustering present in residuals
- Best GARCH: **(2, 2) with Skewed Student-t distribution** — justified by kurtosis ≈ 8 and slight positive skew in residuals
- Post-GARCH diagnostics: ARCH p = 1.00, Ljung-Box p = 1.00 — residuals fully cleaned

### 4. Multivariate Modelling — VAR and ARDL

**Why not VECM?**
The Johansen cointegration test was run on the 7-variable system but yielded extremely large trace statistics (e.g. 13,870 vs critical value 125). This is a well-known spurious result: Johansen requires all variables to be I(1), and when run on a mixed I(0)/I(1) system with a large sample it almost always rejects, making the cointegration conclusion meaningless. VECM was therefore not used.

**Model selection logic:**

| Scenario | Model Used |
|---|---|
| All variables I(0) | VAR |
| All variables I(1) + cointegrated | VECM (not applicable here) |
| Mix of I(0) and I(1) — none I(2) | ARDL |
| ARCH effects in residuals | GARCH on residuals |

**VAR (Vector Autoregression)**
- Temperature first-differenced to achieve stationarity before fitting
- Lag order p selected via AIC/BIC/HQIC grid — criteria plotted as line charts; BIC minimum used as the parsimonious choice
- Portmanteau whiteness test and Jarque-Bera normality test run on residuals
- IRF traces dynamic responses to shocks across all variables
- FEVD decomposes forecast error variance by source variable

**ARDL (Pesaran, Shin & Smith 2001)**
- Accepts mixed I(0)/I(1) variables without pre-classification — the correct choice given temperature is I(1) while all other variables are I(0)
- Lag p selected manually via AIC/BIC heatmap; exog lags selected via stepwise `ardl_select_order`
- PSS Bounds Test: F-statistic compared to I(0) lower bound and I(1) upper bound at 1%, 5%, 10% significance
- If bounds test confirms a long-run relationship: ECM fitted with ECT coefficient (α < 0 and significant confirms stable equilibrium); speed of adjustment and shock half-life reported
- CUSUM stability test confirms structural stability of coefficients

**GARCH on Residuals**
- Engle (1982) ARCH LM test applied to both VAR and ARDL residuals
- If ARCH effects detected: GARCH(p,q) grid searched with order selection via AIC/BIC heatmap; EGARCH and GJR-GARCH also tested for asymmetric effects
- Best GARCH: **(2, 2) with Skewed Student-t** — consistent with SARIMA findings across stations
- Standardised residual diagnostics confirm ARCH removal post-GARCH

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

# Load data
df = pd.read_csv('data/aqi_delhi_ncr.csv',
                 parse_dates=['datetime'],
                 index_col='datetime')

# --- SARIMA (manual grid) ---
# AIC/BIC heatmaps printed for each (p,d,q) and (P,D,Q) combination
# See notebooks/S1-NSIT_Dwarka_Delhi.ipynb

# --- VAR ---
# run_var(data, var_cols, target='aqi', max_lags=14)
# Temperature auto-differenced; lag selected via BIC grid

# --- ARDL + Bounds Test ---
# run_ardl(data, dependent='aqi', regressors=[...], max_lag=8)
# PSS bounds test + ECM if long-run relationship confirmed

# --- ARCH / GARCH ---
# run_arch_garch(residuals, garch_orders=[(1,1),(1,2),(2,1),(2,2)])
# ARCH LM test → GARCH grid → conditional volatility plot
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
Despite widespread heteroskedasticity (Breusch-Pagan p ≈ 0), AQI skewness was near zero across all stations (max 0.31). The heteroskedasticity was structural — driven by seasonal pollution cycles — and required GARCH rather than log transformation.

**2. SARIMA significantly outperforms ARIMA**
The intraday seasonal cycle (s = 4) is strong enough that plain ARIMA captures it accidentally through near-unit-root AR coefficients, causing unrealistic oscillating forecasts. SARIMA explicitly models this seasonal structure, improving AIC by ~116 points and producing cleaner forecasts.

**3. GARCH improves uncertainty quantification, not point forecasts**
GARCH does not change MAE/RMSE — it improves CI coverage by modelling volatility clustering. This is critical for public health decisions: models now provide narrow confidence bands during calm summer periods and wide bands during volatile winter smog episodes.

**4. VAR vs ARDL — different strengths**
VAR captures symmetric short-run interdependencies and allows IRF/FEVD analysis of how shocks propagate across variables. ARDL is more appropriate for point forecasting of AQI specifically — it has a clear dependent variable, handles the mixed I(0)/I(1) system without differencing all variables, and confirms a long-run equilibrium through the Bounds Test. The ARDL ECT coefficient (α ≈ −0.085 at NSIT Dwarka) implies approximately 8.5% of any disequilibrium corrects per 6-hour period, with full correction in roughly 3 days.

**5. VECM was not applicable**
The Johansen test was carried out but yielded spurious results (trace statistics far exceeding critical values) due to the mixed integration order of the variable set. ARDL's bounds test is the statistically correct long-run test for this data structure.

**6. LSTM captures non-linear regime behaviour**
During the extreme winter smog period (December 2025), LSTM outperformed all statistical models by learning non-linear, regime-switching relationships from multivariate features — behaviour that linear models like SARIMA and VAR inherently cannot capture.

---

## 📉 Limitations

- SARIMAX underperformed relative to SARIMA due to the challenge of obtaining accurate future exogenous values; lag-4 features partially addressed this
- VAR requires all variables to be stationary — differencing temperature before VAR means the model captures changes in temperature rather than temperature levels, which may lose some interpretability
- ARDL long-run coefficients are sensitive to lag selection; BIC-based parsimony was prioritised to avoid overfitting
- LSTM performance is sensitive to hyperparameters and may require retuning for different stations or time periods
- Statistical models (SARIMA, ARDL) trained on typical conditions may underperform during structurally different extreme pollution episodes

---

## 🔭 Future Work

- [ ] Extend analysis to all 23 Delhi/NCR monitoring stations
- [ ] Implement SARIMA-LSTM hybrid model for improved accuracy
- [ ] Add real-time forecasting pipeline using OpenAQ API
- [ ] Deploy as a web dashboard with Streamlit
- [ ] Incorporate meteorological forecast data (IMD) as exogenous inputs for SARIMAX
- [ ] Explore ARDL-GARCH joint estimation for simultaneous mean and variance modelling
- [ ] Explore Transformer-based architectures for long-horizon forecasting

---

## 📚 References

- Box, G.E.P., Jenkins, G.M. (1976). *Time Series Analysis: Forecasting and Control*
- Engle, R.F. (1982). Autoregressive Conditional Heteroskedasticity. *Econometrica*
- Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics*
- Pesaran, M.H., Shin, Y., Smith, R.J. (2001). Bounds testing approaches to the analysis of level relationships. *Journal of Applied Econometrics*
- Sims, C.A. (1980). Macroeconomics and Reality. *Econometrica* — foundational VAR paper
- Diebold, F.X., Mariano, R.S. (1995). Comparing Predictive Accuracy. *Journal of Business & Economic Statistics*
- Hyndman, R.J., Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.)
- Central Pollution Control Board (CPCB), India — AQI Methodology

---

## 👤 Author

**Arnab**
- Project: AQI Forecasting — Delhi/NCR
- Institution: Banaras Hindu University
- Contact: arnab.bhu.stcomp@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <i>Built as part of a research project on urban air quality forecasting in Delhi/NCR, India.</i>
</p>
