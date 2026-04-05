# Statistical Arbitrage & Pairs Trading Engine
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Statsmodels](https://img.shields.io/badge/Stats-Statsmodels-informational)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
> Full institutional pairs trading system: universe screening (Engle-Granger + Johansen) → Kalman Filter dynamic hedge ratio → OU parameter estimation → XGBoost regime classifier → event-driven backtest → multi-pair portfolio with cross-correlation constraint.
---
Project Structure
```
project4_stat_arb/
├── config.py           ← Screening thresholds, KF params, entry/exit rules
├── data_gen.py         ← Cointegrated + non-cointegrated synthetic universe
├── screener.py         ← Correlation → EG → ADF → Johansen → OU → entry-z
├── kalman_filter.py    ← 2-state KF for dynamic hedge ratio β(t)
├── ou_model.py         ← OU estimation, z-score, half-life, stationary σ
├── regime.py           ← Hurst + autocorr features → GBM regime classifier
├── trader.py           ← Event-driven pair backtest engine
├── portfolio.py        ← Greedy pair selection + cross-correlation constraint
├── analytics.py        ← Sharpe, profit factor, win rate, max drawdown
├── dashboard.py        ← 8-panel stat-arb analytics dashboard
├── main.py             ← Pipeline orchestrator
└── requirements.txt
```
---
How to Run
```bash
cd project4_stat_arb
pip install -r requirements.txt
python main.py
```
Expected terminal output:
```
Universe generated: 12 series × 3,000 bars
Screening 15 pairs...
4 cointegrated pairs found.

  Pair              Corr    EG-p      β     HL(d)  entry-z
  EURUSD/GBPUSD    0.891  0.0231  0.7412   18.4    2.100
  AUDUSD/NZDUSD    0.924  0.0118  1.0981    9.2    1.850
  ...

Portfolio: 3 pairs selected (cross-corr constraint: max 0.65)

Backtesting EURUSD/GBPUSD...
  n_trades=47  win_rate=0.621  sharpe=1.43  pf=1.82  dd=-1.24%

Dashboard saved → statarb_dashboard.png
```
---
Pipeline
```
Universe prices
      │
   Screener  (screener.py)
   ├─ Pearson correlation filter  (|ρ| > 0.55)
   ├─ Engle-Granger cointegration test  (p < 0.10)
   ├─ ADF test on OLS residuals
   ├─ Johansen rank test
   ├─ OU parameter estimation  (κ, σ, half-life via AR(1) OLS)
   └─ Optimal entry z-score  (Elliott 1994 approximation)
      │
   KalmanFilter  (kalman_filter.py)
   └─ 2-state KF: β(t) updated every bar — no look-ahead drift
      │
   RegimeClassifier  (regime.py)
   └─ Features: Hurst (R/S), autocorr lag 1/5, vol ratio, |z-score|
      └─ GBM classifier: regime 1 = mean-reverting, 0 = trending
      │
   PairTrader  (trader.py)
   └─ Entry: |z| > entry_z and regime == 1
      Exit:  |z| < 0.5  OR  |z| > 4.0 (stop)  OR  t > 60 bars (time stop)
      │
   Portfolio  (portfolio.py)
   └─ Greedy selection: max 4 pairs, cross-spread corr < 0.65
      │
   Analytics + Dashboard
```
---
Cointegration Screening Logic

Step	Test	Threshold

1	Pearson correlation	|ρ| > 0.55

2	Engle-Granger ADF on residuals	p-value < 0.10

3	ADF on OLS spread	p-value < 0.10

4	OU half-life	1 ≤ HL ≤ 120 days

5	Johansen rank	computed, displayed

---
Kalman Filter — Why It Matters

A static OLS hedge ratio is estimated once and fixed for the entire backtest. In practice, pair relationships drift due to regime changes and carry shifts. The 2-state Kalman Filter updates β every single bar, capturing these drifts before they cause large losses.
```
State: [β, intercept]
Observation: p_a(t) = β(t)·p_b(t) + intercept(t) + ε
Transition:  state(t) = state(t−1) + w  (random walk prior)
```
---
Dashboard Output

`statarb_dashboard.png` — 8-panel dashboard:

Price series (both legs, dual y-axis)

Kalman dynamic hedge ratio β(t)

Kalman-filtered spread

Z-score with trade entry/exit markers (● entry, × exit)

Regime filter overlay (mean-reverting blue / trending red + Hurst line)

Equity curve with green/red fill

Trade P&L distribution (winners vs losers)

Portfolio summary table (all pairs: Sharpe, win rate, profit factor, drawdown)

---
References

Engle & Granger (1987). Co-integration and Error Correction. Econometrica.

Johansen (1988). Statistical Analysis of Cointegration Vectors. J. Econ. Dyn. Control.

Kalman (1960). A New Approach to Linear Filtering and Prediction Problems. J. Basic Eng.

Elliott (1994). Optimal Trading of Mean Reverting Processes. Stanford Technical Report.

Pole (2007). Statistical Arbitrage: Algorithmic Trading Insights. Wiley.

---
Requirements
```
numpy>=1.26
pandas>=2.1
scipy>=1.11
statsmodels>=0.14
scikit-learn>=1.4
matplotlib>=3.8
```
