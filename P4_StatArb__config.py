"""
config.py  –  Statistical Arbitrage & Pairs Trading parameters
"""

# ── Universe ──────────────────────────────────────────────────────────────────
N_BARS       = 3000          # bars per synthetic price series
DATA_SEED    = 42

# ── Cointegration Screening ───────────────────────────────────────────────────
CORR_THRESHOLD  = 0.55       # minimum |Pearson ρ| to proceed
EG_PVALUE       = 0.10       # Engle-Granger ADF p-value threshold
MIN_HALF_LIFE   = 1          # days
MAX_HALF_LIFE   = 120        # days

# ── Kalman Filter ─────────────────────────────────────────────────────────────
KF_DELTA   = 5e-5            # process noise scaling
KF_R       = 0.001           # observation noise variance
KF_WARMUP  = 100             # bars before quoting spread z-scores

# ── Ornstein-Uhlenbeck / Entry-Exit ──────────────────────────────────────────
Z_EXIT     = 0.50            # z-score at which spread is considered mean-reverted
Z_STOP     = 4.00            # z-score stop-loss
MAX_HOLDING = 60             # maximum holding period (bars)

# ── Regime Filter (GBM / XGBoost) ────────────────────────────────────────────
REGIME_WINDOW    = 60        # rolling window for regime features
REGIME_HURST_THR = 0.50      # Hurst < 0.50 → mean-reverting regime

# ── Portfolio ─────────────────────────────────────────────────────────────────
MAX_PAIRS          = 4       # maximum concurrent pairs in book
MAX_CROSS_CORR     = 0.65    # maximum allowed spread–spread correlation

# ── Transaction costs ─────────────────────────────────────────────────────────
TRANSACTION_COST = 0.0002    # one-way; applied on each leg

# ── Output ────────────────────────────────────────────────────────────────────
CHART_OUTPUT = "statarb_dashboard.png"
CHART_DPI    = 150
