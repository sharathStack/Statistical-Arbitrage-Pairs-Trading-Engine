"""
data_gen.py  –  Synthetic cointegrated universe generator

Produces a DataFrame of price series where:
  - Some pairs share a common stochastic trend → cointegrated
  - Others follow independent random walks → not cointegrated
Each cointegrated pair has a known OU spread with a configurable half-life.
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd

import config

# (sym_a, sym_b, is_cointegrated, beta, sigma, ou_speed)
UNIVERSE_SPEC = [
    ("EURUSD", "GBPUSD", True,  0.75, 0.0015, 0.60),
    ("AUDUSD", "NZDUSD", True,  1.10, 0.0018, 1.20),
    ("USDCAD", "USDCHF", True,  0.85, 0.0014, 0.35),
    ("USDJPY", "EURJPY", False, 0.90, 0.0030, 0.00),
    ("XAUUSD", "XAGUSD", True,  0.06, 0.0040, 1.80),
    ("AUDNZD", "CADJPY", False, 1.00, 0.0025, 0.00),
]


def generate() -> pd.DataFrame:
    np.random.seed(config.DATA_SEED)
    n      = config.N_BARS
    prices = {}

    for sym_a, sym_b, coint, beta, sigma, ou_k in UNIVERSE_SPEC:
        common   = np.cumsum(np.random.normal(0, 0.001, n))
        noise_a  = np.cumsum(np.random.normal(0, sigma * 0.45, n))
        noise_b  = np.cumsum(np.random.normal(0, sigma * 0.45, n))

        if coint:
            # OU spread layered onto common trend
            spread   = np.zeros(n)
            ou_shock = np.random.normal(0, sigma * 0.25, n)
            for i in range(1, n):
                spread[i] = spread[i-1] * (1 - ou_k / 252) + ou_shock[i]
            prices[sym_a] = np.exp(1.20 + common + noise_a * 0.25)
            prices[sym_b] = np.exp(1.20 + common * beta + spread + noise_b * 0.25)
        else:
            prices[sym_a] = np.exp(1.20 + common + noise_a)
            prices[sym_b] = np.exp(1.20 + common * beta + noise_b)

    df = pd.DataFrame(prices)
    print(f"Universe generated: {len(df.columns)} series × {len(df):,} bars")
    return df
