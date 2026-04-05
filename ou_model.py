"""
ou_model.py  –  Ornstein-Uhlenbeck process tools

  dX = κ(μ - X)dt + σ dW

Functions:
  estimate()       – fit κ, μ, σ via AR(1) OLS
  z_score()        – rolling z-score of spread
  half_life()      – expected mean-reversion time in bars
  stationary_std() – σ_∞ = σ / sqrt(2κ)
  sim()            – simulate OU path (for unit testing)

Reference:
  Uhlenbeck & Ornstein (1930)
  Elliott (1994) "Optimal Trading of Mean Reverting Processes"
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd


def estimate(spread: np.ndarray) -> dict:
    """Fit OU parameters via AR(1) OLS on the spread series."""
    y, x = spread[1:], spread[:-1]
    n    = len(y)
    sx   = x.sum();  sy   = y.sum()
    sxy  = (x * y).sum();  sxx = (x * x).sum()
    a    = (n * sxy - sx * sy) / (n * sxx - sx**2)  # AR(1) slope
    b    = (sy - a * sx) / n                          # intercept

    a    = np.clip(a, -0.9999, 0.9999)               # numerical guard
    kappa_daily = -math.log(a) if a > 0 else 1e-4
    kappa_ann   = kappa_daily * 252
    mu          = b / (1 - a)
    resids      = y - (a * x + b)
    sigma_ou    = float(resids.std() * math.sqrt(252))
    hl          = math.log(2) / max(kappa_ann, 1e-4) * 252  # in days

    return dict(kappa=round(kappa_ann, 4),
                mu=round(mu, 6),
                sigma=round(sigma_ou, 6),
                half_life=round(hl, 2),
                ar1_coef=round(a, 6))


def z_score(spread: np.ndarray, window: int = 60) -> np.ndarray:
    """Rolling z-score of spread over a lookback window."""
    s    = pd.Series(spread)
    mean = s.rolling(window).mean()
    std  = s.rolling(window).std()
    return ((s - mean) / (std + 1e-10)).values


def stationary_std(kappa_ann: float, sigma_ou: float) -> float:
    """σ_∞ = σ / sqrt(2κ) – stationary standard deviation of OU process."""
    return sigma_ou / math.sqrt(max(2 * kappa_ann / 252, 1e-8))


def half_life(kappa_ann: float) -> float:
    """Expected mean-reversion half-life in days."""
    return math.log(2) / max(kappa_ann / 252, 1e-8)


def sim(n: int = 1000, kappa: float = 1.0, mu: float = 0.0,
        sigma: float = 0.01, x0: float = 0.0, seed: int = 0) -> np.ndarray:
    """Simulate an OU path (discrete Euler-Maruyama scheme)."""
    np.random.seed(seed)
    dt = 1 / 252
    x  = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = (x[i-1]
                + kappa * (mu - x[i-1]) * dt
                + sigma * math.sqrt(dt) * np.random.randn())
    return x
