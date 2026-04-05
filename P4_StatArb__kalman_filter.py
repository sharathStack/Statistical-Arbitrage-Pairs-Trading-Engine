"""
kalman_filter.py  –  Dynamic hedge ratio estimation via Kalman Filter

State vector: [β, intercept]  (2-dimensional)
Observation:  p_a(t) = β(t)·p_b(t) + intercept(t) + ε

The hedge ratio β is time-varying — this is critical for pairs whose
relationship drifts over time (regime changes, carry shifts, etc.)

Reference:
  Kalman (1960) "A New Approach to Linear Filtering and Prediction"
  Pole (2007) "Statistical Arbitrage: Algorithmic Trading Insights"
"""
from __future__ import annotations
import numpy as np
import config


class KalmanFilter:
    """
    2-state Kalman filter for dynamic linear regression.
    P, Q, R follow standard KF notation.
    """

    def __init__(self):
        self.delta = config.KF_DELTA
        self.R     = config.KF_R
        # State: [beta, intercept]
        self.x = np.zeros(2)                 # initial state estimate
        self.P = np.eye(2) * 1.0             # initial error covariance

    def update(self, pa: float, pb: float) -> tuple[float, float, float]:
        """
        Process one observation (pa, pb).

        Returns:
            beta        – current dynamic hedge ratio
            spread      – innovation (pa − predicted pa)
            innov_var   – innovation variance S (useful for z-score normalisation)
        """
        F   = np.array([pb, 1.0])            # observation row vector
        Q   = (self.delta / (1 - self.delta)) * np.eye(2)

        # ── Predict ────────────────────────────────────────────────────────
        P_pred = self.P + Q

        # ── Innovation ─────────────────────────────────────────────────────
        y_hat = float(F @ self.x)
        innov = pa - y_hat
        S     = float(F @ P_pred @ F) + self.R

        # ── Kalman gain ────────────────────────────────────────────────────
        K     = P_pred @ F / S

        # ── Update ─────────────────────────────────────────────────────────
        self.x += K * innov
        self.P  = (np.eye(2) - np.outer(K, F)) @ P_pred

        return float(self.x[0]), innov, S

    def run_series(self, pa: np.ndarray,
                   pb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process full price series and return arrays of:
          betas   – dynamic hedge ratio at each bar
          spreads – innovation (residual) at each bar
          s_arr   – innovation variance
        """
        n = len(pa)
        betas, spreads, s_arr = np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            beta, spread, S = self.update(pa[i], pb[i])
            betas[i]   = beta
            spreads[i] = spread
            s_arr[i]   = S
        return betas, spreads, s_arr
