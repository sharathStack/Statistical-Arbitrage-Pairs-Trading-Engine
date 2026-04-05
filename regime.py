"""
regime.py  –  Regime classifier: mean-reverting vs trending

Features (computed over rolling window):
  - Hurst exponent  (R/S analysis;  H < 0.5 → mean-reverting)
  - AR(1) autocorrelation at lag 1 and lag 5
  - Volatility ratio (short / long realized vol)
  - |Z-score| (how stretched the spread is)
  - Skewness, excess kurtosis

Labels:
  1 = mean-reverting  (stat-arb active)
  0 = trending        (stat-arb suspended)

Model: GradientBoostingClassifier (scikit-learn)
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

import config


class RegimeClassifier:

    def __init__(self):
        self.clf    = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, spread: np.ndarray) -> None:
        W     = config.REGIME_WINDOW
        feats = self._features(spread, W)
        if len(feats) < 20:
            return
        hursts = feats[:, 0]
        labels = (hursts < config.REGIME_HURST_THR).astype(int)

        X_sc = self.scaler.fit_transform(feats)
        tscv = TimeSeriesSplit(n_splits=min(3, len(X_sc) // 10))
        for tr_idx, _ in tscv.split(X_sc):
            if len(tr_idx) > 5:
                self.clf.fit(X_sc[tr_idx], labels[tr_idx])
        self.fitted = True

    def predict(self, spread: np.ndarray) -> np.ndarray:
        W     = config.REGIME_WINDOW
        n     = len(spread)
        if not self.fitted:
            return np.ones(n, dtype=int)
        feats     = self._features(spread, W)
        regime_oos = self.clf.predict(self.scaler.transform(feats))
        # Pad front with 1s (mean-reverting assumed during warm-up)
        full      = np.ones(n, dtype=int)
        full[W:]  = regime_oos[:n - W]
        return full

    # ── Feature builder ───────────────────────────────────────────────────────
    def _features(self, spread: np.ndarray, W: int) -> np.ndarray:
        rows = []
        s    = pd.Series(spread)
        for i in range(W, len(spread)):
            w = spread[i-W:i]
            rows.append([
                self._hurst(w),
                float(s.iloc[i-W:i].autocorr(1)),
                float(s.iloc[i-W:i].autocorr(5)),
                w[-10:].std() / (w.std() + 1e-8),
                abs((w[-1] - w.mean()) / (w.std() + 1e-8)),
                float(stats.skew(w)),
                float(stats.kurtosis(w)),
            ])
        return np.array(rows, dtype=np.float32)

    @staticmethod
    def _hurst(ts: np.ndarray) -> float:
        """Hurst exponent via R/S analysis.  H < 0.5 → mean-revert."""
        lags = range(2, min(len(ts) // 2, 20))
        tau  = [np.std(np.subtract(ts[lag:], ts[:-lag])) + 1e-12 for lag in lags]
        try:
            return float(np.polyfit(np.log(list(lags)), np.log(tau), 1)[0])
        except Exception:
            return 0.5
