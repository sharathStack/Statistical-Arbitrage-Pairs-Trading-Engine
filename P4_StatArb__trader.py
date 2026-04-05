"""
trader.py  –  Event-driven backtest for a single cointegrated pair

Signal logic (z-score of Kalman-filtered spread):
  z > +entry_z  →  SHORT spread  (sell A, buy B)
  z < -entry_z  →  LONG  spread  (buy A, sell B)
  |z| < exit_z  →  close position
  |z| > stop_z  →  stop loss

Position sizing: fixed notional (1 unit of spread).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config
from kalman_filter import KalmanFilter
from ou_model      import z_score, estimate
from regime        import RegimeClassifier
from screener      import PairResult


@dataclass
class Trade:
    direction:      int          # +1 long spread, -1 short spread
    entry_idx:      int
    entry_z:        float
    entry_pa:       float
    entry_pb:       float
    hedge_ratio:    float
    exit_idx:       int   = -1
    exit_z:         float = 0.0
    pnl:            float = 0.0
    holding_bars:   int   = 0


class PairTrader:

    def __init__(self, pair: PairResult):
        self.pair      = pair
        self.entry_z   = pair.entry_z
        self.exit_z    = config.Z_EXIT
        self.stop_z    = config.Z_STOP
        self.max_hold  = config.MAX_HOLDING
        self.tc        = config.TRANSACTION_COST

    def backtest(self, pa: np.ndarray, pb: np.ndarray
                 ) -> tuple[list[Trade], np.ndarray, np.ndarray,
                             np.ndarray, np.ndarray]:

        n = len(pa)
        kf = KalmanFilter()
        betas, spreads, _ = kf.run_series(pa, pb)

        lb     = 60
        z_arr  = z_score(spreads, window=lb)

        # Regime filter
        clf = RegimeClassifier()
        try:
            clf.fit(spreads)
            regimes = clf.predict(spreads)
        except Exception:
            regimes = np.ones(n, dtype=int)

        trades:  list[Trade]    = []
        active:  Optional[Trade] = None
        equity   = np.ones(n)
        warmup   = max(config.KF_WARMUP, lb)

        for i in range(warmup, n):
            z = z_arr[i]
            β = betas[i]

            # ── Close check ──────────────────────────────────────────────────
            if active is not None:
                hold   = i - active.entry_idx
                close  = (
                    (active.direction ==  1 and z < self.exit_z)
                    or (active.direction == -1 and z > -self.exit_z)
                    or abs(z) > self.stop_z
                    or hold >= self.max_hold
                )
                if close:
                    pnl_raw = active.direction * (
                        (pa[i] - active.entry_pa)
                        - β * (pb[i] - active.entry_pb)
                    )
                    cost       = self.tc * (abs(pa[i]) + abs(β * pb[i]))
                    active.exit_idx    = i
                    active.exit_z      = z
                    active.pnl         = pnl_raw - cost
                    active.holding_bars = hold
                    trades.append(active)
                    active = None

            # ── Open check ────────────────────────────────────────────────────
            if active is None and regimes[i] == 1:
                if z > self.entry_z:
                    active = Trade(-1, i, z, pa[i], pb[i], β)
                elif z < -self.entry_z:
                    active = Trade(+1, i, z, pa[i], pb[i], β)

            # ── Equity curve ──────────────────────────────────────────────────
            cum = sum(t.pnl for t in trades)
            equity[i] = 1.0 + cum / max(abs(pa[0]), 1e-6)

        return trades, equity, z_arr, spreads, regimes
