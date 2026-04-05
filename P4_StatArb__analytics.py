"""
analytics.py  –  Trade and portfolio performance analytics

Computes:
  win_rate, total_pnl, sharpe, profit_factor,
  avg_holding, max_drawdown, best/worst trade
"""
from __future__ import annotations
import math

import numpy as np
import pandas as pd

from trader import Trade


def trade_stats(trades: list[Trade]) -> dict:
    if not trades:
        return {}
    pnls     = np.array([t.pnl for t in trades])
    holds    = np.array([t.holding_bars for t in trades])
    cum_pnl  = np.cumsum(pnls)
    peak     = np.maximum.accumulate(cum_pnl)
    dd       = cum_pnl - peak
    winners  = pnls[pnls > 0]
    losers   = pnls[pnls < 0]
    sharpe   = (pnls.mean() / (pnls.std() + 1e-12)) * math.sqrt(252)
    pf       = abs(winners.sum()) / max(abs(losers.sum()), 1e-8)

    return {
        "n_trades":      len(trades),
        "win_rate":      round(len(winners) / len(trades), 4),
        "total_pnl":     round(float(pnls.sum()), 5),
        "avg_pnl":       round(float(pnls.mean()), 6),
        "sharpe_ann":    round(sharpe, 3),
        "profit_factor": round(pf, 3),
        "avg_hold_bars": round(float(holds.mean()), 1),
        "max_drawdown":  round(float(dd.min()), 5),
        "best_trade":    round(float(pnls.max()), 5),
        "worst_trade":   round(float(pnls.min()), 5),
    }


def portfolio_summary(pair_stats: list[dict]) -> pd.DataFrame:
    if not pair_stats:
        return pd.DataFrame()
    df = pd.DataFrame(pair_stats)
    cols = ["pair", "n_trades", "win_rate", "sharpe_ann",
            "profit_factor", "total_pnl", "max_drawdown"]
    available = [c for c in cols if c in df.columns]
    return df[available].sort_values("sharpe_ann", ascending=False)
