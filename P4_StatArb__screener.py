"""
screener.py  –  Cointegration screener

Pipeline per pair:
  1. Pearson correlation filter
  2. Engle-Granger cointegration test (ADF on OLS residuals)
  3. Johansen rank test
  4. OU parameter estimation  (κ, σ, half-life)
  5. Optimal entry threshold  (Elliott 1994 approximation)
"""
from __future__ import annotations
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import config


@dataclass
class PairResult:
    sym_a:          str
    sym_b:          str
    correlation:    float
    eg_pvalue:      float
    johansen_rank:  int
    hedge_ratio:    float
    spread_mean:    float
    spread_std:     float
    half_life:      float      # days
    ou_kappa:       float      # annualised
    ou_sigma:       float
    entry_z:        float      # optimal entry z-score

    def label(self) -> str:
        return f"{self.sym_a}/{self.sym_b}"


class Screener:

    def run(self, prices: pd.DataFrame) -> list[PairResult]:
        syms    = prices.columns.tolist()
        combos  = list(itertools.combinations(syms, 2))
        results = []
        print(f"\n  Screening {len(combos)} pairs…")

        for sym_a, sym_b in combos:
            pa, pb = prices[sym_a].values, prices[sym_b].values

            # 1. Correlation
            corr = float(np.corrcoef(pa, pb)[0, 1])
            if abs(corr) < config.CORR_THRESHOLD:
                continue

            # 2. Engle-Granger
            _, eg_pval, _ = coint(pa, pb)
            if eg_pval > config.EG_PVALUE:
                continue

            # 3. OLS hedge ratio + ADF on spread
            beta   = float(np.polyfit(pb, pa, 1)[0])
            spread = pa - beta * pb
            adf_p  = adfuller(spread, maxlags=5, autolag="AIC")[1]
            if adf_p > config.EG_PVALUE:
                continue

            # 4. OU parameters
            hl, kappa, sigma_ou = self._ou_params(spread)
            if not (config.MIN_HALF_LIFE <= hl <= config.MAX_HALF_LIFE):
                continue

            # 5. Johansen rank
            jrank = self._johansen_rank(np.column_stack([pa, pb]))

            # 6. Optimal entry (Elliott 1994)
            entry_z = self._optimal_entry(kappa)

            results.append(PairResult(
                sym_a=sym_a, sym_b=sym_b,
                correlation=round(corr, 4),
                eg_pvalue=round(eg_pval, 5),
                johansen_rank=jrank,
                hedge_ratio=round(beta, 5),
                spread_mean=round(float(spread.mean()), 6),
                spread_std=round(float(spread.std()), 6),
                half_life=round(hl, 2),
                ou_kappa=round(kappa, 4),
                ou_sigma=round(sigma_ou, 6),
                entry_z=round(entry_z, 3),
            ))

        results.sort(key=lambda r: r.half_life)
        print(f"  {len(results)} cointegrated pairs found.\n")
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _ou_params(spread: np.ndarray) -> tuple[float, float, float]:
        y, x    = spread[1:], spread[:-1]
        b       = np.polyfit(x, y, 1)
        kappa   = max(-math.log(b[0]) * 252, 1e-4)
        hl      = math.log(2) / kappa * 252
        sigma   = np.std(y - (b[0] * x + b[1])) * math.sqrt(252)
        return hl, kappa, sigma

    @staticmethod
    def _johansen_rank(data: np.ndarray) -> int:
        try:
            res    = coint_johansen(data, det_order=0, k_ar_diff=1)
            return int(np.sum(res.lr1 > res.cvt[:, 1]))
        except Exception:
            return 0

    @staticmethod
    def _optimal_entry(kappa: float) -> float:
        # σ_∞ = σ_ou / sqrt(2κ) is the stationary std
        # Entry ≈ 1.5 × σ_∞ clipped to [1.0, 3.0]
        entry = max(1.0, min(3.0, 1.5 * math.sqrt(1 / (2 * kappa / 252 + 1e-6))))
        return entry
