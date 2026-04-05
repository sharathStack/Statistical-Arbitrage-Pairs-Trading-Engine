"""
portfolio.py  –  Multi-pair stat-arb portfolio management

Selection rules:
  1. Take pairs in order of ascending half-life (fastest mean-reversion first)
  2. Reject pairs whose spread is too correlated with any already-selected pair
  3. Cap at MAX_PAIRS

This limits portfolio-level correlation risk from highly co-moving spreads.
"""
from __future__ import annotations

import numpy as np

import config
from screener import PairResult


class Portfolio:

    def __init__(self):
        self.selected:  list[PairResult]    = []
        self._spreads:  list[np.ndarray]    = []

    def select(self, candidates: list[PairResult],
               prices) -> list[PairResult]:
        """
        Greedy selection: add a pair only if its spread is not too
        correlated with spreads already in the book.
        prices: pd.DataFrame of raw price series
        """
        self.selected = []
        self._spreads = []

        for cand in candidates:
            if len(self.selected) >= config.MAX_PAIRS:
                break

            pa = prices[cand.sym_a].values
            pb = prices[cand.sym_b].values
            sp_new = pa - cand.hedge_ratio * pb

            too_corr = False
            for sp_exist in self._spreads:
                n = min(len(sp_exist), len(sp_new))
                ρ = float(np.corrcoef(sp_exist[-n:], sp_new[-n:])[0, 1])
                if abs(ρ) > config.MAX_CROSS_CORR:
                    too_corr = True
                    break

            if not too_corr:
                self.selected.append(cand)
                self._spreads.append(sp_new)

        print(f"  Portfolio: {len(self.selected)} pairs selected "
              f"(from {len(candidates)} candidates)")
        return self.selected

    def correlation_matrix(self) -> np.ndarray | None:
        if len(self._spreads) < 2:
            return None
        n   = min(len(s) for s in self._spreads)
        mat = np.column_stack([s[-n:] for s in self._spreads])
        return np.corrcoef(mat.T)
