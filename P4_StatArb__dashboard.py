"""
dashboard.py  –  Statistical Arbitrage visualisation dashboard
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

import config
from trader import Trade

DARK, GRID_C = "#0f1117", "#1a1d2e"
WHITE        = "#e8eaf6"
GREEN        = "#00d4aa"
RED          = "#ff4d6d"
BLUE         = "#4d9de0"
AMBER        = "#f7b731"
PURPLE       = "#a55eea"
COLORS       = [GREEN, BLUE, AMBER, RED, PURPLE]


def _style(ax):
    ax.set_facecolor(GRID_C)
    ax.tick_params(colors=WHITE, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2a2d3e")
    ax.title.set_color(WHITE)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.grid(alpha=0.18, color="#3a3d50")


def plot(pa: np.ndarray, pb: np.ndarray,
         spreads: np.ndarray, z_arr: np.ndarray,
         regimes: np.ndarray, trades: list[Trade],
         equity: np.ndarray, betas: np.ndarray,
         portfolio_df: pd.DataFrame,
         pair_label: str) -> None:

    fig = plt.figure(figsize=(20, 13), facecolor=DARK)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ── 1. Price series ───────────────────────────────────────────────────────
    ax1  = fig.add_subplot(gs[0, :2])
    _style(ax1)
    ax1r = ax1.twinx()
    ax1.plot(pa, color=BLUE,  linewidth=1.0, label=pair_label.split("/")[0])
    ax1r.plot(pb, color=AMBER, linewidth=1.0, alpha=0.75,
              label=pair_label.split("/")[1])
    ax1.set_title(f"Price Series  ─  {pair_label}", fontweight="bold")
    ax1.set_ylabel(pair_label.split("/")[0], color=BLUE)
    ax1r.set_ylabel(pair_label.split("/")[1], color=AMBER)
    ax1r.tick_params(colors=WHITE, labelsize=7)
    lines_a, la = ax1.get_legend_handles_labels()
    lines_b, lb = ax1r.get_legend_handles_labels()
    ax1.legend(lines_a + lines_b, la + lb, loc="upper left",
               fontsize=8, facecolor=GRID_C, labelcolor=WHITE)

    # ── 2. Dynamic hedge ratio (Kalman β) ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _style(ax2)
    ax2.plot(betas, color=PURPLE, linewidth=1.0)
    ax2.set_title("Kalman Dynamic Hedge Ratio β(t)", fontweight="bold")
    ax2.set_ylabel("β")

    # ── 3. Spread ─────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    _style(ax3)
    ax3.plot(spreads, color=GREEN, linewidth=0.9, alpha=0.85, label="KF Spread")
    ax3.axhline(float(np.nanmean(spreads)), color=WHITE,
                linestyle="--", alpha=0.35, label="Mean")
    ax3.set_title("Kalman-Filtered Spread", fontweight="bold")
    ax3.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 4. Z-score with trade markers ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    _style(ax4)
    ax4.plot(z_arr, color=PURPLE, linewidth=0.9, alpha=0.85)
    entry_z = trades[0].hedge_ratio if trades else config.Z_EXIT  # proxy
    for thr, col, ls in [(2.0, RED, "--"), (-2.0, GREEN, "--"),
                          (config.Z_EXIT, WHITE, ":"), (-config.Z_EXIT, WHITE, ":")]:
        ax4.axhline(thr, color=col, linestyle=ls, alpha=0.55, linewidth=0.9)
    for t in trades[:60]:
        col = GREEN if t.direction == 1 else RED
        ax4.scatter(t.entry_idx, t.entry_z, color=col, s=18, zorder=5)
        if t.exit_idx > 0:
            ax4.scatter(t.exit_idx, t.exit_z, color=AMBER,
                        marker="x", s=20, zorder=5)
    ax4.set_title("Z-Score + Trade Entries / Exits", fontweight="bold")

    # ── 5. Regime filter ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    _style(ax5)
    ax5.fill_between(range(len(regimes)), regimes, alpha=0.55,
                     color=BLUE, label="Mean-Rev (1)")
    ax5.fill_between(range(len(regimes)), 1 - regimes, alpha=0.25,
                     color=RED, label="Trending (0)")
    ax5.set_title("Regime (GBM classifier)", fontweight="bold")
    ax5.set_ylim(-0.05, 1.2)
    ax5.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 6. Equity curve ───────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    _style(ax6)
    ax6.plot(equity, color=GREEN, linewidth=1.4)
    ax6.axhline(1.0, color=WHITE, linestyle="--", alpha=0.35)
    ax6.fill_between(range(len(equity)), equity, 1.0,
                     where=(equity >= 1.0), alpha=0.15, color=GREEN)
    ax6.fill_between(range(len(equity)), equity, 1.0,
                     where=(equity <  1.0), alpha=0.15, color=RED)
    ax6.set_title("Equity Curve", fontweight="bold")
    ax6.set_ylabel("Equity (start=1.0)")

    # ── 7. P&L distribution ───────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    _style(ax7)
    if trades:
        pnls = [t.pnl for t in trades]
        ax7.hist([p for p in pnls if p > 0], bins=20, color=GREEN,
                 alpha=0.75, label="Winners", edgecolor="none")
        ax7.hist([p for p in pnls if p <= 0], bins=20, color=RED,
                 alpha=0.75, label="Losers", edgecolor="none")
        ax7.axvline(0, color=WHITE, linestyle="--", alpha=0.4)
        ax7.set_title("Trade P&L Distribution", fontweight="bold")
        ax7.legend(fontsize=7, facecolor=GRID_C, labelcolor=WHITE)

    # ── 8. Portfolio summary table ────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, :])
    _style(ax8)
    ax8.axis("off")
    if not portfolio_df.empty:
        cols   = portfolio_df.columns.tolist()
        n_rows = len(portfolio_df)
        col_w  = 1.0 / (len(cols) + 0.5)

        # Header
        for j, col in enumerate(cols):
            ax8.text(j * col_w + col_w * 0.5, 0.90, col,
                     transform=ax8.transAxes, ha="center", va="top",
                     fontsize=8.5, color=AMBER, fontweight="bold",
                     family="monospace")
        # Rows
        for i, (_, row) in enumerate(portfolio_df.iterrows()):
            y = 0.90 - (i + 1) * 0.22
            for j, col in enumerate(cols):
                val = row[col]
                txt = f"{val:.4f}" if isinstance(val, float) else str(val)
                color = GREEN if col == "sharpe_ann" and isinstance(val, float) and val > 0 else WHITE
                ax8.text(j * col_w + col_w * 0.5, y, txt,
                         transform=ax8.transAxes, ha="center", va="top",
                         fontsize=8, color=color, family="monospace")

        ax8.set_title("Portfolio Summary", fontweight="bold", color=WHITE)
        ax8.title.set_color(WHITE)

    fig.suptitle(f"Statistical Arbitrage Dashboard  ─  {pair_label}",
                 fontsize=14, fontweight="bold", color=WHITE, y=1.01)
    plt.savefig(config.CHART_OUTPUT, dpi=config.CHART_DPI,
                bbox_inches="tight", facecolor=DARK)
    print(f"\nDashboard saved → {config.CHART_OUTPUT}")
