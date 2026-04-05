"""
main.py  –  Statistical Arbitrage & Pairs Trading entry point

Run order:
  1. Generate synthetic cointegrated universe
  2. Screen all pairs (EG + Johansen + OU)
  3. Print screener results table
  4. Portfolio selection (cross-correlation constraint)
  5. Full backtest on each selected pair
  6. Portfolio summary
  7. Deep-dive: best pair dashboard
"""

import config
from data_gen  import generate
from screener  import Screener
from trader    import PairTrader
from portfolio import Portfolio
from analytics import trade_stats, portfolio_summary
from ou_model  import estimate
import dashboard
from kalman_filter import KalmanFilter


def main():
    print("═" * 62)
    print("  STATISTICAL ARBITRAGE & PAIRS TRADING ENGINE")
    print("═" * 62)

    # ── 1. Universe ───────────────────────────────────────────────────────────
    print("\n[1] Generating synthetic universe…")
    prices = generate()

    # ── 2. Cointegration screening ────────────────────────────────────────────
    print("\n[2] Cointegration screening…")
    screener = Screener()
    pairs    = screener.run(prices)

    if not pairs:
        print("  No cointegrated pairs found. Adjust thresholds in config.py.")
        return

    # ── 3. Print screener table ───────────────────────────────────────────────
    print(f"\n  {'Pair':<22} {'Corr':>6}  {'EG-p':>7}  "
          f"{'β':>8}  {'HL(d)':>7}  {'κ':>6}  {'entry-z':>8}  {'J-rank':>7}")
    print("  " + "─" * 75)
    for p in pairs:
        print(f"  {p.label():<22} {p.correlation:>6.3f}  {p.eg_pvalue:>7.4f}  "
              f"{p.hedge_ratio:>8.4f}  {p.half_life:>7.1f}  "
              f"{p.ou_kappa:>6.3f}  {p.entry_z:>8.3f}  {p.johansen_rank:>7}")

    # ── 4. Portfolio selection ────────────────────────────────────────────────
    print("\n[3] Portfolio selection (cross-correlation constraint)…")
    port     = Portfolio()
    selected = port.select(pairs, prices)

    corr_mat = port.correlation_matrix()
    if corr_mat is not None:
        print(f"\n  Spread cross-correlation matrix:")
        labels = [p.label() for p in selected]
        col_w  = max(len(l) for l in labels) + 2
        header = " " * col_w + "".join(f"{l:>{col_w}}" for l in labels)
        print("  " + header)
        for i, row_label in enumerate(labels):
            row_str = "".join(f"{corr_mat[i,j]:>{col_w}.3f}" for j in range(len(labels)))
            print(f"  {row_label:<{col_w}}{row_str}")

    # ── 5. Backtest all selected pairs ────────────────────────────────────────
    print("\n[4] Backtesting selected pairs…")
    all_stats = []
    best_pair_data = None

    for pair in selected:
        pa = prices[pair.sym_a].values
        pb = prices[pair.sym_b].values

        trader = PairTrader(pair)
        trades, equity, z_arr, spreads, regimes = trader.backtest(pa, pb)

        stats = trade_stats(trades)
        if stats:
            stats["pair"] = pair.label()
            all_stats.append(stats)

        print(f"\n  {pair.label()}")
        print(f"  {'─'*40}")
        for k, v in stats.items():
            if k != "pair":
                print(f"    {k:<22} {v}")

        # OU parameter check (in-sample spread)
        ou = estimate(spreads[config.KF_WARMUP:])
        print(f"    {'ou_half_life(d)':<22} {ou['half_life']}")
        print(f"    {'ou_kappa(ann)':<22} {ou['kappa']}")

        # Keep best (highest Sharpe) for dashboard deep-dive
        if best_pair_data is None or (stats.get("sharpe_ann", -99) >
                                       best_pair_data["stats"].get("sharpe_ann", -99)):
            kf_best = KalmanFilter()
            betas_b, _, _ = kf_best.run_series(pa, pb)
            best_pair_data = dict(pa=pa, pb=pb, spreads=spreads, z_arr=z_arr,
                                  regimes=regimes, trades=trades, equity=equity,
                                  betas=betas_b, label=pair.label(), stats=stats)

    # ── 6. Portfolio summary ──────────────────────────────────────────────────
    print("\n[5] Portfolio Summary:")
    port_df = portfolio_summary(all_stats)
    if not port_df.empty:
        print(port_df.to_string(index=False))

        agg = port_df.select_dtypes(include="number")
        print(f"\n  Avg Sharpe      : {port_df['sharpe_ann'].mean():.3f}")
        print(f"  Avg Win Rate    : {port_df['win_rate'].mean():.3f}")
        print(f"  Total P&L       : {port_df['total_pnl'].sum():.5f}")

    # ── 7. Dashboard ──────────────────────────────────────────────────────────
    if best_pair_data:
        print(f"\n[6] Generating dashboard for best pair "
              f"({best_pair_data['label']})…")
        dashboard.plot(
            pa          = best_pair_data["pa"],
            pb          = best_pair_data["pb"],
            spreads     = best_pair_data["spreads"],
            z_arr       = best_pair_data["z_arr"],
            regimes     = best_pair_data["regimes"],
            trades      = best_pair_data["trades"],
            equity      = best_pair_data["equity"],
            betas       = best_pair_data["betas"],
            portfolio_df= port_df,
            pair_label  = best_pair_data["label"],
        )

    print("\n  Done ✓")


if __name__ == "__main__":
    main()
