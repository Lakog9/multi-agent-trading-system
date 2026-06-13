"""
run_phase1.py
-------------
Phase 1: proves the machinery works end to end.
No agents yet. Runs baselines and reports honest metrics per regime.

DATA_SOURCE = "yfinance"  -> works anywhere, no internet needed
DATA_SOURCE = "yfinance"   -> real data, run on your own machine
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synthetic
import data_layer
from data_layer import DEFAULT_UNIVERSE, BENCHMARK
from harness import run_backtest
from baselines import BuyAndHold, EqualWeightRebalanced, SingleAssetBuyAndHold
import metrics

DATA_SOURCE = "yfinance"   # change to "yfinance" on your machine
COST_BPS = 5.0
INITIAL = 100_000.0


def load_prices():
    if DATA_SOURCE == "yfinance":
        return data_layer.fetch_prices(
            DEFAULT_UNIVERSE + [BENCHMARK], start="2018-01-01"
        )
    return synthetic.generate(DEFAULT_UNIVERSE, benchmark=BENCHMARK)


def run_all(prices):
    universe = DEFAULT_UNIVERSE
    results = {}

    bh = run_backtest(prices[universe], BuyAndHold(universe),
                      initial_capital=INITIAL, cost_bps=COST_BPS)
    results["BuyAndHold"] = bh

    ew = run_backtest(prices[universe], EqualWeightRebalanced(universe, 21),
                      initial_capital=INITIAL, cost_bps=COST_BPS)
    results["EqualWeight(21d)"] = ew

    spy = run_backtest(prices[[BENCHMARK]], SingleAssetBuyAndHold(BENCHMARK),
                       initial_capital=INITIAL, cost_bps=COST_BPS)
    results["Benchmark(SPY)"] = spy

    return results


def main():
    prices = load_prices()
    print(f"Data source : {DATA_SOURCE}")
    print(f"Universe    : {DEFAULT_UNIVERSE}")
    print(f"Benchmark   : {BENCHMARK}")
    print(f"Period      : {prices.index[0].date()} to {prices.index[-1].date()} "
          f"({len(prices)} trading days)\n")

    results = run_all(prices)

    named = {name: metrics.compute_all(res.equity) for name, res in results.items()}
    print("=== FULL PERIOD (out-of-sample, net of costs) ===")
    print(metrics.format_report(named))
    print()
    for name, res in results.items():
        print(f"  {name}: {res.trades} rebalance event(s)")
    print()

    if DATA_SOURCE == "synthetic":
        regimes = synthetic.regime_slices(prices)
        for regime_name, sl in regimes.items():
            seg = {}
            for strat_name, res in results.items():
                eq = res.equity.loc[sl.index[0]:sl.index[-1]]
                eq = eq / eq.iloc[0] * INITIAL
                seg[strat_name] = metrics.compute_all(eq)
            print(f"=== REGIME: {regime_name.upper()} ===")
            print(metrics.format_report(seg))
            print()

    plt.figure(figsize=(11, 6))
    for name, res in results.items():
        plt.plot(res.equity.index, res.equity.values, label=name, linewidth=1.6)
    plt.title("Phase 1 baselines: equity curves (net of costs)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = os.path.join(os.path.dirname(__file__), "phase1_equity.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Equity curve saved to: {out_path}")


if __name__ == "__main__":
    main()
