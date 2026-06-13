"""
run_phase2.py
-------------
Phase 2: Technical Agent vs baselines.
Same harness, same metrics, same costs. Honest comparison.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_layer
from data_layer import DEFAULT_UNIVERSE, BENCHMARK
from harness import run_backtest
from baselines import BuyAndHold, EqualWeightRebalanced, SingleAssetBuyAndHold
from technical_agent import TechnicalAgent
import metrics

COST_BPS = 5.0
INITIAL = 100_000.0


def main():
    print("Fetching prices...")
    prices = data_layer.fetch_prices(
        DEFAULT_UNIVERSE + [BENCHMARK], start="2018-01-01"
    )

    print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()} "
          f"({len(prices)} trading days)\n")

    universe = DEFAULT_UNIVERSE

    # Baselines
    bh  = run_backtest(prices[universe], BuyAndHold(universe),
                       initial_capital=INITIAL, cost_bps=COST_BPS)
    ew  = run_backtest(prices[universe], EqualWeightRebalanced(universe, 21),
                       initial_capital=INITIAL, cost_bps=COST_BPS)
    spy = run_backtest(prices[[BENCHMARK]], SingleAssetBuyAndHold(BENCHMARK),
                       initial_capital=INITIAL, cost_bps=COST_BPS)

    # Technical Agent (rebalances every 5 trading days)
    tech = run_backtest(prices[universe],
                        TechnicalAgent(universe, rebalance_days=5),
                        initial_capital=INITIAL, cost_bps=COST_BPS,
                        warmup=60)

    results = {
        "BuyAndHold":       bh,
        "EqualWeight(21d)": ew,
        "Benchmark(SPY)":   spy,
        "TechnicalAgent":   tech,
    }

    # Full period report
    named = {n: metrics.compute_all(r.equity) for n, r in results.items()}
    print("=== FULL PERIOD (net of costs, 5 bps/trade) ===")
    print(metrics.format_report(named))
    print()
    for n, r in results.items():
        print(f"  {n}: {r.trades} rebalance event(s)")
    print()

    # Show last scores so we can see what the agent "thinks" today
    last_history = prices[universe].dropna()
    agent = TechnicalAgent(universe)
    scores = agent.last_scores(last_history)
    print("=== CURRENT TECHNICAL SCORES (latest available date) ===")
    print(f"  {'Ticker':<8} {'Score':>6}  {'Signal'}")
    print(f"  {'-'*35}")
    for t, s in sorted(scores.items(), key=lambda x: -x[1]):
        if s >= 2:
            label = "BULLISH"
        elif s <= -2:
            label = "BEARISH"
        elif s > 0:
            label = "mild bullish"
        elif s < 0:
            label = "mild bearish"
        else:
            label = "neutral"
        print(f"  {t:<8} {s:>6.1f}  {label}")
    print()

    # Equity curve
    plt.figure(figsize=(12, 6))
    styles = {
        "BuyAndHold":       ("steelblue",  1.2, "--"),
        "EqualWeight(21d)": ("orange",     1.2, "--"),
        "Benchmark(SPY)":   ("gray",       1.2, ":"),
        "TechnicalAgent":   ("green",      2.0, "-"),
    }
    for name, res in results.items():
        color, lw, ls = styles[name]
        plt.plot(res.equity.index, res.equity.values,
                 label=name, color=color, linewidth=lw, linestyle=ls)
    plt.title("Phase 2: Technical Agent vs baselines (net of costs)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "phase2_equity.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Equity curve saved to: {out}")


if __name__ == "__main__":
    main()
