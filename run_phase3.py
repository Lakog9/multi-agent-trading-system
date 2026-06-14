"""
run_phase3.py
-------------
Phase 3: Full council of agents (Technical + Fundamental + Sentiment + Macro)
coordinated by the Orchestrator.

TEST_MODE = True runs only 3 months to keep API costs low (~$1-2).
Set to False for full backtest (~$5-8).
"""

from __future__ import annotations
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import data_layer
from data_layer import DEFAULT_UNIVERSE, BENCHMARK
from harness import run_backtest
from baselines import BuyAndHold, SingleAssetBuyAndHold
from technical_agent import TechnicalAgent
from orchestrator import Orchestrator
import metrics

COST_BPS = 5.0
INITIAL = 100_000.0
TEST_MODE = True   # set False for full backtest


def main():
    print("Fetching prices...")
    prices = data_layer.fetch_prices(
        DEFAULT_UNIVERSE + [BENCHMARK], start="2018-01-01"
    )

    if TEST_MODE:
        prices = prices.loc["2024-01-01":"2024-06-30"]
        print("TEST MODE: running 6 months only (2024-H1)")

    print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()} "
          f"({len(prices)} trading days)\n")

    universe = DEFAULT_UNIVERSE

    print("Running BuyAndHold baseline...")
    bh = run_backtest(prices[universe], BuyAndHold(universe),
                      initial_capital=INITIAL, cost_bps=COST_BPS)

    print("Running SPY benchmark...")
    spy = run_backtest(prices[[BENCHMARK]], SingleAssetBuyAndHold(BENCHMARK),
                       initial_capital=INITIAL, cost_bps=COST_BPS)

    print("Running rule-based TechnicalAgent...")
    tech = run_backtest(prices[universe],
                        TechnicalAgent(universe, rebalance_days=5),
                        initial_capital=INITIAL, cost_bps=COST_BPS,
                        warmup=60)

    print("\nRunning full Orchestrator (council of agents)...")
    orch = Orchestrator(universe, rebalance_days=10)
    council = run_backtest(prices[universe], orch,
                           initial_capital=INITIAL, cost_bps=COST_BPS,
                           warmup=60)

    results = {
        "BuyAndHold":      bh,
        "Benchmark(SPY)":  spy,
        "TechAgent(rules)": tech,
        "Council(LLM)":    council,
    }

    named = {n: metrics.compute_all(r.equity) for n, r in results.items()}
    print("\n=== RESULTS (net of costs, 5 bps/trade) ===")
    print(metrics.format_report(named))
    print()
    for n, r in results.items():
        print(f"  {n}: {r.trades} rebalance event(s)")

    # Print decision log
    if orch.decision_log:
        print("\n=== COUNCIL DECISION LOG ===")
        for d in orch.decision_log:
            print(f"  {d['date']} | sentiment={d['sentiment']} | "
                  f"macro={d['macro']}")
            print(f"    weights={d['weights']}")
            print(f"    reasoning: {d['reasoning']}")

        log_path = os.path.join(os.path.dirname(__file__), "decision_log.json")
        with open(log_path, "w") as f:
            json.dump(orch.decision_log, f, indent=2)
        print(f"\nDecision log saved to: {log_path}")

    # Equity curve
    plt.figure(figsize=(12, 6))
    styles = {
        "BuyAndHold":       ("steelblue", 1.2, "--"),
        "Benchmark(SPY)":   ("gray",      1.2, ":"),
        "TechAgent(rules)": ("orange",    1.5, "--"),
        "Council(LLM)":     ("green",     2.0, "-"),
    }
    for name, res in results.items():
        color, lw, ls = styles[name]
        plt.plot(res.equity.index, res.equity.values,
                 label=name, color=color, linewidth=lw, linestyle=ls)
    plt.title("Phase 3: Council of Agents vs baselines (net of costs)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "phase3_equity.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nEquity curve saved to: {out}")


if __name__ == "__main__":
    main()
