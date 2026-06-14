"""
run_phase2_llm.py
-----------------
Phase 2.5: LLM Technical Agent vs rule-based vs baselines.

WARNING: This makes real API calls. Cost estimate: ~$0.50-1.00
for the full 2018-2026 period (424 rebalance events).

To test cheaply first, set TEST_MODE = True (runs only 6 months).
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
from baselines import BuyAndHold, SingleAssetBuyAndHold
from technical_agent import TechnicalAgent
from llm_technical_agent import LLMTechnicalAgent
import metrics

COST_BPS = 5.0
INITIAL = 100_000.0
TEST_MODE = True   # set False for full backtest (costs ~$1)


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

    print("Running BuyAndHold...")
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

    print("Running LLM TechnicalAgent (API calls)...")
    llm_agent = LLMTechnicalAgent(universe, rebalance_days=5)
    llm = run_backtest(prices[universe], llm_agent,
                       initial_capital=INITIAL, cost_bps=COST_BPS,
                       warmup=60)
    print(f"Last LLM reasoning: '{llm_agent.last_reasoning}'")
    print(f"Last LLM weights:    {llm_agent.last_weights}\n")

    results = {
        "BuyAndHold":      bh,
        "Benchmark(SPY)":  spy,
        "TechAgent(rules)": tech,
        "TechAgent(LLM)":  llm,
    }

    named = {n: metrics.compute_all(r.equity) for n, r in results.items()}
    print("=== RESULTS (net of costs, 5 bps/trade) ===")
    print(metrics.format_report(named))
    print()
    for n, r in results.items():
        print(f"  {n}: {r.trades} rebalance event(s)")

    plt.figure(figsize=(12, 6))
    styles = {
        "BuyAndHold":       ("steelblue", 1.2, "--"),
        "Benchmark(SPY)":   ("gray",      1.2, ":"),
        "TechAgent(rules)": ("orange",    1.5, "--"),
        "TechAgent(LLM)":   ("green",     2.0, "-"),
    }
    for name, res in results.items():
        color, lw, ls = styles[name]
        plt.plot(res.equity.index, res.equity.values,
                 label=name, color=color, linewidth=lw, linestyle=ls)
    plt.title("Phase 2.5: LLM Technical Agent vs rule-based vs baselines")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "phase2_llm_equity.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nEquity curve saved to: {out}")


if __name__ == "__main__":
    main()
