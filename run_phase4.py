"""
run_phase4.py
-------------
Phase 4: Full council with Bull/Bear debate and Trader Agent.

TEST_MODE = True runs 3 months (~$2-3 in API costs).
Each rebalance now makes ~6 API calls (4 agents + 3 debate + 1 trader).
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
from orchestrator_v2 import OrchestratorV2
import metrics

COST_BPS = 5.0
INITIAL  = 100_000.0
TEST_MODE = True   # set False for full backtest


def main():
    print("Fetching prices...")
    prices = data_layer.fetch_prices(
        DEFAULT_UNIVERSE + [BENCHMARK], start="2018-01-01"
    )

    if TEST_MODE:
        prices = prices.loc["2024-01-01":"2024-06-30"]
        print("TEST MODE: 2024-H1 (6 months)")

    print(f"Period: {prices.index[0].date()} to "
          f"{prices.index[-1].date()} ({len(prices)} trading days)\n")

    universe = DEFAULT_UNIVERSE

    print("Running BuyAndHold...")
    bh = run_backtest(prices[universe], BuyAndHold(universe),
                      initial_capital=INITIAL, cost_bps=COST_BPS)

    print("Running SPY benchmark...")
    spy = run_backtest(prices[[BENCHMARK]], SingleAssetBuyAndHold(BENCHMARK),
                       initial_capital=INITIAL, cost_bps=COST_BPS)

    print("\nRunning OrchestratorV2 (with debate)...")
    orch = OrchestratorV2(universe, rebalance_days=10)
    council = run_backtest(prices[universe], orch,
                           initial_capital=INITIAL, cost_bps=COST_BPS,
                           warmup=60)

    results = {
        "BuyAndHold":    bh,
        "Benchmark(SPY)": spy,
        "Council+Debate": council,
    }

    named = {n: metrics.compute_all(r.equity) for n, r in results.items()}
    print("\n=== RESULTS (net of costs, 5 bps/trade) ===")
    print(metrics.format_report(named))
    print()
    for n, r in results.items():
        print(f"  {n}: {r.trades} rebalance event(s)")

    if orch.decision_log:
        print("\n=== DECISION LOG SUMMARY ===")
        for d in orch.decision_log:
            print(f"\n  {d['date']} | stance={d['stance']} | "
                  f"sentiment={d['sentiment']} | macro={d['macro']}")
            print(f"  weights={d['weights']}")
            print(f"  conviction={d['conviction']:.2f} | cash={d['cash']:.0%}")
            print(f"  risks: {d['key_risks']}")
            print(f"  opps:  {d['key_opps']}")
            print(f"  reasoning: {d['reasoning']}")

        log_path = os.path.join(os.path.dirname(__file__),
                                "decision_log_v2.json")
        with open(log_path, "w") as f:
            json.dump(orch.decision_log, f, indent=2)
        print(f"\nDecision log saved to: {log_path}")

    plt.figure(figsize=(12, 6))
    styles = {
        "BuyAndHold":     ("steelblue", 1.2, "--"),
        "Benchmark(SPY)": ("gray",      1.2, ":"),
        "Council+Debate": ("green",     2.0, "-"),
    }
    for name, res in results.items():
        color, lw, ls = styles[name]
        plt.plot(res.equity.index, res.equity.values,
                 label=name, color=color, linewidth=lw, linestyle=ls)
    plt.title("Phase 4: Council + Bull/Bear Debate (net of costs)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "phase4_equity.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nEquity curve saved to: {out}")


if __name__ == "__main__":
    main()
