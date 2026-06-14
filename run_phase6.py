"""
run_phase6.py
-------------
Phase 6: Full system with Memory + Reflection.

Runs the agent TWICE over the same period:
  Run 1: empty memory (no prior experience)
  Run 2: with the memory accumulated from Run 1

If the learning loop works, Run 2 should show the agent referencing
past lessons. This is the core test of whether reflection helps.

TEST_MODE = True: 6 months (~$4-5 API cost across two runs).
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
from orchestrator_v4 import OrchestratorV4
import metrics

COST_BPS  = 5.0
INITIAL   = 100_000.0
TEST_MODE = True
MEMORY_PATH = "memory_store.json"


def main():
    # Fresh start: clear any existing memory
    if os.path.exists(MEMORY_PATH):
        os.remove(MEMORY_PATH)
        print("Cleared existing memory for a clean test.\n")

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

    bh = run_backtest(prices[universe], BuyAndHold(universe),
                      initial_capital=INITIAL, cost_bps=COST_BPS)
    spy = run_backtest(prices[[BENCHMARK]], SingleAssetBuyAndHold(BENCHMARK),
                       initial_capital=INITIAL, cost_bps=COST_BPS)

    # === RUN 1: empty memory ===
    print("\n" + "="*60)
    print("RUN 1: Agent with EMPTY memory")
    print("="*60)
    orch1 = OrchestratorV4(universe, rebalance_days=10,
                           memory_path=MEMORY_PATH)
    run1 = run_backtest(prices[universe], orch1,
                        initial_capital=INITIAL, cost_bps=COST_BPS,
                        warmup=60)
    orch1.save_memory()
    print(f"\nRun 1 accumulated {len(orch1.memory)} memories.")

    # === RUN 2: with memory from Run 1 ===
    print("\n" + "="*60)
    print("RUN 2: Agent WITH memory from Run 1")
    print("="*60)
    orch2 = OrchestratorV4(universe, rebalance_days=10,
                           memory_path=MEMORY_PATH)
    print(f"Run 2 starts with {len(orch2.memory)} memories from Run 1.\n")
    run2 = run_backtest(prices[universe], orch2,
                        initial_capital=INITIAL, cost_bps=COST_BPS,
                        warmup=60)
    orch2.save_memory()

    results = {
        "BuyAndHold":     bh,
        "Benchmark(SPY)": spy,
        "Run1(no memory)": run1,
        "Run2(with memory)": run2,
    }

    named = {n: metrics.compute_all(r.equity) for n, r in results.items()}
    print("\n=== RESULTS (net of costs, 5 bps/trade) ===")
    print(metrics.format_report(named))
    print()
    for n, r in results.items():
        print(f"  {n}: {r.trades} rebalance event(s)")

    # Show accumulated memories
    print("\n=== ACCUMULATED LESSONS (long-term memory) ===")
    for e in orch2.memory.entries:
        if e.layer == "long":
            print(f"  [imp={e.importance:.2f}] {e.content}")

    plt.figure(figsize=(12, 6))
    styles = {
        "BuyAndHold":        ("steelblue", 1.2, "--"),
        "Benchmark(SPY)":    ("gray",      1.2, ":"),
        "Run1(no memory)":   ("orange",    1.5, "-"),
        "Run2(with memory)": ("green",     2.0, "-"),
    }
    for name, res in results.items():
        color, lw, ls = styles[name]
        plt.plot(res.equity.index, res.equity.values,
                 label=name, color=color, linewidth=lw, linestyle=ls)
    plt.title("Phase 6: Memory + Reflection (net of costs)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "phase6_equity.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nEquity curve saved to: {out}")


if __name__ == "__main__":
    main()
