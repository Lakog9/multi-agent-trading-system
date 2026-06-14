"""
run_phase5.py
-------------
Phase 5: Full system with Risk Management.
TEST_MODE = True: 6 months (~$3-4 API cost).
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
from orchestrator_v3 import OrchestratorV3
import metrics

COST_BPS  = 5.0
INITIAL   = 100_000.0
TEST_MODE = True


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

    print("\nRunning OrchestratorV2 (no risk manager)...")
    v2 = OrchestratorV2(universe, rebalance_days=10)
    council_v2 = run_backtest(prices[universe], v2,
                              initial_capital=INITIAL, cost_bps=COST_BPS,
                              warmup=60)

    print("\nRunning OrchestratorV3 (with risk manager)...")
    v3 = OrchestratorV3(universe, rebalance_days=10)

    # Feed equity updates to risk manager after each bar
    # We wrap run_backtest with a custom loop for this
    from harness import run_backtest as _rb
    council_v3 = _rb(prices[universe], v3,
                     initial_capital=INITIAL, cost_bps=COST_BPS,
                     warmup=60)

    # Feed equity curve into risk manager retrospectively
    # (in production this would be real-time)
    for val in council_v3.equity.values:
        v3.risk_mgr.update_equity(float(val))

    results = {
        "BuyAndHold":      bh,
        "Benchmark(SPY)":  spy,
        "Council(no risk)": council_v2,
        "Council+Risk":    council_v3,
    }

    named = {n: metrics.compute_all(r.equity) for n, r in results.items()}
    print("\n=== RESULTS (net of costs, 5 bps/trade) ===")
    print(metrics.format_report(named))
    print()
    for n, r in results.items():
        print(f"  {n}: {r.trades} rebalance event(s)")

    # Risk manager warnings summary
    if v3.risk_mgr.warnings_log:
        print("\n=== RISK MANAGER LOG ===")
        for entry in v3.risk_mgr.warnings_log:
            if entry.get("warnings"):
                print(f"  {entry['date']}: {entry['warnings']}")
                print(f"    vol={entry['port_vol']:.1%} | "
                      f"dd={entry['drawdown']:.1%}")

    # Decision log: proposed vs final (shows risk adjustments)
    if v3.decision_log:
        print("\n=== PROPOSED vs FINAL WEIGHTS ===")
        for d in v3.decision_log:
            prop_total = sum(d['proposed'].values())
            final_total = sum(d['final'].values())
            if abs(prop_total - final_total) > 0.01:
                print(f"  {d['date']}: proposed={prop_total:.0%} "
                      f"-> final={final_total:.0%} "
                      f"(risk adjustment: "
                      f"{final_total-prop_total:+.0%})")

        log_path = os.path.join(os.path.dirname(__file__),
                                "decision_log_v3.json")
        with open(log_path, "w") as f:
            json.dump(v3.decision_log, f, indent=2)
        print(f"\nDecision log saved to: {log_path}")

    plt.figure(figsize=(12, 6))
    styles = {
        "BuyAndHold":       ("steelblue", 1.2, "--"),
        "Benchmark(SPY)":   ("gray",      1.2, ":"),
        "Council(no risk)": ("orange",    1.5, "--"),
        "Council+Risk":     ("green",     2.0, "-"),
    }
    for name, res in results.items():
        color, lw, ls = styles[name]
        plt.plot(res.equity.index, res.equity.values,
                 label=name, color=color, linewidth=lw, linestyle=ls)
    plt.title("Phase 5: Full Council + Risk Management (net of costs)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "phase5_equity.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nEquity curve saved to: {out}")


if __name__ == "__main__":
    main()
