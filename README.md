# Multi-Agent LLM Trading System

A council of specialized LLM agents managing a portfolio with daily decisions,
built with one non-negotiable priority: an evaluation methodology that does not
let us fool ourselves.

## Status
**Phase 1 complete:** data layer, point-in-time backtest harness, baselines.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_phase1.py
```

For real data: set `DATA_SOURCE = "yfinance"` in `run_phase1.py`.

## Layout
src/

metrics.py      # CAGR, Sharpe, Sortino, Calmar, drawdown, win rate

harness.py      # point-in-time backtest engine, walk-forward wrapper

baselines.py    # buy-and-hold, equal-weight, single-asset benchmark

data_layer.py   # yfinance fetch + cache + point-in-time accessor

synthetic.py    # realistic fake data with bull/bear/sideways regimes

run_phase1.py     # ties everything together, prints report + equity curve

DESIGN.md         # full architecture and research basis

## Phases
1. Data layer, harness, baselines (done)
2. Technical Agent vs baselines
3. Full analyst team (fundamental, sentiment, macro)
4. Bull/Bear debate + trader allocation
5. Risk management team
6. Layered memory + reflection loop
7. Anonymization audit + paper trading
