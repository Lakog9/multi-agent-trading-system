"""
harness.py
----------
The backtesting engine. Two rules enforced structurally:
1. POINT-IN-TIME: decision on day T sees only data up to T.
2. NEXT-BAR EXECUTION: weights decided on T execute on T+1.
"""

from __future__ import annotations
import pandas as pd


class BacktestResult:
    def __init__(self, equity: pd.Series, weights_log: pd.DataFrame, trades: int):
        self.equity = equity
        self.weights_log = weights_log
        self.trades = trades


def run_backtest(
    prices: pd.DataFrame,
    strategy,
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
    warmup: int = 0,
) -> BacktestResult:
    dates = prices.index
    tickers = list(prices.columns)

    cash = float(initial_capital)
    shares = {t: 0.0 for t in tickers}

    equity = pd.Series(index=dates, dtype=float)
    weights_records = []
    pending = None
    trade_count = 0
    cost_rate = cost_bps / 10_000.0

    for i, date in enumerate(dates):
        px = prices.loc[date]

        if pending is not None:
            pv = cash + sum(shares[t] * px[t] for t in tickers)
            for t in tickers:
                target_w = pending.get(t, 0.0)
                target_dollars = target_w * pv
                target_shares = target_dollars / px[t]
                delta_shares = target_shares - shares[t]
                notional = abs(delta_shares * px[t])
                cash -= delta_shares * px[t]
                cash -= notional * cost_rate
                shares[t] = target_shares
            trade_count += 1
            pending = None

        pv = cash + sum(shares[t] * px[t] for t in tickers)
        equity.loc[date] = pv

        if i >= warmup:
            history = prices.loc[:date]
            signal = strategy.target_weights(date, history)
            if signal is not None:
                rec = {"date": date}
                rec.update(signal)
                weights_records.append(rec)
                pending = signal

    weights_log = (
        pd.DataFrame(weights_records).set_index("date")
        if weights_records else pd.DataFrame(index=pd.Index([], name="date"))
    )
    return BacktestResult(equity=equity, weights_log=weights_log, trades=trade_count)


def walk_forward(
    prices: pd.DataFrame,
    strategy_factory,
    train_days: int = 252 * 2,
    test_days: int = 63,
    initial_capital: float = 100_000.0,
    cost_bps: float = 5.0,
):
    dates = prices.index
    n = len(dates)
    segments = []
    capital = initial_capital
    pos = train_days

    while pos < n:
        train_slice = prices.iloc[pos - train_days:pos]
        test_slice = prices.iloc[pos:pos + test_days]
        if len(test_slice) == 0:
            break
        strat = strategy_factory(train_slice)
        res = run_backtest(test_slice, strat, initial_capital=capital, cost_bps=cost_bps)
        segments.append(res.equity)
        capital = float(res.equity.iloc[-1])
        pos += test_days

    if not segments:
        raise ValueError("Not enough data for even one walk-forward window.")
    stitched = pd.concat(segments)
    stitched = stitched[~stitched.index.duplicated(keep="first")]
    return stitched
