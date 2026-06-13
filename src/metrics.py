"""
metrics.py
----------
Performance metrics computed from an equity curve (portfolio value over time).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()

def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)

def cagr(equity: pd.Series) -> float:
    n_days = len(equity)
    years = n_days / TRADING_DAYS
    if years <= 0:
        return 0.0
    growth = equity.iloc[-1] / equity.iloc[0]
    return float(growth ** (1.0 / years) - 1.0)

def annual_volatility(equity: pd.Series) -> float:
    r = daily_returns(equity)
    return float(r.std() * np.sqrt(TRADING_DAYS))

def sharpe_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    r = daily_returns(equity)
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = r - daily_rf
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))

def sortino_ratio(equity: pd.Series, risk_free_rate: float = 0.0) -> float:
    r = daily_returns(equity)
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = r - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(excess.mean() / downside.std() * np.sqrt(TRADING_DAYS))

def max_drawdown(equity: pd.Series) -> float:
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    return float(drawdown.min())

def calmar_ratio(equity: pd.Series) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity) / mdd)

def win_rate(equity: pd.Series) -> float:
    r = daily_returns(equity)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())

def compute_all(equity: pd.Series, risk_free_rate: float = 0.0) -> dict:
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "annual_volatility": annual_volatility(equity),
        "sharpe": sharpe_ratio(equity, risk_free_rate),
        "sortino": sortino_ratio(equity, risk_free_rate),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity),
        "win_rate": win_rate(equity),
    }

def format_report(named_metrics: dict[str, dict]) -> str:
    rows = [
        ("Total Return", "total_return", "pct"),
        ("CAGR", "cagr", "pct"),
        ("Annual Vol", "annual_volatility", "pct"),
        ("Sharpe", "sharpe", "num"),
        ("Sortino", "sortino", "num"),
        ("Max Drawdown", "max_drawdown", "pct"),
        ("Calmar", "calmar", "num"),
        ("Win Rate", "win_rate", "pct"),
    ]
    names = list(named_metrics.keys())
    col_w = max(14, max(len(n) for n in names) + 2)
    header = f"{'Metric':<16}" + "".join(f"{n:>{col_w}}" for n in names)
    lines = [header, "-" * len(header)]
    for label, key, kind in rows:
        line = f"{label:<16}"
        for n in names:
            v = named_metrics[n][key]
            s = f"{v*100:.2f}%" if kind == "pct" else f"{v:.2f}"
            line += f"{s:>{col_w}}"
        lines.append(line)
    return "\n".join(lines)
