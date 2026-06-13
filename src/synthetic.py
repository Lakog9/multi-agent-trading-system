"""
synthetic.py
------------
Generates fake-but-realistic daily price data for testing the pipeline
without any market data connection. Three regimes: bull, bear, sideways.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _regime_drifts(n_days: int) -> np.ndarray:
    third = n_days // 3
    bull = np.full(third, 0.0011)
    bear = np.full(third, -0.0009)
    side = np.full(n_days - 2 * third, 0.0001)
    return np.concatenate([bull, bear, side])


def generate(
    tickers: list[str],
    benchmark: str = "SPY",
    start: str = "2018-01-01",
    n_days: int = 252 * 6,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    market_drift = _regime_drifts(n_days)
    market_vol = 0.008
    market_factor = market_drift + rng.normal(0, market_vol, n_days)

    betas = rng.uniform(0.8, 1.2, len(tickers))
    extra_drift = rng.normal(0.0001, 0.0001, len(tickers))
    idio_vol = rng.uniform(0.004, 0.008, len(tickers))

    cols = {}
    price_panel = np.zeros((n_days, len(tickers)))
    for j, t in enumerate(tickers):
        daily_ret = (
            betas[j] * market_factor
            + extra_drift[j]
            + rng.normal(0, idio_vol[j], n_days)
        )
        prices = 100.0 * np.cumprod(1.0 + daily_ret)
        price_panel[:, j] = prices
        cols[t] = prices

    cols[benchmark] = price_panel.mean(axis=1)

    df = pd.DataFrame(cols, index=dates)
    df.index.name = "date"
    return df


def regime_slices(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(prices)
    third = n // 3
    return {
        "bull": prices.iloc[:third],
        "bear": prices.iloc[third:2 * third],
        "sideways": prices.iloc[2 * third:],
    }
