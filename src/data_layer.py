"""
data_layer.py
-------------
Daily price data fetcher with caching and point-in-time accessor.
On your machine: uses yfinance (pip install yfinance).
In restricted environments: use synthetic.py instead.
"""

from __future__ import annotations
import os
import pandas as pd

DEFAULT_UNIVERSE = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ"]
BENCHMARK = "SPY"

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_prices(
    tickers: list[str],
    start: str = "2016-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = "_".join(sorted(tickers)) + f"_{start}_{end or 'today'}"
    cache_path = os.path.join(CACHE_DIR, f"{key}.parquet")

    if use_cache and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is not installed. Run: pip install yfinance"
        ) from e

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    close = close.dropna(how="all").sort_index()
    close.index.name = "date"

    if use_cache:
        close.to_parquet(cache_path)
    return close


class PriceData:
    """Thin wrapper that makes point-in-time access explicit and hard to misuse."""
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices.sort_index()

    def as_of(self, date) -> pd.DataFrame:
        return self.prices.loc[:date]

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def dates(self):
        return self.prices.index
