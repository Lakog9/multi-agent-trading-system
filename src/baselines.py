"""
baselines.py
------------
The strategies every future agent must beat to justify its existence.
"""

from __future__ import annotations
import pandas as pd


class BuyAndHold:
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self._done = False

    def target_weights(self, date, price_history: pd.DataFrame):
        if self._done:
            return None
        self._done = True
        w = 1.0 / len(self.tickers)
        return {t: w for t in self.tickers}


class EqualWeightRebalanced:
    def __init__(self, tickers: list[str], rebalance_days: int = 21):
        self.tickers = tickers
        self.rebalance_days = rebalance_days
        self._counter = 0
        self._started = False

    def target_weights(self, date, price_history: pd.DataFrame):
        if not self._started:
            self._started = True
            self._counter = 0
            w = 1.0 / len(self.tickers)
            return {t: w for t in self.tickers}
        self._counter += 1
        if self._counter >= self.rebalance_days:
            self._counter = 0
            w = 1.0 / len(self.tickers)
            return {t: w for t in self.tickers}
        return None


class SingleAssetBuyAndHold:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self._done = False

    def target_weights(self, date, price_history: pd.DataFrame):
        if self._done:
            return None
        self._done = True
        return {self.ticker: 1.0}
