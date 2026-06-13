"""
technical_agent.py
------------------
The Technical Agent. Computes indicators for each ticker, scores them,
and produces target weights. No LLM yet: pure rule-based signal aggregation.

This is intentional. We first prove the technical signals have merit on their
own before adding LLM reasoning on top. If the signals are worthless, the LLM
cannot save them.

Scoring logic per ticker (each component: -1 bearish, 0 neutral, +1 bullish):
  1. Trend:     short SMA above long SMA
  2. Momentum:  MACD histogram positive and increasing
  3. RSI:       not overbought (< 70), not oversold (> 30), mid-range bullish
  4. Bollinger: price not overextended (%B between 0.2 and 0.8)
  5. Volume:    volume confirming direction (above average on up days)

Final weight allocation:
  - Score each ticker from -5 to +5
  - Only go long tickers with score > 0
  - Weight proportional to score (positive scores only)
  - Cap any single position at MAX_WEIGHT
  - Remainder stays in cash (defensive when signals are weak)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from indicators import (
    sma, ema, rsi, macd, bollinger_bands, volume_sma_ratio, trend_strength
)

MAX_WEIGHT = 0.35   # no single ticker above 35% of portfolio
MIN_HISTORY = 60    # need at least 60 days to compute reliable indicators


def _score_ticker(prices: pd.Series) -> float:
    """
    Score a single ticker from -5 to +5 based on technical signals.
    Returns 0.0 if not enough history.
    """
    if len(prices) < MIN_HISTORY:
        return 0.0

    score = 0.0

    # 1. Trend: SMA20 vs SMA50
    ts = trend_strength(prices, short=20, long=50)
    if ts.isna().iloc[-1]:
        pass
    elif ts.iloc[-1] > 0.01:
        score += 1.0    # clear uptrend
    elif ts.iloc[-1] < -0.01:
        score -= 1.0    # clear downtrend

    # 2. Momentum: MACD histogram
    m = macd(prices)
    hist = m["histogram"]
    if not hist.isna().iloc[-1]:
        if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
            score += 1.0    # positive and rising
        elif hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]:
            score -= 1.0    # negative and falling

    # 3. RSI
    r = rsi(prices)
    if not r.isna().iloc[-1]:
        rv = r.iloc[-1]
        if 45 < rv < 65:
            score += 1.0    # healthy bullish range
        elif rv >= 70:
            score -= 0.5    # overbought, reduce enthusiasm
        elif rv <= 30:
            score -= 1.0    # oversold, avoid catching falling knife

    # 4. Bollinger %B
    bb = bollinger_bands(prices)
    pct_b = bb["pct_b"]
    if not pct_b.isna().iloc[-1]:
        pb = pct_b.iloc[-1]
        if 0.4 < pb < 0.8:
            score += 1.0    # price in upper half but not overextended
        elif pb > 1.0 or pb < 0.0:
            score -= 1.0    # outside the bands

    # 5. Short-term price momentum (5-day return)
    ret_5d = (prices.iloc[-1] / prices.iloc[-6] - 1) if len(prices) >= 6 else 0
    if ret_5d > 0.01:
        score += 1.0
    elif ret_5d < -0.01:
        score -= 1.0

    return score


def _scores_to_weights(scores: dict[str, float]) -> dict[str, float]:
    """Convert raw scores to portfolio weights."""
    positive = {t: s for t, s in scores.items() if s > 0}
    if not positive:
        return {}   # all cash

    total = sum(positive.values())
    weights = {t: s / total for t, s in positive.items()}

    # Apply position cap
    capped = {t: min(w, MAX_WEIGHT) for t, w in weights.items()}

    # Renormalise after capping
    total_capped = sum(capped.values())
    if total_capped > 0:
        capped = {t: w / total_capped for t, w in capped.items()}

    # Scale down to leave cash when conviction is low
    # (average score of selected tickers, normalised to 0-1)
    avg_score = sum(positive[t] * capped[t] for t in capped) / 5.0
    conviction = min(max(avg_score, 0.3), 1.0)
    return {t: w * conviction for t, w in capped.items()}


class TechnicalAgent:
    """
    Implements the strategy interface expected by run_backtest:
        target_weights(date, price_history) -> dict[ticker, weight] | None
    """
    def __init__(self, tickers: list[str], rebalance_days: int = 5):
        self.tickers = tickers
        self.rebalance_days = rebalance_days
        self._counter = 0
        self._started = False

    def target_weights(self,
                       date,
                       price_history: pd.DataFrame) -> dict | None:
        # Rebalance every N days
        if not self._started:
            self._started = True
            self._counter = 0
        else:
            self._counter += 1
            if self._counter < self.rebalance_days:
                return None
            self._counter = 0

        scores = {}
        for t in self.tickers:
            if t not in price_history.columns:
                scores[t] = 0.0
                continue
            prices = price_history[t].dropna()
            scores[t] = _score_ticker(prices)

        weights = _scores_to_weights(scores)
        return weights if weights else {}

    def last_scores(self, price_history: pd.DataFrame) -> dict[str, float]:
        """Utility: inspect current scores without triggering rebalance logic."""
        return {
            t: _score_ticker(price_history[t].dropna())
            for t in self.tickers
            if t in price_history.columns
        }
