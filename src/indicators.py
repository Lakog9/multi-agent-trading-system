"""
indicators.py
-------------
Technical indicators computed from a price+volume DataFrame.
Each function takes a price Series (or DataFrame) and returns a Series.

All indicators are computed only from data up to the current bar,
so they are safe to use inside the point-in-time harness.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return prices.rolling(window).mean()


def ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return prices.ewm(span=span, adjust=False).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index (0-100).
    Above 70: overbought. Below 30: oversold.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(prices: pd.Series,
         fast: int = 12,
         slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.
    Positive histogram: bullish momentum. Negative: bearish.
    """
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def bollinger_bands(prices: pd.Series,
                    window: int = 20,
                    n_std: float = 2.0) -> pd.DataFrame:
    """
    Bollinger Bands: middle (SMA), upper, lower, and %B position.
    %B = 0: price at lower band. %B = 1: price at upper band.
    %B > 1 or < 0: price outside the bands (potential reversal signal).
    """
    mid = sma(prices, window)
    std = prices.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    bandwidth = (upper - lower) / mid
    pct_b = (prices - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "mid": mid,
        "upper": upper,
        "lower": lower,
        "pct_b": pct_b,
        "bandwidth": bandwidth,
    })


def atr(high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 14) -> pd.Series:
    """
    Average True Range: measures volatility.
    Higher ATR = more volatile = wider stops needed.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/window, adjust=False).mean()


def volume_sma_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Current volume divided by its moving average.
    > 1.5: unusually high volume (confirms breakouts).
    < 0.5: unusually low volume (weak signal).
    """
    return volume / volume.rolling(window).mean()


def trend_strength(prices: pd.Series,
                   short: int = 20,
                   long: int = 50) -> pd.Series:
    """
    (short SMA - long SMA) / long SMA.
    Positive: uptrend. Negative: downtrend.
    Magnitude indicates strength.
    """
    return (sma(prices, short) - sma(prices, long)) / sma(prices, long)
