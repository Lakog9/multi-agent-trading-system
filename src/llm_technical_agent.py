"""
llm_technical_agent.py
----------------------
Technical Agent powered by Claude API.
Computes indicators, sends them as a structured report to Claude,
and asks for target weights with reasoning.

Cost control:
- Rebalances every 5 trading days (not daily)
- One API call per rebalance (not per ticker)
- Caches the last decision for inspection
"""

from __future__ import annotations
import json
import pandas as pd
import anthropic
from indicators import rsi, macd, bollinger_bands, trend_strength

MIN_HISTORY = 60
REBALANCE_DAYS = 5
MAX_WEIGHT = 0.35
CLIENT = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a quantitative technical analyst managing a stock portfolio.
You receive technical indicator readings for a set of stocks and must allocate portfolio weights.

Rules:
- Weights must sum to 1.0 or less (remainder is cash)
- No single stock above 0.35
- Be defensive (hold cash) when signals are weak or mixed
- Respond ONLY with valid JSON, no explanation, no markdown

Output format:
{
  "weights": {"AAPL": 0.20, "MSFT": 0.15, ...},
  "reasoning": "one sentence summary"
}"""


def _build_indicator_report(prices: pd.DataFrame, tickers: list[str]) -> str:
    """Build a concise indicator summary for all tickers."""
    lines = []
    for t in tickers:
        if t not in prices.columns:
            continue
        px = prices[t].dropna()
        if len(px) < MIN_HISTORY:
            lines.append(f"{t}: insufficient history")
            continue

        r = rsi(px).iloc[-1]
        m = macd(px)
        hist = m["histogram"].iloc[-1]
        hist_prev = m["histogram"].iloc[-2]
        bb = bollinger_bands(px)
        pct_b = bb["pct_b"].iloc[-1]
        ts = trend_strength(px).iloc[-1]
        ret_5d = (px.iloc[-1] / px.iloc[-6] - 1) * 100

        lines.append(
            f"{t}: RSI={r:.1f}, MACD_hist={hist:.3f} ({'rising' if hist > hist_prev else 'falling'}), "
            f"BB%={pct_b:.2f}, Trend={ts:.3f}, 5d_return={ret_5d:.1f}%"
        )
    return "\n".join(lines)


class LLMTechnicalAgent:
    """
    Strategy interface: target_weights(date, price_history) -> dict | None
    """
    def __init__(self, tickers: list[str], rebalance_days: int = REBALANCE_DAYS):
        self.tickers = tickers
        self.rebalance_days = rebalance_days
        self._counter = 0
        self._started = False
        self.last_reasoning = ""
        self.last_weights = {}

    def target_weights(self, date, price_history: pd.DataFrame) -> dict | None:
        if not self._started:
            self._started = True
            self._counter = 0
        else:
            self._counter += 1
            if self._counter < self.rebalance_days:
                return None
            self._counter = 0

        report = _build_indicator_report(price_history, self.tickers)
        prompt = f"Date: {date}\n\nTechnical indicators:\n{report}\n\nAllocate portfolio weights."

        try:
            response = CLIENT.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            parsed = json.loads(raw)
            weights = {k: float(v) for k, v in parsed.get("weights", {}).items()
                       if k in self.tickers}
            self.last_reasoning = parsed.get("reasoning", "")
            self.last_weights = weights
            return weights if weights else {}
        except Exception as e:
            print(f"  [LLM error on {date}: {e}]")
            return {}

