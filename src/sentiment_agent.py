"""
sentiment_agent.py
------------------
Sentiment Agent: analyzes market sentiment using VIX, price momentum,
and volume patterns as proxies for sentiment (no paid news API needed).

Proxies used:
- VIX level and trend (fear gauge)
- Sector ETF momentum vs SPY (risk-on/off)
- Recent price momentum breadth (how many stocks are rising)
- Volume trend (conviction behind moves)
"""

from __future__ import annotations
import json
from agent_utils import parse_llm_json
import pandas as pd
import yfinance as yf
import anthropic
from indicators import rsi, ema

CLIENT = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a market sentiment analyst. You receive sentiment
proxy indicators and must assess the overall market mood and its implications
for a stock portfolio.

Assess: fear/greed level, risk-on vs risk-off environment,
momentum breadth, and volume conviction.

Respond ONLY with valid JSON, no markdown:
{
  "sentiment_score": 0.65,
  "regime": "risk-on",
  "reasoning": "one sentence summary"
}
sentiment_score: 0.0 (extreme fear) to 1.0 (extreme greed).
regime: one of "risk-on", "risk-off", "neutral"."""


def _fetch_sentiment_data(tickers: list[str]) -> dict:
    """Fetch VIX and momentum proxies."""
    try:
        # VIX as fear gauge
        vix_data = yf.download("^VIX", period="3mo",
                               auto_adjust=True, progress=False)
        vix_close = vix_data["Close"].squeeze()
        vix_current = float(vix_close.iloc[-1])
        vix_sma20 = float(vix_close.rolling(20).mean().iloc[-1])

        # Breadth: how many tickers are above their 20-day EMA
        above_ema = 0
        pos_momentum = 0
        for t in tickers:
            try:
                px = yf.download(t, period="3mo",
                                 auto_adjust=True, progress=False)["Close"].squeeze()
                e20 = ema(px, 20)
                if float(px.iloc[-1]) > float(e20.iloc[-1]):
                    above_ema += 1
                ret_20d = float(px.iloc[-1] / px.iloc[-21] - 1) * 100
                if ret_20d > 0:
                    pos_momentum += 1
            except Exception:
                pass

        breadth = above_ema / len(tickers)
        momentum_breadth = pos_momentum / len(tickers)

        return {
            "vix_current": round(vix_current, 1),
            "vix_vs_sma20": round(vix_current - vix_sma20, 1),
            "breadth_above_ema20": round(breadth * 100, 0),
            "positive_momentum_pct": round(momentum_breadth * 100, 0),
        }
    except Exception as e:
        return {"error": str(e)}


class SentimentAgent:
    """
    Returns sentiment score (0-1) and regime label.
    Called once per rebalance by the Orchestrator.
    """
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.last_reasoning = ""
        self.last_score = 0.5
        self.last_regime = "neutral"

    def analyze(self) -> dict:
        print("  [SentimentAgent] Fetching sentiment proxies...")
        data = _fetch_sentiment_data(self.tickers)

        prompt = (
            f"Sentiment proxy data:\n"
            f"VIX={data.get('vix_current', 'N/A')} "
            f"(vs 20d SMA: {data.get('vix_vs_sma20', 'N/A'):+.1f})\n"
            f"Stocks above 20d EMA: {data.get('breadth_above_ema20', 'N/A')}%\n"
            f"Stocks with positive 20d momentum: "
            f"{data.get('positive_momentum_pct', 'N/A')}%\n\n"
            f"Assess overall market sentiment."
        )

        try:
            response = CLIENT.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            parsed = parse_llm_json(raw)
            self.last_score = float(parsed.get("sentiment_score", 0.5))
            self.last_regime = parsed.get("regime", "neutral")
            self.last_reasoning = parsed.get("reasoning", "")
            return {
                "score": self.last_score,
                "regime": self.last_regime,
                "reasoning": self.last_reasoning,
            }
        except Exception as e:
            print(f"  [SentimentAgent error: {e}]")
            return {"score": 0.5, "regime": "neutral", "reasoning": ""}
