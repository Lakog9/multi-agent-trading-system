"""
fundamental_agent.py
--------------------
Fundamental Agent: analyzes valuation, growth, and balance sheet health.
Uses yfinance to fetch financials, sends structured report to Claude.
Returns a score and weight recommendation per ticker.
"""

from __future__ import annotations
import json
from agent_utils import parse_llm_json
import yfinance as yf
import anthropic

CLIENT = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a fundamental analyst. You receive financial metrics
for a set of stocks and must assess their investment attractiveness.

Analyze: valuation (P/E vs sector), growth (revenue/earnings trend),
profitability (margins), and balance sheet health (debt).

Respond ONLY with valid JSON, no markdown:
{
  "scores": {"AAPL": 0.7, "MSFT": 0.4, ...},
  "reasoning": "one sentence summary"
}
Scores range from 0.0 (avoid) to 1.0 (strong buy). Use 0.5 for neutral."""


def _fetch_fundamentals(ticker: str) -> dict:
    """Fetch key fundamental metrics from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio":        round(info.get("trailingPE", 0) or 0, 1),
            "forward_pe":      round(info.get("forwardPE", 0) or 0, 1),
            "revenue_growth":  round((info.get("revenueGrowth", 0) or 0) * 100, 1),
            "earnings_growth": round((info.get("earningsGrowth", 0) or 0) * 100, 1),
            "profit_margin":   round((info.get("profitMargins", 0) or 0) * 100, 1),
            "debt_to_equity":  round(info.get("debtToEquity", 0) or 0, 1),
            "roe":             round((info.get("returnOnEquity", 0) or 0) * 100, 1),
            "sector":          info.get("sector", "Unknown"),
        }
    except Exception:
        return {}


def _build_fundamental_report(tickers: list[str]) -> str:
    lines = []
    for t in tickers:
        data = _fetch_fundamentals(t)
        if not data:
            lines.append(f"{t}: data unavailable")
            continue
        lines.append(
            f"{t} [{data['sector']}]: "
            f"P/E={data['pe_ratio']} (fwd={data['forward_pe']}), "
            f"Rev growth={data['revenue_growth']}%, "
            f"EPS growth={data['earnings_growth']}%, "
            f"Margin={data['profit_margin']}%, "
            f"D/E={data['debt_to_equity']}, "
            f"ROE={data['roe']}%"
        )
    return "\n".join(lines)


class FundamentalAgent:
    """
    Returns fundamental scores per ticker (0.0 to 1.0).
    Called once per rebalance by the Orchestrator.
    """
    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.last_reasoning = ""
        self.last_scores = {}

    def analyze(self) -> dict[str, float]:
        print("  [FundamentalAgent] Fetching financials...")
        report = _build_fundamental_report(self.tickers)
        prompt = f"Fundamental metrics:\n{report}\n\nScore each stock."

        try:
            response = CLIENT.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            parsed = parse_llm_json(raw)
            scores = {k: float(v) for k, v in parsed.get("scores", {}).items()
                      if k in self.tickers}
            self.last_reasoning = parsed.get("reasoning", "")
            self.last_scores = scores
            return scores
        except Exception as e:
            print(f"  [FundamentalAgent error: {e}]")
            return {t: 0.5 for t in self.tickers}
