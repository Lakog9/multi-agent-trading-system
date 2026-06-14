"""
macro_agent.py
--------------
Macro Agent: assesses the broad economic regime using publicly
available market proxies (no paid data needed).

Proxies used:
- 10Y Treasury yield level and trend (interest rate environment)
- 2Y vs 10Y spread (yield curve: inversion = recession risk)
- DXY (dollar strength: affects international earnings)
- Gold vs SPY ratio (risk-off signal)
- Sector rotation: XLF, XLE, XLK, XLV relative to SPY
"""

from __future__ import annotations
import json
from agent_utils import parse_llm_json
import yfinance as yf
import anthropic

CLIENT = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a macro economist and portfolio strategist.
You receive macroeconomic proxy indicators and must classify the current
economic regime and its implications for a stock portfolio.

Assess: interest rate environment, yield curve shape, dollar trend,
risk appetite, and which sectors are favored in this regime.

Respond ONLY with valid JSON, no markdown:
{
  "regime": "late_cycle",
  "rate_environment": "high_and_falling",
  "favored_sectors": ["Technology", "Healthcare"],
  "risk_appetite": 0.6,
  "reasoning": "one sentence summary"
}
regime: one of early_cycle, mid_cycle, late_cycle, recession
rate_environment: one of low_and_rising, high_and_rising, high_and_falling, low_and_stable
risk_appetite: 0.0 (very defensive) to 1.0 (very aggressive)"""


def _fetch_macro_data() -> dict:
    """Fetch macro proxy data from yfinance."""
    results = {}
    try:
        # 10Y Treasury yield
        tnx = yf.download("^TNX", period="6mo",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        results["yield_10y"] = round(float(tnx.iloc[-1]), 2)
        results["yield_10y_3m_change"] = round(
            float(tnx.iloc[-1] - tnx.iloc[-63]), 2)

        # 2Y Treasury yield
        irx = yf.download("^IRX", period="6mo",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        results["yield_2y"] = round(float(irx.iloc[-1]) / 10, 2)
        results["yield_curve_2s10s"] = round(
            results["yield_10y"] - results["yield_2y"], 2)

        # DXY (dollar index)
        dxy = yf.download("DX-Y.NYB", period="3mo",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        results["dxy_current"] = round(float(dxy.iloc[-1]), 1)
        results["dxy_1m_change"] = round(
            float(dxy.iloc[-1] / dxy.iloc[-21] - 1) * 100, 1)

        # Gold vs SPY ratio (risk-off indicator)
        gold = yf.download("GLD", period="3mo",
                           auto_adjust=True, progress=False)["Close"].squeeze()
        spy = yf.download("SPY", period="3mo",
                          auto_adjust=True, progress=False)["Close"].squeeze()
        ratio = gold / spy
        results["gold_spy_ratio_trend"] = round(
            float(ratio.iloc[-1] / ratio.iloc[-21] - 1) * 100, 1)

        # Sector rotation (vs SPY, 1 month)
        sectors = {"Tech": "XLK", "Financials": "XLF",
                   "Energy": "XLE", "Healthcare": "XLV"}
        spy_ret = float(spy.iloc[-1] / spy.iloc[-21] - 1)
        sector_rel = {}
        for name, etf in sectors.items():
            try:
                s = yf.download(etf, period="3mo",
                                auto_adjust=True, progress=False)["Close"].squeeze()
                rel = float(s.iloc[-1] / s.iloc[-21] - 1) - spy_ret
                sector_rel[name] = round(rel * 100, 1)
            except Exception:
                pass
        results["sector_vs_spy_1m"] = sector_rel

    except Exception as e:
        results["error"] = str(e)

    return results


class MacroAgent:
    """
    Returns macro regime assessment.
    Called once per rebalance by the Orchestrator.
    """
    def __init__(self):
        self.last_reasoning = ""
        self.last_regime = "mid_cycle"
        self.last_risk_appetite = 0.5
        self.last_favored_sectors = []

    def analyze(self) -> dict:
        print("  [MacroAgent] Fetching macro data...")
        data = _fetch_macro_data()

        prompt = (
            f"Macro indicators:\n"
            f"10Y yield: {data.get('yield_10y', 'N/A')}% "
            f"(3m change: {data.get('yield_10y_3m_change', 'N/A'):+.2f}%)\n"
            f"2Y yield: {data.get('yield_2y', 'N/A')}%\n"
            f"Yield curve (2s10s): {data.get('yield_curve_2s10s', 'N/A')}%\n"
            f"DXY: {data.get('dxy_current', 'N/A')} "
            f"(1m change: {data.get('dxy_1m_change', 'N/A'):+.1f}%)\n"
            f"Gold/SPY ratio 1m change: "
            f"{data.get('gold_spy_ratio_trend', 'N/A'):+.1f}%\n"
            f"Sector vs SPY (1m): {data.get('sector_vs_spy_1m', {})}\n\n"
            f"Classify the macro regime."
        )

        try:
            response = CLIENT.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            parsed = parse_llm_json(raw)
            self.last_regime = parsed.get("regime", "mid_cycle")
            self.last_risk_appetite = float(parsed.get("risk_appetite", 0.5))
            self.last_favored_sectors = parsed.get("favored_sectors", [])
            self.last_reasoning = parsed.get("reasoning", "")
            return {
                "regime": self.last_regime,
                "rate_environment": parsed.get("rate_environment", "unknown"),
                "favored_sectors": self.last_favored_sectors,
                "risk_appetite": self.last_risk_appetite,
                "reasoning": self.last_reasoning,
            }
        except Exception as e:
            print(f"  [MacroAgent error: {e}]")
            return {
                "regime": "mid_cycle",
                "risk_appetite": 0.5,
                "favored_sectors": [],
                "reasoning": "",
            }
