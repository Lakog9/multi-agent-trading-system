"""
orchestrator.py
---------------
The Orchestrator: collects reports from all specialist agents and
synthesizes them into final portfolio weights.

Flow per rebalance:
1. Technical Agent  -> per-ticker technical scores
2. Fundamental Agent -> per-ticker fundamental scores
3. Sentiment Agent  -> market sentiment score + regime
4. Macro Agent      -> macro regime + risk appetite
5. Orchestrator     -> synthesizes all into final weights
"""

from __future__ import annotations
import json
from agent_utils import parse_llm_json
import pandas as pd
import anthropic

from llm_technical_agent import LLMTechnicalAgent, _build_indicator_report
from fundamental_agent import FundamentalAgent
from sentiment_agent import SentimentAgent
from macro_agent import MacroAgent

CLIENT = anthropic.Anthropic()

SYSTEM_PROMPT = """You are the chief portfolio manager of a quantitative fund.
You receive reports from four specialist agents:
- Technical Agent: short-term price signals per stock
- Fundamental Agent: valuation and growth scores per stock
- Sentiment Agent: overall market mood and regime
- Macro Agent: economic cycle and rate environment

Synthesize all inputs into final portfolio weights.

Rules:
- Weights must sum to 1.0 or less (remainder is cash)
- No single stock above 0.35
- Reduce overall exposure when macro or sentiment is bearish
- Favor stocks that score well on BOTH technical and fundamental
- Be explicit about which inputs drove the final decision

Respond ONLY with valid JSON, no markdown:
{
  "weights": {"AAPL": 0.20, "MSFT": 0.25, ...},
  "cash": 0.15,
  "conviction": 0.7,
  "reasoning": "two sentence summary of key drivers"
}"""


class Orchestrator:
    """
    Coordinates all agents and implements the strategy interface
    expected by run_backtest.
    """
    def __init__(self, tickers: list[str], rebalance_days: int = 5):
        self.tickers = tickers
        self.rebalance_days = rebalance_days
        self._counter = 0
        self._started = False

        self.tech_agent = LLMTechnicalAgent(tickers, rebalance_days=99999)
        self.fund_agent = FundamentalAgent(tickers)
        self.sent_agent = SentimentAgent(tickers)
        self.macro_agent = MacroAgent()

        self.last_weights = {}
        self.last_reasoning = ""
        self.decision_log = []

    def target_weights(self, date, price_history: pd.DataFrame) -> dict | None:
        if not self._started:
            self._started = True
            self._counter = 0
        else:
            self._counter += 1
            if self._counter < self.rebalance_days:
                return None
            self._counter = 0

        print(f"\n[Orchestrator] Rebalancing on {date}")

        # 1. Technical signals
        tech_report = _build_indicator_report(price_history, self.tickers)
        tech_weights = self.tech_agent.target_weights(date, price_history) or {}

        # 2. Fundamental scores (fetched fresh, cached by yfinance)
        fund_scores = self.fund_agent.analyze()

        # 3. Sentiment
        sentiment = self.sent_agent.analyze()

        # 4. Macro
        macro = self.macro_agent.analyze()

        # 5. Synthesize
        prompt = (
            f"Date: {date}\n\n"
            f"=== TECHNICAL REPORT ===\n{tech_report}\n"
            f"Technical suggested weights: {tech_weights}\n\n"
            f"=== FUNDAMENTAL SCORES (0-1) ===\n"
            + "\n".join(f"  {t}: {fund_scores.get(t, 0.5):.2f}"
                        for t in self.tickers)
            + f"\nFundamental reasoning: {self.fund_agent.last_reasoning}\n\n"
            f"=== SENTIMENT ===\n"
            f"Score: {sentiment['score']:.2f}, "
            f"Regime: {sentiment['regime']}\n"
            f"Reasoning: {sentiment['reasoning']}\n\n"
            f"=== MACRO ===\n"
            f"Regime: {macro['regime']}, "
            f"Risk appetite: {macro['risk_appetite']:.2f}\n"
            f"Favored sectors: {macro['favored_sectors']}\n"
            f"Reasoning: {macro['reasoning']}\n\n"
            f"Synthesize into final portfolio weights."
        )

        try:
            response = CLIENT.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            parsed = parse_llm_json(raw)
            weights = {k: float(v) for k, v in parsed.get("weights", {}).items()
                       if k in self.tickers}
            self.last_weights = weights
            self.last_reasoning = parsed.get("reasoning", "")

            self.decision_log.append({
                "date": str(date),
                "weights": weights,
                "conviction": parsed.get("conviction", 0.5),
                "reasoning": self.last_reasoning,
                "sentiment": sentiment["regime"],
                "macro": macro["regime"],
            })

            print(f"  Weights: {weights}")
            print(f"  Reasoning: {self.last_reasoning}")
            return weights

        except Exception as e:
            print(f"  [Orchestrator error: {e}]")
            return {}
