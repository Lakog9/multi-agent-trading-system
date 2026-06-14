"""
orchestrator_v2.py
------------------
Upgraded Orchestrator with Bull/Bear debate and Trader Agent.

Flow per rebalance:
1. Technical Agent  -> per-ticker technical scores + indicators
2. Fundamental Agent -> per-ticker fundamental scores
3. Sentiment Agent  -> market sentiment + regime
4. Macro Agent      -> economic regime + risk appetite
5. Debate Agent     -> Bull vs Bear (2 rounds) -> synthesis thesis
6. Trader Agent     -> translates thesis into final weights
"""

from __future__ import annotations
import json
import pandas as pd
import anthropic

from llm_technical_agent import _build_indicator_report
from fundamental_agent import FundamentalAgent
from sentiment_agent import SentimentAgent
from macro_agent import MacroAgent
from debate_agent import run_debate
from trader_agent import compute_weights

CLIENT = anthropic.Anthropic()


class OrchestratorV2:
    """
    Full council: Technical + Fundamental + Sentiment + Macro
                  + Bull/Bear Debate + Trader Agent.
    """
    def __init__(self, tickers: list[str], rebalance_days: int = 10):
        self.tickers = tickers
        self.rebalance_days = rebalance_days
        self._counter = 0
        self._started = False

        self.fund_agent  = FundamentalAgent(tickers)
        self.sent_agent  = SentimentAgent(tickers)
        self.macro_agent = MacroAgent()

        self.last_weights   = {}
        self.last_reasoning = ""
        self.decision_log   = []

    def target_weights(self, date, price_history: pd.DataFrame) -> dict | None:
        if not self._started:
            self._started = True
            self._counter = 0
        else:
            self._counter += 1
            if self._counter < self.rebalance_days:
                return None
            self._counter = 0

        print(f"\n[OrchestratorV2] Rebalancing on {date}")

        # 1. Technical indicators
        print("  [1/6] Technical indicators...")
        tech_report = _build_indicator_report(price_history, self.tickers)

        # Simple tech scores for Trader Agent input
        from indicators import rsi, macd, trend_strength
        tech_scores = {}
        for t in self.tickers:
            if t not in price_history.columns:
                tech_scores[t] = 0.0
                continue
            px = price_history[t].dropna()
            if len(px) < 50:
                tech_scores[t] = 0.0
                continue
            score = 0.0
            ts = trend_strength(px).iloc[-1]
            if not pd.isna(ts):
                score += 1.0 if ts > 0.01 else (-1.0 if ts < -0.01 else 0.0)
            hist = macd(px)["histogram"]
            if not pd.isna(hist.iloc[-1]):
                score += 1.0 if (hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]) else (
                         -1.0 if (hist.iloc[-1] < 0 and hist.iloc[-1] < hist.iloc[-2]) else 0.0)
            r = rsi(px).iloc[-1]
            if not pd.isna(r):
                score += 1.0 if 45 < r < 65 else (-0.5 if r >= 70 else (-1.0 if r <= 30 else 0.0))
            tech_scores[t] = score

        # 2. Fundamental
        print("  [2/6] Fundamental analysis...")
        fund_scores = self.fund_agent.analyze()

        # 3. Sentiment
        print("  [3/6] Sentiment analysis...")
        sentiment = self.sent_agent.analyze()

        # 4. Macro
        print("  [4/6] Macro analysis...")
        macro = self.macro_agent.analyze()

        # 5. Bull/Bear debate
        print("  [5/6] Bull/Bear debate (2 rounds)...")
        evidence = (
            f"Technical indicators:\n{tech_report}\n\n"
            f"Fundamental scores: "
            + ", ".join(f"{t}={fund_scores.get(t,0.5):.2f}"
                        for t in self.tickers)
            + f"\n\nSentiment: {sentiment['regime']} "
            f"(score={sentiment['score']:.2f})\n"
            f"Macro: {macro['regime']}, "
            f"risk_appetite={macro['risk_appetite']:.2f}, "
            f"favored={macro['favored_sectors']}"
        )
        debate = run_debate(self.tickers, evidence)
        print(f"  Debate stance: {debate.get('overall_stance', 'N/A')}")
        print(f"  Bull: {debate.get('bull_case','')[:120]}...")
        print(f"  Bear: {debate.get('bear_case','')[:120]}...")

        # 6. Trader Agent
        print("  [6/6] Trader Agent computing weights...")
        trade = compute_weights(
            self.tickers, debate,
            tech_scores, fund_scores,
            sentiment, macro,
        )

        weights = trade["weights"]
        self.last_weights   = weights
        self.last_reasoning = trade["reasoning"]

        print(f"  Final weights: {weights}")
        print(f"  Cash: {trade.get('cash', 0):.0%} | "
              f"Conviction: {trade.get('conviction', 0):.2f}")
        print(f"  Reasoning: {self.last_reasoning}")

        self.decision_log.append({
            "date":         str(date),
            "weights":      weights,
            "cash":         trade.get("cash", 0),
            "conviction":   trade.get("conviction", 0.5),
            "stance":       debate.get("overall_stance", ""),
            "key_risks":    debate.get("key_risks", ""),
            "key_opps":     debate.get("key_opportunities", ""),
            "sentiment":    sentiment["regime"],
            "macro":        macro["regime"],
            "reasoning":    self.last_reasoning,
        })

        return weights if weights else {}
