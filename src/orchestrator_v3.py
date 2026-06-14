"""
orchestrator_v3.py
------------------
Full system: Technical + Fundamental + Sentiment + Macro
           + Bull/Bear Debate + Trader Agent + Risk Manager

This is the complete Phase 5 council.
"""

from __future__ import annotations
import json
import pandas as pd

from llm_technical_agent import _build_indicator_report
from fundamental_agent import FundamentalAgent
from sentiment_agent import SentimentAgent
from macro_agent import MacroAgent
from debate_agent import run_debate
from trader_agent import compute_weights
from risk_manager import RiskManager
from indicators import rsi, macd, trend_strength


class OrchestratorV3:
    def __init__(self, tickers: list[str], rebalance_days: int = 10):
        self.tickers = tickers
        self.rebalance_days = rebalance_days
        self._counter = 0
        self._started = False

        self.fund_agent  = FundamentalAgent(tickers)
        self.sent_agent  = SentimentAgent(tickers)
        self.macro_agent = MacroAgent()
        self.risk_mgr    = RiskManager()

        self.last_weights   = {}
        self.last_reasoning = ""
        self.decision_log   = []

    def _tech_scores(self, price_history: pd.DataFrame) -> dict[str, float]:
        scores = {}
        for t in self.tickers:
            if t not in price_history.columns:
                scores[t] = 0.0
                continue
            px = price_history[t].dropna()
            if len(px) < 50:
                scores[t] = 0.0
                continue
            score = 0.0
            ts = trend_strength(px).iloc[-1]
            if not pd.isna(ts):
                score += 1.0 if ts > 0.01 else (-1.0 if ts < -0.01 else 0.0)
            hist = macd(px)["histogram"]
            if not pd.isna(hist.iloc[-1]):
                score += (1.0 if (hist.iloc[-1] > 0 and
                          hist.iloc[-1] > hist.iloc[-2])
                          else (-1.0 if (hist.iloc[-1] < 0 and
                          hist.iloc[-1] < hist.iloc[-2]) else 0.0))
            r = rsi(px).iloc[-1]
            if not pd.isna(r):
                score += (1.0 if 45 < r < 65
                          else (-0.5 if r >= 70
                          else (-1.0 if r <= 30 else 0.0)))
            scores[t] = score
        return scores

    def target_weights(self, date, price_history: pd.DataFrame) -> dict | None:
        if not self._started:
            self._started = True
            self._counter = 0
        else:
            self._counter += 1
            if self._counter < self.rebalance_days:
                return None
            self._counter = 0

        print(f"\n[OrchestratorV3] Rebalancing on {date}")

        # 1. Technical
        print("  [1/7] Technical indicators...")
        tech_report = _build_indicator_report(price_history, self.tickers)
        tech_scores = self._tech_scores(price_history)

        # 2. Fundamental
        print("  [2/7] Fundamental analysis...")
        fund_scores = self.fund_agent.analyze()

        # 3. Sentiment
        print("  [3/7] Sentiment analysis...")
        sentiment = self.sent_agent.analyze()

        # 4. Macro
        print("  [4/7] Macro analysis...")
        macro = self.macro_agent.analyze()

        # 5. Debate
        print("  [5/7] Bull/Bear debate...")
        evidence = (
            f"Technical:\n{tech_report}\n\n"
            f"Fundamentals: "
            + ", ".join(f"{t}={fund_scores.get(t,0.5):.2f}"
                        for t in self.tickers)
            + f"\nSentiment: {sentiment['regime']} "
            f"(score={sentiment['score']:.2f})\n"
            f"Macro: {macro['regime']}, "
            f"risk_appetite={macro['risk_appetite']:.2f}"
        )
        debate = run_debate(self.tickers, evidence)
        print(f"  Stance: {debate.get('overall_stance','N/A')}")

        # 6. Trader
        print("  [6/7] Trader Agent...")
        trade = compute_weights(
            self.tickers, debate,
            tech_scores, fund_scores,
            sentiment, macro,
        )
        proposed = trade["weights"]
        print(f"  Proposed: {proposed}")

        # 7. Risk Manager (hard constraints + LLM soft check)
        print("  [7/7] Risk Manager review...")
        final = self.risk_mgr.review(
            proposed, price_history,
            macro_regime=macro.get("regime", "mid_cycle"),
            date=date,
        )

        self.last_weights   = final
        self.last_reasoning = trade["reasoning"]

        print(f"  Final:    {final}")
        print(f"  Cash: {1-sum(final.values()):.0%} | "
              f"Conviction: {trade.get('conviction',0):.2f}")

        self.decision_log.append({
            "date":       str(date),
            "proposed":   proposed,
            "final":      final,
            "conviction": trade.get("conviction", 0.5),
            "stance":     debate.get("overall_stance", ""),
            "sentiment":  sentiment["regime"],
            "macro":      macro["regime"],
            "reasoning":  self.last_reasoning,
        })

        return final if final else {}
