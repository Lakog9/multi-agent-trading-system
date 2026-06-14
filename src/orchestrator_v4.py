"""
orchestrator_v4.py
------------------
Full system with Memory + Reflection (Phase 6).

New vs V3:
  - Before deciding, retrieves relevant past lessons and injects them
    into the Trader Agent's context.
  - After each decision's outcome is known (next rebalance), reflects
    and writes a lesson into long-term memory.

This closes the learning loop: the agent's own history shapes future choices.
"""

from __future__ import annotations
import json
import pandas as pd
from datetime import datetime

from llm_technical_agent import _build_indicator_report
from fundamental_agent import FundamentalAgent
from sentiment_agent import SentimentAgent
from macro_agent import MacroAgent
from debate_agent import run_debate
from trader_agent import compute_weights
from risk_manager import RiskManager
from memory import MemoryStore
from reflection import reflect_on_outcome, build_memory_context
from indicators import rsi, macd, trend_strength


class OrchestratorV4:
    def __init__(self, tickers: list[str], rebalance_days: int = 10,
                 memory_path: str = "memory_store.json"):
        self.tickers = tickers
        self.rebalance_days = rebalance_days
        self._counter = 0
        self._started = False

        self.fund_agent  = FundamentalAgent(tickers)
        self.sent_agent  = SentimentAgent(tickers)
        self.macro_agent = MacroAgent()
        self.risk_mgr    = RiskManager()
        self.memory      = MemoryStore(memory_path)

        self.last_weights   = {}
        self.last_reasoning = ""
        self.decision_log   = []

        # Track pending decision for reflection on next rebalance
        self._pending_reflection = None
        self._portfolio_value_at_decision = None

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

    def _portfolio_value(self, weights, price_history):
        """Proxy: weighted price level, used to estimate realized return."""
        val = 0.0
        for t, w in weights.items():
            if t in price_history.columns:
                val += w * float(price_history[t].dropna().iloc[-1])
        return val

    def target_weights(self, date, price_history: pd.DataFrame) -> dict | None:
        if not self._started:
            self._started = True
            self._counter = 0
        else:
            self._counter += 1
            if self._counter < self.rebalance_days:
                return None
            self._counter = 0

        now = pd.Timestamp(date).to_pydatetime()
        print(f"\n[OrchestratorV4] Rebalancing on {date}")

        # === REFLECTION on previous decision (now that we see the outcome) ===
        if self._pending_reflection is not None:
            prev = self._pending_reflection
            prev_val = prev["portfolio_proxy"]
            curr_val = self._portfolio_value(prev["final"], price_history)
            bh_val_prev = prev["bh_proxy"]
            bh_val_curr = sum(
                float(price_history[t].dropna().iloc[-1])
                for t in self.tickers if t in price_history.columns
            ) / len(self.tickers)

            realized = (curr_val / prev_val - 1) if prev_val else 0.0
            bh_return = (bh_val_curr / bh_val_prev - 1) if bh_val_prev else 0.0

            print("  [Reflection] Reviewing previous decision...")
            lesson = reflect_on_outcome(prev, realized, bh_return, now)
            if lesson["lesson"]:
                layer = "long" if lesson["importance"] >= 0.7 else "mid"
                self.memory.add(
                    content=f"[{prev['date'][:10]}] {lesson['lesson']}",
                    layer=layer,
                    importance=lesson["importance"],
                    tickers=list(prev["final"].keys()),
                    now=now,
                )
                print(f"  [Reflection] Lesson ({lesson['lesson_type']}, "
                      f"imp={lesson['importance']:.2f}): {lesson['lesson']}")
            self._pending_reflection = None

        # === RETRIEVE relevant past lessons ===
        query = f"trading decision {' '.join(self.tickers)}"
        memories = self.memory.retrieve(query, now, top_k=4)
        memory_context = build_memory_context(memories)
        if memories:
            print(f"  [Memory] Retrieved {len(memories)} relevant lesson(s)")

        # === STANDARD PIPELINE ===
        print("  [1/7] Technical...")
        tech_report = _build_indicator_report(price_history, self.tickers)
        tech_scores = self._tech_scores(price_history)

        print("  [2/7] Fundamental...")
        fund_scores = self.fund_agent.analyze()

        print("  [3/7] Sentiment...")
        sentiment = self.sent_agent.analyze()

        print("  [4/7] Macro...")
        macro = self.macro_agent.analyze()

        print("  [5/7] Debate...")
        evidence = (
            f"Technical:\n{tech_report}\n\n"
            f"Fundamentals: "
            + ", ".join(f"{t}={fund_scores.get(t,0.5):.2f}"
                        for t in self.tickers)
            + f"\nSentiment: {sentiment['regime']} "
            f"(score={sentiment['score']:.2f})\n"
            f"Macro: {macro['regime']}, "
            f"risk_appetite={macro['risk_appetite']:.2f}\n\n"
            f"=== PAST LESSONS ===\n{memory_context}"
        )
        debate = run_debate(self.tickers, evidence)
        print(f"  Stance: {debate.get('overall_stance','N/A')}")

        print("  [6/7] Trader...")
        trade = compute_weights(
            self.tickers, debate,
            tech_scores, fund_scores,
            sentiment, macro,
        )
        proposed = trade["weights"]

        print("  [7/7] Risk Manager...")
        final = self.risk_mgr.review(
            proposed, price_history,
            macro_regime=macro.get("regime", "mid_cycle"),
            date=date,
        )

        self.last_weights   = final
        self.last_reasoning = trade["reasoning"]
        print(f"  Final: {final}")

        # Store this decision for reflection next time
        decision = {
            "date":       str(date),
            "final":      final,
            "proposed":   proposed,
            "stance":     debate.get("overall_stance", ""),
            "sentiment":  sentiment["regime"],
            "macro":      macro["regime"],
            "reasoning":  trade["reasoning"],
        }
        self.decision_log.append(decision)

        self._pending_reflection = {
            **decision,
            "portfolio_proxy": self._portfolio_value(final, price_history),
            "bh_proxy": sum(
                float(price_history[t].dropna().iloc[-1])
                for t in self.tickers if t in price_history.columns
            ) / len(self.tickers),
        }

        # Periodic memory housekeeping
        self.memory.prune(now)

        return final if final else {}

    def save_memory(self):
        self.memory.save()
        print(f"  [Memory] Saved {len(self.memory)} entries to "
              f"{self.memory.path}")
