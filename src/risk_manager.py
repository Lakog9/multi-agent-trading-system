"""
risk_manager.py
---------------
Risk Management: the last line of defense before execution.
The Risk Officer has hard veto power over the Trader Agent.

Hard constraints (always enforced, no exceptions):
  1. No single position above MAX_POSITION (35%)
  2. Total gross exposure below MAX_GROSS (95%)
  3. Minimum cash floor MIN_CASH (5%)
  4. Circuit breaker: if portfolio drawdown exceeds MAX_DD_THRESHOLD,
     go to minimum exposure until drawdown recovers
  5. Bear regime override: reduce max gross exposure in bear markets

Soft constraints (LLM-based, can be overridden by hard constraints):
  - Portfolio volatility check
  - Concentration risk warning
  - Correlation-based diversification check
"""

from __future__ import annotations
import json
import pandas as pd
import numpy as np
import anthropic
from agent_utils import parse_llm_json

CLIENT = anthropic.Anthropic()

# Hard limits
MAX_POSITION    = 0.35   # no single stock above 35%
MAX_GROSS       = 0.95   # total invested never above 95%
MIN_CASH        = 0.05   # always keep 5% cash minimum
MAX_DD_THRESHOLD = 0.15  # circuit breaker at 15% drawdown
BEAR_MAX_GROSS  = 0.60   # in bear regime, max 60% invested

SYSTEM_PROMPT = """You are a risk officer reviewing a proposed portfolio.
Check for over-concentration and correlation risk.

Respond ONLY with valid JSON, no markdown, no code blocks.
Keep ALL text very short. Max 2 warnings, each under 10 words.

{"approved": true, "warnings": ["short warning"], "adjustments": {"NVDA": -0.05}, "reasoning": "under 15 words"}

adjustments: weight changes, empty {} if none needed."""


def _apply_hard_constraints(
    weights: dict[str, float],
    macro_regime: str = "mid_cycle",
    in_drawdown: bool = False,
) -> dict[str, float]:
    """
    Enforce hard position limits. These cannot be overridden.
    Returns adjusted weights.
    """
    if not weights:
        return {}

    # Circuit breaker: extreme drawdown -> minimum exposure
    if in_drawdown:
        print("  [RiskManager] CIRCUIT BREAKER: drawdown limit hit, "
              "reducing to 20% gross exposure")
        total = sum(weights.values())
        if total > 0:
            scale = 0.20 / total
            return {t: w * scale for t, w in weights.items()}

    # Bear regime: cap gross exposure
    max_gross = BEAR_MAX_GROSS if macro_regime == "recession" else MAX_GROSS

    # 1. Cap individual positions
    weights = {t: min(w, MAX_POSITION) for t, w in weights.items()}

    # 2. Cap total gross exposure
    total = sum(weights.values())
    if total > max_gross:
        scale = max_gross / total
        weights = {t: w * scale for t, w in weights.items()}

    # 3. Ensure minimum cash (scale down if needed)
    total = sum(weights.values())
    if total > (1.0 - MIN_CASH):
        scale = (1.0 - MIN_CASH) / total
        weights = {t: w * scale for t, w in weights.items()}

    return weights


def _compute_portfolio_vol(
    weights: dict[str, float],
    price_history: pd.DataFrame,
    window: int = 60,
) -> float:
    """Estimate annualized portfolio volatility from recent returns."""
    try:
        tickers = [t for t in weights if t in price_history.columns]
        if not tickers:
            return 0.0
        returns = price_history[tickers].pct_change().dropna().tail(window)
        w = np.array([weights.get(t, 0.0) for t in tickers])
        cov = returns.cov().values * 252
        port_var = w @ cov @ w
        return float(np.sqrt(max(port_var, 0)))
    except Exception:
        return 0.0


def _check_drawdown(equity_history: list[float]) -> tuple[bool, float]:
    """
    Check if current drawdown exceeds circuit breaker threshold.
    Returns (triggered, current_drawdown).
    """
    if len(equity_history) < 2:
        return False, 0.0
    peak = max(equity_history)
    current = equity_history[-1]
    dd = (current - peak) / peak
    return dd < -MAX_DD_THRESHOLD, dd


class RiskManager:
    """
    Wraps the Trader Agent output with hard and soft risk checks.
    """
    def __init__(self):
        self.equity_history: list[float] = []
        self.warnings_log: list[dict] = []

    def update_equity(self, value: float):
        self.equity_history.append(value)

    def review(
        self,
        proposed_weights: dict[str, float],
        price_history: pd.DataFrame,
        macro_regime: str = "mid_cycle",
        date=None,
    ) -> dict[str, float]:
        """
        Full risk review pipeline:
        1. Check circuit breaker
        2. Apply hard constraints
        3. LLM soft check
        4. Apply any LLM adjustments (within hard limits)
        Returns final approved weights.
        """
        # 1. Circuit breaker check
        in_drawdown, dd = _check_drawdown(self.equity_history)
        if in_drawdown:
            print(f"  [RiskManager] Drawdown={dd:.1%}, "
                  f"circuit breaker active")

        # 2. Hard constraints (always applied first)
        weights = _apply_hard_constraints(
            proposed_weights, macro_regime, in_drawdown
        )

        # 3. Portfolio volatility estimate
        port_vol = _compute_portfolio_vol(weights, price_history)

        # 4. LLM soft check
        try:
            w_summary = ", ".join(
                f"{t}={w:.2f}" for t, w in sorted(
                    weights.items(), key=lambda x: -x[1])
            )
            prompt = (
                f"Date: {date}\n"
                f"Proposed weights (after hard limits): {w_summary}\n"
                f"Estimated portfolio vol: {port_vol:.1%} annualized\n"
                f"Macro regime: {macro_regime}\n"
                f"Current drawdown: {dd:.1%}\n\n"
                f"Review for concentration and correlation risks."
            )
            response = CLIENT.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = parse_llm_json(response.content[0].text.strip())

            warnings = parsed.get("warnings", [])
            adjustments = parsed.get("adjustments", {})
            reasoning = parsed.get("reasoning", "")

            if warnings:
                print(f"  [RiskManager] Warnings: {warnings}")
            if reasoning:
                print(f"  [RiskManager] {reasoning}")

            # Apply LLM adjustments, then re-enforce hard limits
            if adjustments:
                for t, delta in adjustments.items():
                    if t in weights:
                        weights[t] = max(0.0, weights[t] + float(delta))
                weights = _apply_hard_constraints(
                    weights, macro_regime, in_drawdown
                )

            self.warnings_log.append({
                "date": str(date),
                "warnings": warnings,
                "port_vol": port_vol,
                "drawdown": dd,
                "final_weights": weights,
            })

        except Exception as e:
            print(f"  [RiskManager LLM error: {e}]")

        return weights
