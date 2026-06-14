"""
trader_agent.py
---------------
Trader Agent: converts the debate thesis + risk appetite into
final target weights.

Inputs:
- Debate thesis (ticker confidence scores + stance)
- Technical scores (from Technical Agent)
- Fundamental scores (from Fundamental Agent)
- Sentiment (score + regime)
- Macro (risk appetite + favored sectors)

Output:
- Final target weights per ticker
- Cash allocation
- Conviction score
"""

from __future__ import annotations
import json
import anthropic
from agent_utils import parse_llm_json

CLIENT = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a systematic trader executing portfolio decisions.
You receive a synthesized investment thesis from a bull/bear debate,
plus supporting signals from technical, fundamental, sentiment, and macro analysis.

Your job is to translate the thesis into precise portfolio weights.

Rules:
- Weights must sum to 1.0 or less (remainder is cash)
- No single position above 0.35
- Scale overall exposure by macro risk_appetite AND sentiment score
  (e.g. if both are 0.4, keep 60%+ in cash)
- Favor tickers where debate confidence AND technical/fundamental agree
- Penalize tickers where signals conflict

Respond ONLY with valid JSON, no markdown:
{
  "weights": {"AAPL": 0.20, "MSFT": 0.25, ...},
  "cash": 0.20,
  "gross_exposure": 0.80,
  "conviction": 0.7,
  "reasoning": "two sentence explanation of key allocation decisions"
}"""


def compute_weights(
    tickers: list[str],
    debate_result: dict,
    tech_scores: dict,
    fund_scores: dict,
    sentiment: dict,
    macro: dict,
) -> dict:
    """
    Calls Claude to translate all signals into final weights.
    """
    confidence = debate_result.get("ticker_confidence", {})
    stance = debate_result.get("overall_stance", "neutral")
    risks = debate_result.get("key_risks", "")
    opps = debate_result.get("key_opportunities", "")
    debate_reasoning = debate_result.get("reasoning", "")

    ticker_summary = []
    for t in tickers:
        ticker_summary.append(
            f"{t}: debate={confidence.get(t, 0.5):.2f}, "
            f"tech={tech_scores.get(t, 0.0):.1f}/5, "
            f"fundamental={fund_scores.get(t, 0.5):.2f}"
        )

    prompt = (
        f"=== DEBATE THESIS ===\n"
        f"Overall stance: {stance}\n"
        f"Key opportunities: {opps}\n"
        f"Key risks: {risks}\n"
        f"Synthesis: {debate_reasoning}\n\n"
        f"=== TICKER SIGNALS ===\n"
        + "\n".join(ticker_summary)
        + f"\n\n=== MARKET CONTEXT ===\n"
        f"Sentiment: score={sentiment.get('score', 0.5):.2f}, "
        f"regime={sentiment.get('regime', 'neutral')}\n"
        f"Macro: regime={macro.get('regime', 'mid_cycle')}, "
        f"risk_appetite={macro.get('risk_appetite', 0.5):.2f}, "
        f"favored_sectors={macro.get('favored_sectors', [])}\n\n"
        f"Translate into final portfolio weights."
    )

    try:
        response = CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=350,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        parsed = parse_llm_json(raw)
        weights = {k: float(v) for k, v in parsed.get("weights", {}).items()
                   if k in tickers}
        return {
            "weights": weights,
            "cash": parsed.get("cash", 0.0),
            "gross_exposure": parsed.get("gross_exposure", sum(weights.values())),
            "conviction": parsed.get("conviction", 0.5),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        print(f"  [TraderAgent error: {e}]")
        return {
            "weights": {},
            "cash": 1.0,
            "gross_exposure": 0.0,
            "conviction": 0.0,
            "reasoning": "",
        }
