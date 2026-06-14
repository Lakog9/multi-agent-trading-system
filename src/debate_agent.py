"""
debate_agent.py
---------------
Bull and Bear researchers debate the investment thesis.
This is the only place in the system where free-form dialogue is allowed,
and it is bounded to exactly 2 rounds to control cost and rambling.

After the debate, a synthesis step produces a single investment thesis
with an explicit confidence score per ticker.
"""

from __future__ import annotations
import json
import anthropic
from agent_utils import parse_llm_json

CLIENT = anthropic.Anthropic()

BULL_SYSTEM = """You are an optimistic equity analyst (Bull Researcher).
You argue for increasing exposure to stocks based on the evidence provided.
Focus on: upside catalysts, positive momentum, growth potential, macro tailwinds.
Be specific and reference the actual data. Be concise (3-5 sentences)."""

BEAR_SYSTEM = """You are a skeptical equity analyst (Bear Researcher).
You argue for reducing or avoiding exposure based on the evidence provided.
Focus on: downside risks, overbought signals, macro headwinds, valuation concerns.
Be specific and reference the actual data. Be concise (3-5 sentences)."""

SYNTHESIS_SYSTEM = """You are a senior portfolio strategist synthesizing a bull/bear debate.
Read both arguments and produce a balanced investment thesis.

Respond ONLY with valid JSON, no markdown:
{
  "ticker_confidence": {"AAPL": 0.6, "MSFT": 0.8, ...},
  "overall_stance": "moderately bullish",
  "key_risks": "one sentence",
  "key_opportunities": "one sentence",
  "reasoning": "two sentence synthesis"
}
ticker_confidence: 0.0 (strong avoid) to 1.0 (strong buy). 0.5 = neutral."""


def run_debate(
    tickers: list[str],
    evidence: str,
) -> dict:
    """
    Run a 2-round bull/bear debate and synthesize into a thesis.
    Returns confidence scores per ticker plus reasoning.
    """
    prompt_base = (
        f"Stocks under consideration: {', '.join(tickers)}\n\n"
        f"Evidence from analyst reports:\n{evidence}"
    )

    # Round 1: Bull opens
    bull_r1 = CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=BULL_SYSTEM,
        messages=[{"role": "user",
                   "content": prompt_base + "\n\nMake the bull case."}],
    ).content[0].text

    # Round 1: Bear responds
    bear_r1 = CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=BEAR_SYSTEM,
        messages=[{"role": "user",
                   "content": prompt_base
                   + f"\n\nThe bull analyst argues:\n{bull_r1}"
                   + "\n\nMake the bear case and rebut the bull."}],
    ).content[0].text

    # Round 2: Bull rebuts
    bull_r2 = CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=BULL_SYSTEM,
        messages=[{"role": "user",
                   "content": prompt_base
                   + f"\n\nYour bull case:\n{bull_r1}"
                   + f"\n\nBear rebuttal:\n{bear_r1}"
                   + "\n\nRespond briefly to the bear's key points."}],
    ).content[0].text

    # Synthesis
    synthesis_prompt = (
        f"Evidence:\n{evidence}\n\n"
        f"Bull case (round 1):\n{bull_r1}\n\n"
        f"Bear case:\n{bear_r1}\n\n"
        f"Bull rebuttal:\n{bull_r2}\n\n"
        f"Synthesize into a final investment thesis."
    )

    synth_raw = CLIENT.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=350,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": synthesis_prompt}],
    ).content[0].text

    try:
        result = parse_llm_json(synth_raw)
        result["bull_case"] = bull_r1
        result["bear_case"] = bear_r1
        return result
    except Exception as e:
        print(f"  [Debate synthesis error: {e}]")
        return {
            "ticker_confidence": {t: 0.5 for t in tickers},
            "overall_stance": "neutral",
            "key_risks": "",
            "key_opportunities": "",
            "reasoning": "",
        }
