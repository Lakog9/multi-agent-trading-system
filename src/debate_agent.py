"""
debate_agent.py
---------------
Bull and Bear researchers debate the investment thesis.
Bounded to 2 rounds. Synthesis produces confidence scores per ticker.
"""

from __future__ import annotations
import json
import re
import anthropic
from agent_utils import parse_llm_json

CLIENT = anthropic.Anthropic()

BULL_SYSTEM = """You are an optimistic equity analyst (Bull Researcher).
Argue for increasing exposure based on the evidence. Be concise (3-5 sentences).
NO markdown headers, NO bullet points, plain text only."""

BEAR_SYSTEM = """You are a skeptical equity analyst (Bear Researcher).
Argue for reducing exposure based on the evidence. Be concise (3-5 sentences).
NO markdown headers, NO bullet points, plain text only."""

SYNTHESIS_SYSTEM = """You are a senior portfolio strategist synthesizing a bull/bear debate.
Respond ONLY with valid JSON, no markdown, no code blocks, no extra text.
All string values must be plain text with no newlines, no quotes inside strings.

{
  "ticker_confidence": {"AAPL": 0.6, "MSFT": 0.8},
  "overall_stance": "moderately bullish",
  "key_risks": "plain text one sentence",
  "key_opportunities": "plain text one sentence",
  "reasoning": "plain text two sentences"
}

ticker_confidence: 0.0 (strong avoid) to 1.0 (strong buy)."""


def _clean_for_json(text: str) -> str:
    """Remove characters that break JSON string parsing."""
    text = re.sub(r'[\x00-\x1f\x7f]', ' ', text)
    text = text.replace('"', "'")
    return text.strip()


def run_debate(tickers: list[str], evidence: str) -> dict:
    prompt_base = (
        f"Stocks: {', '.join(tickers)}\n\n"
        f"Evidence:\n{evidence}"
    )

    try:
        # Round 1: Bull opens
        bull_r1 = CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=250,
            system=BULL_SYSTEM,
            messages=[{"role": "user",
                       "content": prompt_base + "\n\nMake the bull case in plain text."}],
        ).content[0].text.strip()

        # Round 1: Bear responds
        bear_r1 = CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=250,
            system=BEAR_SYSTEM,
            messages=[{"role": "user",
                       "content": prompt_base
                       + f"\n\nBull argues: {bull_r1[:300]}"
                       + "\n\nMake the bear case in plain text."}],
        ).content[0].text.strip()

        # Round 2: Bull rebuts
        bull_r2 = CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            system=BULL_SYSTEM,
            messages=[{"role": "user",
                       "content": f"Bear argues: {bear_r1[:300]}"
                       + "\n\nBrief rebuttal in plain text."}],
        ).content[0].text.strip()

        # Synthesis
        synthesis_prompt = (
            f"Tickers: {', '.join(tickers)}\n"
            f"Bull: {_clean_for_json(bull_r1[:400])}\n"
            f"Bear: {_clean_for_json(bear_r1[:400])}\n"
            f"Bull rebuttal: {_clean_for_json(bull_r2[:200])}\n\n"
            f"Output ONLY the JSON object. No markdown. No explanation."
        )

        raw = CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": synthesis_prompt}],
        ).content[0].text.strip()

        result = parse_llm_json(raw)
        result["bull_case"] = bull_r1[:200]
        result["bear_case"] = bear_r1[:200]
        return result

    except Exception as e:
        print(f"  [Debate error: {e}]")
        return {
            "ticker_confidence": {t: 0.5 for t in tickers},
            "overall_stance": "neutral",
            "key_risks": "",
            "key_opportunities": "",
            "reasoning": "",
            "bull_case": "",
            "bear_case": "",
        }
