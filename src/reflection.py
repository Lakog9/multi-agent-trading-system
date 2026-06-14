"""
reflection.py
-------------
The learning mechanism. After outcomes are known, the agent reflects on
what happened and writes a lesson into long-term memory.

Two functions:
  1. record_decision: stores a decision + context as it happens
  2. reflect_on_outcome: once we know the return that followed, the agent
     compares expectation vs reality and writes a lesson.

The lesson is what surfaces in future retrievals for similar setups.
This is how the system gains experience without fine-tuning.
"""

from __future__ import annotations
from datetime import datetime
import anthropic
from agent_utils import parse_llm_json

CLIENT = anthropic.Anthropic()

REFLECTION_SYSTEM = """You are a trading analyst reviewing a past decision and its outcome.
Compare what was expected with what actually happened, and extract ONE concise lesson.

Respond ONLY with valid JSON, no markdown, no code blocks:
{"lesson": "under 25 words, actionable", "importance": 0.7, "lesson_type": "timing"}

importance: 0.0-1.0. Higher for surprising or costly outcomes that should
strongly influence future decisions.
lesson_type: one of "timing", "sizing", "selection", "risk", "macro"."""


def reflect_on_outcome(
    decision: dict,
    realized_return: float,
    benchmark_return: float,
    now: datetime,
) -> dict:
    """
    Given a past decision and the return that followed, generate a lesson.

    decision: dict with keys like date, weights, reasoning, stance, etc.
    realized_return: the portfolio return over the period after the decision
    benchmark_return: buy-and-hold return over the same period (for comparison)
    """
    outperformance = realized_return - benchmark_return

    prompt = (
        f"DECISION on {decision.get('date', 'unknown')}:\n"
        f"Stance: {decision.get('stance', 'N/A')}\n"
        f"Weights: {decision.get('final', decision.get('weights', {}))}\n"
        f"Reasoning: {decision.get('reasoning', '')[:300]}\n"
        f"Sentiment: {decision.get('sentiment', 'N/A')}, "
        f"Macro: {decision.get('macro', 'N/A')}\n\n"
        f"OUTCOME over the following period:\n"
        f"Portfolio return: {realized_return:+.2%}\n"
        f"Buy-and-hold return: {benchmark_return:+.2%}\n"
        f"Relative performance: {outperformance:+.2%}\n\n"
        f"What is the key lesson? Was the reasoning validated or contradicted?"
    )

    try:
        response = CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = parse_llm_json(response.content[0].text.strip())
        return {
            "lesson": parsed.get("lesson", ""),
            "importance": float(parsed.get("importance", 0.5)),
            "lesson_type": parsed.get("lesson_type", "general"),
            "outperformance": outperformance,
        }
    except Exception as e:
        print(f"  [Reflection error: {e}]")
        return {
            "lesson": "",
            "importance": 0.5,
            "lesson_type": "general",
            "outperformance": outperformance,
        }


def build_memory_context(memories: list, max_chars: int = 800) -> str:
    """
    Format retrieved memories into a context block for the next decision.
    """
    if not memories:
        return "No relevant past experiences."

    lines = ["Relevant lessons from past decisions:"]
    total = 0
    for m in memories:
        line = f"- {m.content}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n".join(lines)
