"""
agent_utils.py
--------------
Shared utilities for all LLM agents.
"""
import json
import re


def parse_llm_json(raw: str) -> dict:
    """
    Robustly parse JSON from LLM response.
    Handles markdown code blocks, extra whitespace, etc.
    """
    # Strip markdown code blocks if present
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    if not raw:
        raise ValueError("Empty response from LLM")

    return json.loads(raw)
