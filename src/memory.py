"""
memory.py
---------
Layered memory system inspired by FinMem.

Each memory entry has:
  - content: the text of the memory (a decision, an outcome, a lesson)
  - timestamp: when it was created
  - layer: "short", "mid", or "long" (controls decay speed)
  - importance: 0-1 (high importance decays slower)
  - access_count: how often it has been retrieved (frequently used = sticky)

Retrieval ranks candidates by a blend of three scores:
  - novelty:   recency (exponential decay by layer)
  - relevance: semantic similarity to the query (keyword overlap proxy)
  - importance: assigned weight

We use a simple keyword-overlap relevance instead of embeddings to keep it
dependency-free and cheap. This can be upgraded to embeddings later.
"""

from __future__ import annotations
import json
import math
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field


# Decay half-life in days per layer. Long-term memories barely fade.
LAYER_HALFLIFE = {
    "short": 5.0,     # daily news, fades in a week
    "mid":   30.0,    # earnings reactions, multi-week trends
    "long":  180.0,   # structural facts, regime lessons
}


@dataclass
class MemoryEntry:
    content: str
    timestamp: str
    layer: str = "mid"
    importance: float = 0.5
    access_count: int = 0
    tickers: list = field(default_factory=list)

    def age_days(self, now: datetime) -> float:
        created = datetime.fromisoformat(self.timestamp)
        return max((now - created).total_seconds() / 86400.0, 0.0)

    def novelty(self, now: datetime) -> float:
        """Exponential recency decay based on layer half-life."""
        hl = LAYER_HALFLIFE.get(self.layer, 30.0)
        age = self.age_days(now)
        return math.exp(-math.log(2) * age / hl)


def _keyword_relevance(query: str, content: str, tickers: list) -> float:
    """
    Cheap relevance proxy: fraction of query keywords appearing in the memory,
    plus a bonus if tickers overlap.
    """
    q_words = set(w.lower().strip(".,!?") for w in query.split() if len(w) > 3)
    c_words = set(w.lower().strip(".,!?") for w in content.split() if len(w) > 3)
    if not q_words:
        return 0.0
    overlap = len(q_words & c_words) / len(q_words)

    # Ticker bonus
    q_tickers = set(t for t in query.split() if t.isupper() and len(t) <= 5)
    ticker_bonus = 0.3 if q_tickers & set(tickers) else 0.0

    return min(overlap + ticker_bonus, 1.0)


class MemoryStore:
    """
    Persistent memory store backed by a JSON file.
    Survives across backtest runs so the agent accumulates experience.
    """
    def __init__(self, path: str = "memory_store.json"):
        self.path = path
        self.entries: list[MemoryEntry] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    data = json.load(f)
                self.entries = [MemoryEntry(**e) for e in data]
            except Exception:
                self.entries = []

    def save(self):
        with open(self.path, "w") as f:
            json.dump([asdict(e) for e in self.entries], f, indent=2)

    def add(self, content: str, layer: str = "mid",
            importance: float = 0.5, tickers: list = None,
            now: datetime = None):
        entry = MemoryEntry(
            content=content,
            timestamp=(now or datetime.now()).isoformat(),
            layer=layer,
            importance=importance,
            tickers=tickers or [],
        )
        self.entries.append(entry)

    def retrieve(self, query: str, now: datetime,
                 top_k: int = 5,
                 w_novelty: float = 0.3,
                 w_relevance: float = 0.4,
                 w_importance: float = 0.3) -> list[MemoryEntry]:
        """
        Rank all memories by blended score and return the top_k.
        Increments access_count for retrieved memories (they get stickier).
        """
        if not self.entries:
            return []

        scored = []
        for e in self.entries:
            nov = e.novelty(now)
            rel = _keyword_relevance(query, e.content, e.tickers)
            imp = e.importance
            score = w_novelty * nov + w_relevance * rel + w_importance * imp
            scored.append((score, e))

        scored.sort(key=lambda x: -x[0])
        top = [e for _, e in scored[:top_k]]
        for e in top:
            e.access_count += 1
        return top

    def prune(self, now: datetime, min_novelty: float = 0.01):
        """
        Remove short/mid memories that have decayed below a threshold,
        unless they have been accessed often (proven useful).
        Long-term memories are never pruned.
        """
        kept = []
        for e in self.entries:
            if e.layer == "long":
                kept.append(e)
            elif e.novelty(now) > min_novelty or e.access_count >= 3:
                kept.append(e)
        self.entries = kept

    def __len__(self):
        return len(self.entries)
