"""Ranking metrics for top-k recommendation evaluation.

All functions take a recommended list and a set of relevant items and return a
per-user score in [0, 1]. Average scores across users with average_metrics.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Sequence

import numpy as np


def _top_k(recommended: Iterable[int], k: int) -> list[int]:
    return list(recommended)[: max(k, 0)]


def precision_at_k(recommended: Iterable[int], relevant: Iterable[int], k: int = 10) -> float:
    """Fraction of the top-k recommendations that are relevant."""
    rec = _top_k(recommended, k)
    if k <= 0 or not rec:
        return 0.0
    relevant = set(relevant)
    return sum(1 for m in rec if m in relevant) / k


def recall_at_k(recommended: Iterable[int], relevant: Iterable[int], k: int = 10) -> float:
    """Fraction of relevant items recovered in the top-k recommendations."""
    rec = _top_k(recommended, k)
    relevant = set(relevant)
    if not relevant:
        return 0.0
    hits = sum(1 for m in rec if m in relevant)
    return hits / len(relevant)


def map_at_k(recommended: Iterable[int], relevant: Iterable[int], k: int = 10) -> float:
    """Mean average precision over the top-k recommendations."""
    rec = _top_k(recommended, k)
    relevant = set(relevant)
    if not rec or not relevant:
        return 0.0
    hits, ap = 0, 0.0
    for i, m in enumerate(rec, start=1):
        if m in relevant:
            hits += 1
            ap += hits / i
    return ap / min(len(relevant), k)


def ndcg_at_k(
    recommended: Iterable[int],
    relevant: Iterable[int],
    k: int = 10,
    grades: Mapping[int, float] | None = None,
) -> float:
    """Normalized discounted cumulative gain over the top-k recommendations.

    Relevance is binary by default. When grades maps movie_id to a rating, the
    rating is used as the gain for relevant items.
    """
    rec = _top_k(recommended, k)
    relevant = set(relevant)
    if not rec or not relevant:
        return 0.0

    def gain(movie_id: int) -> float:
        if movie_id not in relevant:
            return 0.0
        return float(grades[movie_id]) if grades is not None else 1.0

    dcg = sum(gain(m) / np.log2(i + 1) for i, m in enumerate(rec, start=1))
    ideal = sorted((gain(m) for m in relevant), reverse=True)
    idcg = sum(g / np.log2(i + 1) for i, g in enumerate(ideal, start=1))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


def hit_rate_at_k(recommended: Iterable[int], relevant: Iterable[int], k: int = 10) -> float:
    """1.0 if any relevant item appears in the top-k recommendations, else 0.0."""
    rec = _top_k(recommended, k)
    relevant = set(relevant)
    return 1.0 if any(m in relevant for m in rec) else 0.0


def average_metrics(scores: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Macro-average per-user metric dicts."""
    if not scores:
        return {}
    keys = list(scores[0].keys())
    return {key: float(np.mean([s[key] for s in scores])) for key in keys}
