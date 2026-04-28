"""Reciprocal Rank Fusion (RRF) over multiple ranked lists."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def rrf_fuse(
    ranked_lists: Iterable[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Each list contains qa_ids in descending relevance.

    score(d) = Σ_i  w_i / (k + rank_i(d))
    """
    lists = [list(rl) for rl in ranked_lists]
    if weights is None:
        weights = [1.0] * len(lists)
    if len(weights) != len(lists):
        raise ValueError("weights length must match number of lists")

    score: dict[str, float] = defaultdict(float)
    for w, rl in zip(weights, lists):
        for rank, doc_id in enumerate(rl):
            score[doc_id] += w / (k + rank + 1)
    return sorted(score.items(), key=lambda x: -x[1])
