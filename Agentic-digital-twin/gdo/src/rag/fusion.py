"""Reciprocal Rank Fusion (RRF): combina rankings por posición, sin normalizar."""
from __future__ import annotations


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """rankings: listas de chunk_ids ordenadas (mejor -> peor).

    Devuelve [(chunk_id, score)] ordenado desc. Score = sum 1/(k + rank + 1).
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
