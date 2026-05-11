from typing import List, Dict

def rrf_combine(*ranked_lists: List[str], k: float = 60.0) -> List[str]:
    """
    Recibe múltiples listas ordenadas (BM25, vectorial, dense) y devuelve una lista fusionada usando RRF.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1.0)
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in sorted_items]
