"""Re-ranqueo post-retrieval con Cross-Encoder (MS MARCO MiniLM)."""
from __future__ import annotations

import numpy as np


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # carga perezosa
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_n: int | None = None):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        order = np.argsort(scores)[::-1]
        top_n = top_n or len(candidates)
        return [(candidates[i], float(scores[i])) for i in order[:top_n]]
