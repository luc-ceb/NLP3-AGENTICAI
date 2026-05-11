from typing import Dict, List, Tuple, Optional
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: Optional[str] = None):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: List[Tuple[str, str, Dict]]) -> List[Tuple[str, str, Dict, float]]:
        pairs = [(query, c[1]) for c in candidates]
        scores = self.model.predict(pairs)
        out = [(c[0], c[1], c[2], float(s)) for c, s in zip(candidates, scores)]
        out.sort(key=lambda x: x[3], reverse=True)
        return out
