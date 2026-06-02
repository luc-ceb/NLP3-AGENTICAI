"""Retrievers denso (FAISS) y disperso (BM25) + tokenizador y embedder.

Diseñados para inyección de dependencias: el embedder real envuelve
SentenceTransformer, pero se puede inyectar uno alternativo (tests).
"""
from __future__ import annotations

import json
import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_STOP = {
    "de", "la", "el", "en", "y", "a", "los", "las", "del", "un", "una", "que",
    "se", "con", "por", "para", "su", "al", "lo", "es", "como", "mas", "o",
    "the", "of", "to", "and", "in",
}


def tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    return [w for w in re.findall(r"[a-z0-9]+", t) if len(w) > 1 and w not in _STOP]


def load_meta(index_dir: str | Path) -> list[dict]:
    p = Path(index_dir) / "meta.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]


@dataclass
class Hit:
    chunk_id: str
    chunk: dict
    score: float
    retriever: str = ""


class Embedder:
    """Embedder denso por defecto (SentenceTransformer all-MiniLM-L6-v2)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # carga perezosa
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        v = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
        return v.astype("float32")


class BM25Retriever:
    def __init__(self, index_dir: str | Path):
        self.index_dir = Path(index_dir)
        self.meta = load_meta(index_dir)
        with (self.index_dir / "bm25.pkl").open("rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [Hit(self.meta[i]["chunk_id"], self.meta[i], float(scores[i]), "bm25")
                for i in order if scores[i] > 0]


class DenseRetriever:
    def __init__(self, index_dir: str | Path, embedder: Embedder | None = None):
        import faiss  # carga perezosa
        self.index_dir = Path(index_dir)
        self.meta = load_meta(index_dir)
        self.index = faiss.read_index(str(self.index_dir / "dense.faiss"))
        if embedder is None:
            cfg = json.loads((self.index_dir / "dense.json").read_text())
            embedder = Embedder(cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"))
        self.embedder = embedder

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q = self.embedder.encode([query])
        scores, idx = self.index.search(q, k)
        out = []
        for rank, i in enumerate(idx[0]):
            if i == -1:
                continue
            out.append(Hit(self.meta[i]["chunk_id"], self.meta[i], float(scores[0][rank]), "dense"))
        return out
