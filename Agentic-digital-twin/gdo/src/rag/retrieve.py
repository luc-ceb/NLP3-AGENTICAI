"""Retriever híbrido: BM25 + denso -> RRF -> Cross-Encoder -> cap por doc.

Es el núcleo de recuperación del Auditor Normativo. Devuelve pasajes con su
cita direccionable ([source | tema | ts] para el manual, [source, p.X] para PDF).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .fusion import reciprocal_rank_fusion
from .retrievers import BM25Retriever, DenseRetriever, Embedder


def cite(c: dict) -> str:
    if c.get("tema"):
        ts = f" | {c['ts_start']}" if c.get("ts_start") else ""
        return f"{c['source']} | {c['tema']}{ts}"
    if c.get("page"):
        return f"{c['source']}, p.{c['page']}"
    return c["source"]


@dataclass
class RetrievedPassage:
    chunk: dict
    rerank_score: float

    @property
    def citation(self) -> str:
        return cite(self.chunk)


class HybridRetriever:
    def __init__(self, dense, sparse, reranker=None, rrf_k: int = 60):
        self.dense = dense
        self.sparse = sparse
        self.reranker = reranker
        self.rrf_k = rrf_k

    @classmethod
    def build_default(cls, index_dir: str | Path, embedder: Embedder | None = None,
                      rerank: bool = True):
        """Constructor de producción (carga modelos reales de HuggingFace)."""
        dense = DenseRetriever(index_dir, embedder=embedder)
        sparse = BM25Retriever(index_dir)
        reranker = None
        if rerank:
            from .reranker import CrossEncoderReranker
            reranker = CrossEncoderReranker()
        return cls(dense, sparse, reranker)

    def retrieve(self, query: str, k_dense: int = 10, k_sparse: int = 10,
                 k_rrf: int = 12, top_n: int = 5, per_doc_cap: int = 3
                 ) -> list[RetrievedPassage]:
        d = self.dense.search(query, k_dense)
        s = self.sparse.search(query, k_sparse)
        by_id = {h.chunk_id: h.chunk for h in d + s}

        fused = reciprocal_rank_fusion(
            [[h.chunk_id for h in d], [h.chunk_id for h in s]], self.rrf_k
        )
        candidates = [by_id[cid] for cid, _ in fused[:k_rrf]]

        if self.reranker:
            ranked = self.reranker.rerank(query, candidates)
        else:
            ranked = [(c, 0.0) for c in candidates]

        out: list[RetrievedPassage] = []
        per_doc: dict[str, int] = {}
        for c, score in ranked:
            doc = c["doc_id"]
            if per_doc.get(doc, 0) >= per_doc_cap:
                continue
            per_doc[doc] = per_doc.get(doc, 0) + 1
            out.append(RetrievedPassage(c, score))
            if len(out) >= top_n:
                break
        return out

    def format_context(self, passages: list[RetrievedPassage]) -> str:
        return "\n\n".join(f"[{p.citation}]\n{p.chunk['text']}" for p in passages)
