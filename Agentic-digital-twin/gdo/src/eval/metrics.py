"""Métricas de evaluación: IR (retrieval) y predicado de relevancia.

Relevancia por etiqueta (sin anotación chunk-a-chunk): un chunk es relevante si
cumple TODOS los criterios especificados en la etiqueta (tipo_doc / keywords / source).
"""
from __future__ import annotations

import math


def is_relevant(chunk: dict, label: dict) -> bool:
    if label.get("tipo_doc"):
        if chunk.get("tipo_doc") not in label["tipo_doc"]:
            return False
    if label.get("keywords"):
        txt = (chunk.get("text") or "").lower()
        if not any(kw.lower() in txt for kw in label["keywords"]):
            return False
    if label.get("source"):
        if label["source"].lower() not in (chunk.get("source") or "").lower():
            return False
    return bool(label)  # etiqueta vacía -> no evaluable


def hit_at_k(rels: list[int], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def precision_at_k(rels: list[int], k: int) -> float:
    k = min(k, len(rels))
    return sum(rels[:k]) / k if k else 0.0


def mrr(rels: list[int]) -> float:
    for i, r in enumerate(rels, 1):
        if r:
            return 1.0 / i
    return 0.0


def _dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(rels: list[int], k: int) -> float:
    idcg = _dcg(sorted(rels, reverse=True), k)
    return _dcg(rels, k) / idcg if idcg else 0.0
