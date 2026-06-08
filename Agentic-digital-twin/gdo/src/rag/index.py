"""Construcción y persistencia de los índices denso (FAISS) y disperso (BM25).

Layout de salida (data/index/):
  meta.jsonl     chunks en orden (compartido por ambos índices)
  bm25.pkl       objeto BM25Okapi serializado
  dense.faiss    índice FAISS (IndexFlatIP, coseno sobre vectores normalizados)
  dense.json     {model_name, dim}
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from .retrievers import Embedder, tokenize

log = logging.getLogger(__name__)


def _load_chunks(chunks_path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(chunks_path).read_text(encoding="utf-8").splitlines()]


def build_bm25_index(chunks: list[dict], out_dir: Path) -> None:
    from rank_bm25 import BM25Okapi
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    log.info("BM25 construido sobre %d chunks", len(chunks))


def build_dense_index(chunks: list[dict], out_dir: Path,
                      embedder: Embedder | None = None,
                      model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    import faiss
    embedder = embedder or Embedder(model_name)
    embs = embedder.encode([c["text"] for c in chunks])
    dim = int(embs.shape[1])
    index = faiss.IndexFlatIP(dim)   # coseno: vectores normalizados + producto interno
    index.add(embs)
    faiss.write_index(index, str(out_dir / "dense.faiss"))
    name = getattr(embedder, "model_name", model_name)
    (out_dir / "dense.json").write_text(json.dumps({"model_name": name, "dim": dim}))
    log.info("Índice denso construido: %d vectores, dim %d", index.ntotal, dim)


def build_all(chunks_path: str | Path, out_dir: str | Path,
              embedder: Embedder | None = None, vector_backend: str | None = None) -> None:
    import os
    backend = (vector_backend or os.getenv("VECTOR_BACKEND", "pinecone")).lower()
    chunks = _load_chunks(chunks_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # meta + BM25 (disperso) siempre local
    with (out / "meta.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    build_bm25_index(chunks, out)
    if backend == "pinecone":
        build_pinecone_index(chunks, embedder=embedder)
    else:
        build_dense_index(chunks, out, embedder=embedder)
    log.info("Índices listos (denso=%s) en %s", backend, out)


def build_pinecone_index(chunks: list[dict], embedder: Embedder | None = None,
                         index_name: str | None = None, namespace: str | None = None,
                         batch_size: int = 100, api_key: str | None = None,
                         cloud: str | None = None, region: str | None = None) -> None:
    """Embebe los chunks (MiniLM local) y los upsertea a un índice Pinecone."""
    import os
    from pinecone import Pinecone, ServerlessSpec

    embedder = embedder or Embedder()
    name = index_name or os.getenv("PINECONE_INDEX", "gdo")
    ns = namespace if namespace is not None else os.getenv("PINECONE_NAMESPACE", "")
    pc = Pinecone(api_key=api_key or os.getenv("PINECONE_API_KEY"))

    embs = embedder.encode([c["text"] for c in chunks])
    dim = int(embs.shape[1])
    if not pc.has_index(name):
        pc.create_index(name=name, dimension=dim, metric="cosine",
                        spec=ServerlessSpec(cloud=cloud or os.getenv("PINECONE_CLOUD", "aws"),
                                            region=region or os.getenv("PINECONE_REGION", "us-east-1")))
    index = pc.Index(name)

    vectors = []
    for c, e in zip(chunks, embs):
        md = {k: v for k, v in c.items() if v is not None}  # Pinecone no acepta null
        vectors.append({"id": c["chunk_id"], "values": e.tolist(), "metadata": md})
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i:i + batch_size], namespace=ns or None)
    log.info("Pinecone: %d vectores upserted en '%s' (dim=%d, ns=%s)",
             len(vectors), name, dim, ns or "default")
