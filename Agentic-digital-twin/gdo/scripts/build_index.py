"""Construye los índices denso (FAISS) + disperso (BM25) desde chunks.jsonl.

Uso (en tu máquina, descarga el modelo de embeddings la 1ra vez):
    python scripts/build_index.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.index import build_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

if __name__ == "__main__":
    build_all(ROOT / "data" / "processed" / "chunks.jsonl", ROOT / "data" / "index")
    print("\nÍndices construidos en data/index/ (meta.jsonl, bm25.pkl, dense.faiss)")
