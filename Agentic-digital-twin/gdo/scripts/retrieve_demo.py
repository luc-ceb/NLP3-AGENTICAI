"""Smoke test del retriever híbrido sobre la base de conocimiento.

Uso (en tu máquina, con índices ya construidos):
    python scripts/retrieve_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.retrieve import HybridRetriever  # noqa: E402

PREGUNTAS = [
    "¿Qué pasos debo seguir al cerrar la franquicia cada noche?",
    "¿A qué temperatura debe estar la chocolatera?",
    "¿Qué hago si la fila en hora pico es muy larga?",
]

if __name__ == "__main__":
    r = HybridRetriever.build_default(ROOT / "data" / "index")
    for q in PREGUNTAS:
        print("\n" + "=" * 80 + f"\nPREGUNTA: {q}")
        for i, p in enumerate(r.retrieve(q, top_n=4), 1):
            print(f"\n[{i}] (rerank={p.rerank_score:.2f})  {p.citation}")
            print(f"     {p.chunk['text'][:200]}...")
