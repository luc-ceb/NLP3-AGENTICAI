"""Demo del Caso 1 — Clasificador / router de encuestas (LangGraph).

Clasifica las quejas de mala experiencia en un único tema (taxonomía fija),
agrega el volumen por tema y por sucursal, y rutea el tema DOMINANTE al manual
operativo para producir una acción correctiva citada.

Uso:
    VECTOR_BACKEND=faiss python scripts/clasificar_encuestas_demo.py [LIMITE] [mala|buena]
    # LIMITE: tope de quejas a clasificar (las más recientes), para acotar costo.
    #         Por defecto 120; pasá 0 para clasificar TODAS.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.retrieve import HybridRetriever            # noqa: E402
from src.agents.auditor_rag import NormativeAuditor      # noqa: E402
from src.agents.caso1 import ClasificadorEncuestas        # noqa: E402
from src.llm.clients import make_llm                      # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DB = ROOT / "data" / "gdo.duckdb"
INDEX = ROOT / "data" / "index"

if __name__ == "__main__":
    limite = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    tabla = sys.argv[2] if len(sys.argv) > 2 else "mala"
    provider = os.getenv("LLM_PROVIDER", "groq")
    os.environ.setdefault("VECTOR_BACKEND", "faiss")

    con = duckdb.connect(str(DB), read_only=True)
    # max_tokens holgado: la clasificación por lotes y la síntesis producen JSON
    # extensos; con el default (1024) la respuesta se trunca y no parsea.
    llm = make_llm(provider=provider, max_tokens=2048)
    auditor = NormativeAuditor(HybridRetriever.build_default(INDEX), llm)
    clf = ClasificadorEncuestas(con, auditor, llm)

    res = clf.clasificar(tabla=tabla, limite=(limite or None))
    print(res)

    # Detalle por sucursal: tema dominante de cada PDV (cuando hubo clasificación).
    if res.por_sucursal:
        orden = {t: i for i, (t, _) in enumerate(res.ranking())}
        print("\nTEMA DOMINANTE POR SUCURSAL:")
        for num in sorted(res.por_sucursal):
            conteo = res.por_sucursal[num]
            tema = min(conteo, key=lambda t: (-conteo[t], orden.get(t, 99)))
            print(f"  {num[:8]:10} {tema:22} ({conteo[tema]}/{sum(conteo.values())})")
