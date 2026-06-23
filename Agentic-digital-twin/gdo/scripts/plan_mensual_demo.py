"""Demo del Caso 5 — Plan mensual de diagnóstico (LangGraph que envuelve al Caso 2).

Itera el Caso 2 sobre las 6 sucursales para un mes y arma el reporte rankeado.

Uso:
    VECTOR_BACKEND=faiss python scripts/plan_mensual_demo.py [YYYY-MM]
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.retrieve import HybridRetriever        # noqa: E402
from src.agents.auditor_rag import NormativeAuditor  # noqa: E402
from src.agents.caso5 import PlanificadorMensual      # noqa: E402
from src.llm.clients import make_llm                  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

DB = ROOT / "data" / "gdo.duckdb"
INDEX = ROOT / "data" / "index"

if __name__ == "__main__":
    mes = sys.argv[1] if len(sys.argv) > 1 else None
    provider = os.getenv("LLM_PROVIDER", "groq")
    os.environ.setdefault("VECTOR_BACKEND", "faiss")

    con = duckdb.connect(str(DB), read_only=True)
    # max_tokens holgado: la síntesis mensual y los diagnósticos producen JSON
    # extensos; con el default (1024) la respuesta se trunca y no parsea.
    llm = make_llm(provider=provider, max_tokens=2048)
    auditor = NormativeAuditor(HybridRetriever.build_default(INDEX), llm)
    plan = PlanificadorMensual(con, auditor, llm)

    res = plan.generar(mes)
    print(res.reporte)
