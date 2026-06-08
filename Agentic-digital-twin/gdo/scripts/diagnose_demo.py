"""Demo end-to-end del Supervisor (producción: DuckDB + índices + Ollama).

Ruteo por agente (optimización de latencia/costo, §7):
  - Analista (Text-to-SQL): modelo de código  -> SQL_MODEL    (default qwen2.5-coder:14b)
  - Auditor / Supervisor (razonamiento)        -> REASON_MODEL (default qwen3:14b)

Uso:
    USE_OLLAMA=1 python scripts/diagnose_demo.py
    # opcional: SQL_MODEL=qwen2.5-coder:14b REASON_MODEL=qwen3:14b
    # (si no tenés el coder: ollama pull qwen2.5-coder:14b)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_layer.duckdb_loader import load_sources   # noqa: E402
from src.agents.analyst_sql import TextToSQLAnalyst       # noqa: E402
from src.rag.retrieve import HybridRetriever              # noqa: E402
from src.agents.auditor_rag import NormativeAuditor       # noqa: E402
from src.agents.supervisor import SupervisorAgent         # noqa: E402
from src.llm.clients import make_llm                   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

DB = ROOT / "data" / "gdo.duckdb"
INDEX = ROOT / "data" / "index"

# Notas de dominio para el Analista (corrige el typo 'emai' y el join a sucursal)
HINTS = (
    "Notas de dominio:\n"
    "- En las tablas de encuestas la columna de email es 'emai' (sin la l final).\n"
    "- La satisfacción es binaria por tabla: encuestas_buena_experiencia vs "
    "encuestas_mala_experiencia (no hay puntaje numérico).\n"
    "- Para atribuir una encuesta a una sucursal, uní por email: "
    "ventas.email = encuestas_mala_experiencia.emai (y luego ventas.branchofficeid).\n"
    "- El texto de la queja está en 'origen_respuesta_texto'. Para buscar temas usá "
    "coincidencia SIN distinción de mayúsculas y por raíz, ej.: "
    "lower(origen_respuesta_texto) LIKE '%derret%' (capta derretido/derretida/derritió)."
)

# Pregunta estrella: dato (quejas de cadena de frío) -> norma (manual de cámara de frío)
PREGUNTA = "¿Cuántas quejas de mala experiencia mencionan que el helado estaba derretido o blando?"

if __name__ == "__main__":
    fast_provider = os.getenv("FAST_PROVIDER", "groq")
    reason_provider = os.getenv("REASON_PROVIDER", "anthropic")
    fast_model = os.getenv("FAST_MODEL")
    reason_model = os.getenv("REASON_MODEL")
    print(f"FAST={fast_provider}:{fast_model or '(default)'} | "
          f"REASON={reason_provider}:{reason_model or '(default)'} | "
          f"backend={os.getenv('VECTOR_BACKEND', 'pinecone')}\n")

    if not DB.exists():
        load_sources(ROOT / "data" / "raw", DB)
    con = duckdb.connect(str(DB), read_only=True)

    fast_llm = make_llm(fast_model, provider=fast_provider)      # SQL, grader, rewrite, claims
    reason_llm = make_llm(reason_model, provider=reason_provider)  # reconcile (diagnóstico final)

    analyst = TextToSQLAnalyst(con, fast_llm, hints=HINTS)
    auditor = NormativeAuditor(HybridRetriever.build_default(INDEX), fast_llm)
    supervisor = SupervisorAgent(analyst, auditor, llm=fast_llm, reconcile_llm=reason_llm)

    res = supervisor.diagnose(PREGUNTA)
    print("=" * 80)
    print(f"SQL del Analista: {res.sql}")
    print(f"Hechos:\n{res.facts}\n")
    print(res)
