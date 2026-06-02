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
from src.llm.clients import OllamaClient                  # noqa: E402

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
    sql_model = os.getenv("SQL_MODEL", "qwen2.5-coder:14b")
    reason_model = os.getenv("REASON_MODEL") or os.getenv("OLLAMA_MODEL", "qwen3:14b")
    print(f"SQL_MODEL={sql_model}  |  REASON_MODEL={reason_model}\n")

    if not DB.exists():
        load_sources(ROOT / "data" / "raw", DB)
    con = duckdb.connect(str(DB), read_only=True)

    sql_llm = OllamaClient(model=sql_model)       # rápido, especializado en SQL
    reason_llm = OllamaClient(model=reason_model)  # razonamiento para reconciliar

    analyst = TextToSQLAnalyst(con, sql_llm, hints=HINTS)
    auditor = NormativeAuditor(HybridRetriever.build_default(INDEX), reason_llm)
    supervisor = SupervisorAgent(analyst, auditor, reason_llm)

    res = supervisor.diagnose(PREGUNTA)
    print("=" * 80)
    print(f"SQL del Analista: {res.sql}")
    print(f"Hechos:\n{res.facts}\n")
    print(res)
