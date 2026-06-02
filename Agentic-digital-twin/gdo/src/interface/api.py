"""API FastAPI del Gemelo Digital Operativo.

Orquesta la consulta del usuario a través del Supervisor y dispara la acción
automática (registro en log). Seguridad en el borde: validación de input
(Pydantic) y API key opcional (header x-api-key, vía env GDO_API_KEY).

Levantar:  uvicorn src.interface.api:app --reload
Docs:      http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ..actions.notifier import LogNotifier

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

HINTS = (
    "Notas de dominio:\n"
    "- En las tablas de encuestas la columna de email es 'emai' (sin la l final).\n"
    "- La satisfacción es binaria por tabla: encuestas_buena_experiencia vs "
    "encuestas_mala_experiencia (no hay puntaje numérico).\n"
    "- Para atribuir una encuesta a una sucursal, uní por email: "
    "ventas.email = encuestas_mala_experiencia.emai (y luego ventas.branchofficeid).\n"
    "- El texto de la queja está en 'origen_respuesta_texto'. Para buscar temas usá "
    "coincidencia SIN distinción de mayúsculas y por raíz, ej.: "
    "lower(origen_respuesta_texto) LIKE '%derret%'."
)


class DiagnoseRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500,
                          description="Pregunta operativa en lenguaje natural.")


class DiagnoseResponse(BaseModel):
    question: str
    diagnosis: str
    sql: str = ""
    audits: list = []
    citations: list = []


def build_supervisor():
    """Construcción de producción (DuckDB + índices + Ollama)."""
    import duckdb
    from ..agents.analyst_sql import TextToSQLAnalyst
    from ..agents.auditor_rag import NormativeAuditor
    from ..rag.retrieve import HybridRetriever
    from ..agents.supervisor import SupervisorAgent
    from ..llm.clients import OllamaClient

    con = duckdb.connect(str(ROOT / "data" / "gdo.duckdb"), read_only=True)
    sql_llm = OllamaClient(model=os.getenv("SQL_MODEL", "qwen2.5-coder:14b"))
    reason_llm = OllamaClient(model=os.getenv("REASON_MODEL") or os.getenv("OLLAMA_MODEL", "qwen3:14b"))
    analyst = TextToSQLAnalyst(con, sql_llm, hints=HINTS)
    auditor = NormativeAuditor(HybridRetriever.build_default(ROOT / "data" / "index"), reason_llm)
    return SupervisorAgent(analyst, auditor, reason_llm)


def create_app(supervisor=None, notifier=None, api_key: str | None = None) -> FastAPI:
    app = FastAPI(title="Gemelo Digital Operativo (GDO)", version="0.1.0")
    state = {
        "sup": supervisor,
        "notif": notifier,
        "key": api_key if api_key is not None else os.getenv("GDO_API_KEY"),
    }

    def get_supervisor():
        if state["sup"] is None:
            state["sup"] = build_supervisor()  # build perezoso (carga modelos al 1er request)
        return state["sup"]

    def get_notifier():
        if state["notif"] is None:
            state["notif"] = LogNotifier()
        return state["notif"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/diagnose", response_model=DiagnoseResponse)
    def diagnose(req: DiagnoseRequest, x_api_key: str | None = Header(default=None)):
        if state["key"] and x_api_key != state["key"]:
            raise HTTPException(status_code=401, detail="API key inválida o faltante")
        result = get_supervisor().diagnose(req.question)
        get_notifier().send(result)  # <-- acción automática: registro en log externo
        return DiagnoseResponse(
            question=result.question, diagnosis=result.diagnosis,
            sql=getattr(result, "sql", ""), audits=getattr(result, "audits", []),
            citations=getattr(result, "citations", []),
        )

    return app


app = create_app()
