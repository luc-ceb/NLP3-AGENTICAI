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
    "- Usá los nombres EXACTOS de tablas y columnas tal como aparecen en el esquema de "
    "arriba. No inventes nombres (p.ej. si la tabla de ventas se llama 'datos_ventas', "
    "no escribas 'ventas').\n"
    "- Para atribuir una encuesta a su sucursal usá la columna 'numero' (código de "
    "sucursal): es la misma clave que datos_ventas.numero, así que un join directo por "
    "'numero' te da sucursal/region. El 'email' también aparece en ambas tablas pero es "
    "nulo en ~50% de las encuestas; preferí 'numero'.\n"
    "- La satisfacción es binaria por tabla: una tabla de encuestas de buena experiencia y "
    "otra de mala (no hay puntaje numérico).\n"
    "- El texto de la queja está en 'origen_respuesta_texto'. Buscá por raíz y SIN "
    "distinción de mayúsculas, ej.: lower(origen_respuesta_texto) LIKE '%derret%'."
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


class ConsultaRequest(BaseModel):
    pregunta: str = Field(..., min_length=5, max_length=500,
                          description="Consulta sobre el manual operativo.")


class ConsultaResponse(BaseModel):
    pregunta: str
    respuesta: str
    citas: list = []
    abstuvo: bool = False


def build_supervisor():
    """Construcción de producción (DuckDB + índices + Ollama)."""
    import duckdb
    from ..agents.analyst_sql import TextToSQLAnalyst
    from ..agents.auditor_rag import NormativeAuditor
    from ..rag.retrieve import HybridRetriever
    from ..agents.supervisor import SupervisorAgent
    from ..llm.clients import make_llm

    con = duckdb.connect(str(ROOT / "data" / "gdo.duckdb"), read_only=True)
    fast_llm = make_llm(os.getenv("FAST_MODEL"), provider=os.getenv("FAST_PROVIDER", "groq"))
    reason_llm = make_llm(os.getenv("REASON_MODEL"), provider=os.getenv("REASON_PROVIDER", "anthropic"))
    retriever = HybridRetriever.build_default(ROOT / "data" / "index")
    analyst = TextToSQLAnalyst(con, fast_llm, hints=HINTS)
    auditor = NormativeAuditor(retriever, fast_llm)
    return SupervisorAgent(analyst, auditor, llm=fast_llm, reconcile_llm=reason_llm,
                           retriever=retriever)


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

    @app.post("/consultar", response_model=ConsultaResponse)
    def consultar(req: ConsultaRequest, x_api_key: str | None = Header(default=None)):
        """Consulta normativa directa sobre el manual (RAG-QA, sin SQL)."""
        if state["key"] and x_api_key != state["key"]:
            raise HTTPException(status_code=401, detail="API key inválida o faltante")
        from ..rag.qa import consultar_norma
        sup = get_supervisor()
        r = consultar_norma(sup.retriever, sup.llm, req.pregunta)
        return ConsultaResponse(pregunta=req.pregunta, respuesta=r["respuesta"],
                                citas=r["citas"], abstuvo=r["abstuvo"])

    return app


app = create_app()