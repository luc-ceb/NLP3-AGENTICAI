"""RAG-QA sobre el manual operativo (ruta "norma" del Caso 2).

Consulta directa al manual, sin pasar por SQL: recupera pasajes con el
HybridRetriever, redacta una respuesta apoyada SOLO en ese contexto y se
ABSTIENE si el manual no cubre la pregunta. Las citas se atan a los pasajes
REALES recuperados (no a lo que transcribe el LLM), igual que en el Auditor.
"""
from __future__ import annotations

from ..agents.auditor_rag import _parse_json

NORMA_QA_PROMPT = """Sos un asistente experto en operaciones de heladerías.
Respondé la PREGUNTA usando SOLO el CONTEXTO (pasajes del manual operativo,
numerados [1], [2], ...). No inventes: si el contexto no contiene la respuesta,
abstenete (encontrada=false) y explicá brevemente que el manual no la cubre.

Respondé SOLO JSON:
{{"encontrada": true|false,
  "respuesta": "<respuesta apoyada en el contexto, o motivo de la abstención>",
  "pasajes": [<números de los pasajes en los que te apoyás>]}}

CONTEXTO:
{context}

PREGUNTA: {question}
"""


def consultar_norma(retriever, llm, question: str, top_n: int = 5) -> dict:
    """Consulta directa al manual operativo vía RAG (sin SQL).

    Devuelve ``{"respuesta", "citas", "abstuvo"}``. Se abstiene si no se recupera
    contexto o si el LLM determina que el manual no cubre la pregunta. Las citas
    corresponden a los pasajes REALMENTE usados por el LLM (con fallback a todos
    los recuperados si no señala ninguno).
    """
    passages = retriever.retrieve(question, top_n=top_n)
    if not passages:
        return {"respuesta": "No encontré información relevante en el manual.",
                "citas": [], "abstuvo": True}

    context = "\n\n".join(
        f"[{i}] ({p.citation})\n{p.chunk['text']}" for i, p in enumerate(passages, 1))
    out = _parse_json(llm.complete(NORMA_QA_PROMPT.format(context=context, question=question)))

    # Abstención: respuesta no parseable o el LLM declaró que el manual no cubre.
    if not out or out.get("encontrada") is False:
        motivo = (out or {}).get("respuesta") or "El manual no cubre esta consulta."
        return {"respuesta": motivo, "citas": [], "abstuvo": True}

    # Citas atadas a los pasajes REALES que el LLM dijo haber usado (con fallback).
    usados = [n for n in (out.get("pasajes") or [])
              if isinstance(n, int) and 1 <= n <= len(passages)]
    citas = [passages[n - 1].citation for n in usados] or [p.citation for p in passages]
    seen: set = set()
    citas = [c for c in citas if not (c in seen or seen.add(c))]  # dedup, conserva orden
    return {"respuesta": out.get("respuesta", ""), "citas": citas, "abstuvo": False}
