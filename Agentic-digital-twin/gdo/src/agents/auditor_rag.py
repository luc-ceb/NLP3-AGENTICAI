"""Agente Auditor Normativo: RAG + veredicto de cumplimiento con auto-reflexión (CRAG).

Dado un hecho/consulta, recupera la norma aplicable (HybridRetriever), evalúa la
calidad del contexto (grader CRAG) y, si es flojo, reescribe la consulta y reintenta.
Luego emite un veredicto cumple/no_cumple/sin_norma con citas direccionables.

El LLM se inyecta (BaseLLM): en producción un OllamaClient; en tests, un stub.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

GRADER_PROMPT = """Sos un evaluador de recuperación (estilo CRAG).
¿El CONTEXTO recuperado contiene información suficiente y relevante para evaluar la CONSULTA?
Clasificá en: correcto (relevante y suficiente), ambiguo (parcial), incorrecto (irrelevante).
Respondé SOLO JSON: {{"grado": "correcto|ambiguo|incorrecto", "motivo": "<breve>"}}

CONSULTA: {query}

CONTEXTO:
{context}
"""

REWRITE_PROMPT = """La búsqueda anterior trajo contexto insuficiente. Motivo: {reason}
Reescribí la consulta para mejorar la recuperación en un manual operativo y documentos
normativos (usá sinónimos del dominio, términos más específicos). 
Respondé SOLO con la nueva consulta, sin comillas ni explicación.

Consulta original: {query}
Nueva consulta:"""

VERDICT_PROMPT = """Sos un auditor normativo de una cadena de heladerías.
Dado un HECHO observado y el CONTEXTO normativo recuperado (cada pasaje viene con su cita
entre corchetes), determiná si el hecho CUMPLE o NO la norma.

Reglas:
- Basate ÚNICAMENTE en el contexto. No inventes normas.
- Si no hay norma aplicable en el contexto, veredicto "sin_norma".
- Citá las fuentes (el texto entre corchetes) en las que te apoyás.
Respondé SOLO JSON:
{{"veredicto": "cumple|no_cumple|sin_norma", "justificacion": "<breve>", "citas": ["<cita>", ...]}}

HECHO: {claim}

CONTEXTO:
{context}
"""


def _parse_json(text: str) -> dict:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I).strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", t, re.S)  # último recurso: primer objeto {...}
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                pass
    return {}


@dataclass
class AuditResult:
    query: str
    verdict: str
    justification: str
    citations: list[str] = field(default_factory=list)
    grade: str = ""
    n_retries: int = 0
    passages: list = field(default_factory=list)

    def __str__(self) -> str:
        cites = "; ".join(self.citations) if self.citations else "—"
        return (f"Consulta: {self.query}\n"
                f"Veredicto: {self.verdict.upper()}  (grado contexto: {self.grade}, "
                f"reintentos: {self.n_retries})\n"
                f"Justificación: {self.justification}\n"
                f"Citas: {cites}")


class NormativeAuditor:
    def __init__(self, retriever, llm, max_retries: int = 1, top_n: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.max_retries = max_retries
        self.top_n = top_n

    def _grade(self, query: str, context: str) -> dict:
        out = _parse_json(self.llm.complete(GRADER_PROMPT.format(query=query, context=context)))
        return out or {"grado": "ambiguo", "motivo": "no se pudo evaluar"}

    def _rewrite(self, query: str, reason: str) -> str:
        new = self.llm.complete(REWRITE_PROMPT.format(query=query, reason=reason)).strip()
        new = re.sub(r"<think>.*?</think>", "", new, flags=re.S | re.I).strip().strip('"')
        return new.splitlines()[0].strip() if new else query

    def _verdict(self, claim: str, context: str) -> dict:
        out = _parse_json(self.llm.complete(VERDICT_PROMPT.format(claim=claim, context=context)))
        return out or {"veredicto": "sin_norma", "justificacion": "respuesta no parseable", "citas": []}

    def audit(self, claim: str) -> AuditResult:
        query, grade, attempt = claim, "incorrecto", 0
        passages = []
        for attempt in range(self.max_retries + 1):
            passages = self.retriever.retrieve(query, top_n=self.top_n)
            context = self.retriever.format_context(passages)
            grade_obj = (self._grade(claim, context) if passages
                         else {"grado": "incorrecto", "motivo": "sin contexto"})
            grade = grade_obj.get("grado", "ambiguo")
            if grade == "correcto" or attempt == self.max_retries:
                break
            query = self._rewrite(claim, grade_obj.get("motivo", ""))
            log.info("CRAG: contexto '%s' -> reescribo a: %s", grade, query)

        if not passages:
            return AuditResult(claim, "sin_norma", "No se recuperó contexto normativo.",
                               [], grade, attempt, passages)

        v = self._verdict(claim, self.retriever.format_context(passages))
        citas = v.get("citas") or [p.citation for p in passages[:3]]
        return AuditResult(claim, v.get("veredicto", "sin_norma"),
                           v.get("justificacion", ""), citas, grade, attempt, passages)
