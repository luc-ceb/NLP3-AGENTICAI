"""Agente Supervisor (LangGraph): orquesta Analista + Auditor y reconcilia.

Grafo: analyze -> claims -> audit -> reconcile.
  - analyze  : Analista (Text-to-SQL) obtiene los HECHOS de las tablas.
  - claims   : convierte los hechos en afirmaciones verificables contra la norma.
  - audit    : Auditor Normativo evalúa cada afirmación (veredicto + pasajes).
  - reconcile: reconcilia realidad vs norma -> diagnóstico (causa raíz + acciones).

Las citas del reporte se atan a los pasajes REALES recuperados por el Auditor,
no a lo que transcribe el LLM (trazabilidad confiable).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from .auditor_rag import _parse_json

CLAIMS_PROMPT = """Dada la PREGUNTA del usuario y los HECHOS (datos de las tablas),
generá entre 1 y 3 afirmaciones breves y verificables contra normas operativas.

Reglas:
- Conservá los términos específicos de la PREGUNTA y los HECHOS. NO generalices:
  si dice "derretido", la afirmación dice "derretido", no "mala experiencia".
- Incluí las cifras concretas de los HECHOS.
- Si el hecho sugiere un problema operativo, nombrá la dimensión normativa
  relevante para poder auditarla (ej.: cadena de frío, conservación, higiene,
  despacho, recepción de mercadería).
Respondé SOLO JSON: {{"claims": ["<afirmacion>", ...]}}

PREGUNTA: {question}

HECHOS:
{facts}
"""

RECONCILE_PROMPT = """Sos el supervisor de diagnóstico operativo de una heladería.
Dada la PREGUNTA, los HECHOS (realidad) y los HALLAZGOS de auditoría (afirmación -> veredicto
contra la norma), reconciliá realidad vs norma y producí un diagnóstico accionable.
Basate solo en los hallazgos provistos.
Respondé SOLO JSON:
{{"diagnostico": "<resumen>", "causa_raiz": "<causa>", "acciones": ["<accion>", ...]}}

PREGUNTA: {question}

HECHOS:
{facts}

HALLAZGOS:
{findings}
"""


class GDOState(TypedDict, total=False):
    question: str
    sql: str
    facts: str
    claims: list[str]
    audits: list[dict]
    citations: list[str]
    diagnosis: str


@dataclass
class DiagnosisResult:
    question: str
    sql: str
    facts: str
    claims: list[str] = field(default_factory=list)
    audits: list[dict] = field(default_factory=list)
    diagnosis: str = ""
    citations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        aud = "\n".join(f"  - [{a['verdict']}] {a['claim']}" for a in self.audits) or "  —"
        cites = "\n".join(f"  · {c}" for c in self.citations) or "  —"
        return (f"PREGUNTA: {self.question}\n\n"
                f"DIAGNÓSTICO:\n{self.diagnosis}\n\n"
                f"HALLAZGOS:\n{aud}\n\n"
                f"CITAS (fuentes reales):\n{cites}")


class SupervisorAgent:
    def __init__(self, analyst, auditor, llm, max_claims: int = 3):
        self.analyst = analyst
        self.auditor = auditor
        self.llm = llm
        self.max_claims = max_claims
        self.app = self._build()

    def _build(self):
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(GDOState)
        g.add_node("analyze", self._analyze)
        g.add_node("claims", self._claims)
        g.add_node("audit", self._audit)
        g.add_node("reconcile", self._reconcile)
        g.add_edge(START, "analyze")
        g.add_edge("analyze", "claims")
        g.add_edge("claims", "audit")
        g.add_edge("audit", "reconcile")
        g.add_edge("reconcile", END)
        return g.compile()

    # --- nodos ---
    def _analyze(self, state: GDOState) -> dict:
        r = self.analyst.ask(state["question"])
        if getattr(r, "ok", False) and r.rows is not None:
            facts = r.rows.head(20).to_string(index=False)
        else:
            facts = f"[sin datos: {getattr(r, 'reason', 'desconocido')}]"
        return {"sql": getattr(r, "sql", ""), "facts": facts}

    def _claims(self, state: GDOState) -> dict:
        out = _parse_json(self.llm.complete(
            CLAIMS_PROMPT.format(question=state["question"], facts=state["facts"])))
        claims = (out.get("claims") or [])[: self.max_claims]
        return {"claims": claims or [state["question"]]}

    def _audit(self, state: GDOState) -> dict:
        audits, cites = [], []
        for claim in state["claims"]:
            res = self.auditor.audit(claim)
            audits.append({"claim": claim, "verdict": res.verdict,
                           "justification": res.justification})
            cites += [p.citation for p in res.passages]      # citas REALES de los pasajes
        seen: set = set()
        cites = [c for c in cites if not (c in seen or seen.add(c))]
        return {"audits": audits, "citations": cites}

    def _reconcile(self, state: GDOState) -> dict:
        findings = "\n".join(
            f"- {a['claim']} -> {a['verdict']}: {a['justification']}" for a in state["audits"])
        out = _parse_json(self.llm.complete(RECONCILE_PROMPT.format(
            question=state["question"], facts=state["facts"], findings=findings)))
        text = out.get("diagnostico", "") or "Sin diagnóstico."
        if out.get("causa_raiz"):
            text += f"\nCausa raíz: {out['causa_raiz']}"
        if out.get("acciones"):
            text += "\nAcciones sugeridas: " + "; ".join(out["acciones"])
        return {"diagnosis": text}

    def diagnose(self, question: str) -> DiagnosisResult:
        s = self.app.invoke({"question": question})
        return DiagnosisResult(
            question=question, sql=s.get("sql", ""), facts=s.get("facts", ""),
            claims=s.get("claims", []), audits=s.get("audits", []),
            diagnosis=s.get("diagnosis", ""), citations=s.get("citations", []))
