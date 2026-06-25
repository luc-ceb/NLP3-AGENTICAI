"""Caso 2 — Diagnóstico de un PDV (LangGraph).

Cruza KPIs de volumen/tráfico (ventas) contra el manual operativo y produce un
diagnóstico accionable con cita, o se ABSTIENE si no hay un problema que auditar.

Grafo:

    kpis --> detectar --> (¿hay desvío negativo?) --no--> END   (abstención, SIN LLM)
                               |
                              sí
                               v
                            auditar --> (¿la norma aplica?) --no--> END  (abstención)
                                              |
                                             sí
                                              v
                                         diagnosticar --> END

Eficiencia de ruteo (requisito del MVP): los nodos `kpis` y `detectar` son SQL +
reglas (sin LLM). El nodo `auditar` (RAG) solo se invoca si `detectar` halló un
desvío negativo accionable; si la sucursal no tiene desvío (o rinde sobre la red),
el grafo termina limpio para ese PDV sin gastar una sola llamada al LLM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from ..kpi import (
    Desvio, PDVKpis, Periodo, compute_kpis, detectar_desvio, periodo_por_defecto,
)
from .auditor_rag import AuditResult, _parse_json

DIAGNOSE_PROMPT = """Sos el supervisor de diagnóstico operativo de una cadena de heladerías.
Una sucursal presenta un DESVÍO en sus KPIs de ventas. El Auditor Normativo ya evaluó la
norma aplicable (VEREDICTO) y recuperó el CONTEXTO del manual operativo. Redactá un
diagnóstico accionable apoyado ÚNICAMENTE en ese contexto.

Reglas:
- Basate solo en el CONTEXTO normativo y el VEREDICTO. No inventes normas ni cifras.
- El diagnóstico conecta el desvío observado con la práctica del manual (causa probable).
- La acción es concreta, verificable y ejecutable por el encargado del PDV.
- Si el contexto no alcanza para fundamentar una acción, abstenete (fundada=false).
Respondé SOLO JSON:
{{"fundada": true|false, "diagnostico": "<resumen breve>", "accion": "<acción concreta>"}}

DESVÍO OBSERVADO:
{desvio}

VEREDICTO DE AUDITORÍA: {veredicto} — {justificacion}

CONTEXTO NORMATIVO (pasajes del manual con su cita entre corchetes):
{context}
"""


class Caso2State(TypedDict, total=False):
    pdv: str
    periodo: Periodo
    kpis: PDVKpis
    desvio: Desvio | None
    audit: AuditResult
    diagnostico: str
    accion: str
    citas: list[str]
    abstuvo: bool
    motivo: str            # motivo de abstención (cuando abstuvo=True)


@dataclass
class Caso2Result:
    """Salida del Caso 2: {pdv, periodo, desvío, diagnóstico, acción, cita | abstención}."""
    pdv: str
    periodo: str               # etiqueta del período analizado ('YYYY-MM' o rango)
    kpis: PDVKpis | None = None
    desvio: Desvio | None = None
    diagnostico: str = ""
    accion: str = ""
    citas: list[str] = field(default_factory=list)
    abstuvo: bool = True
    motivo: str = ""
    verdict: str = ""

    @property
    def severidad(self) -> float:
        """Severidad del desvío (0 si abstuvo) — para rankear PDVs en Caso 5."""
        return self.desvio.severidad if self.desvio else 0.0

    @property
    def orden(self) -> tuple[int, float]:
        """Clave de orden para Caso 5: desvíos primero (por prioridad/severidad)."""
        return self.desvio.orden if self.desvio else (99, 0.0)

    def __str__(self) -> str:
        cab = f"PDV {self.pdv[:8]} · {self.periodo}"
        if self.abstuvo:
            return f"{cab}\nABSTENCIÓN: {self.motivo}"
        cites = "\n".join(f"  · {c}" for c in self.citas) or "  —"
        desv = self.desvio.detalle if self.desvio else "—"
        return (f"{cab}\n"
                f"DESVÍO: {desv}\n"
                f"DIAGNÓSTICO: {self.diagnostico}\n"
                f"ACCIÓN: {self.accion}\n"
                f"CITAS (manual operativo):\n{cites}")


class DiagnosticadorPDV:
    """Caso 2: diagnostica una sucursal en un mes (KPIs vs manual).

    Args:
        con: conexión DuckDB (solo lectura) con la tabla de ventas.
        auditor: ``NormativeAuditor`` (RAG + CRAG) ya construido con su retriever.
        llm: modelo para la síntesis del diagnóstico (el MVP usa un único modelo;
            por defecto reutiliza el del auditor).
    """

    def __init__(self, con, auditor, llm=None):
        self.con = con
        self.auditor = auditor
        self.llm = llm or auditor.llm
        self.app = self._build()

    def _build(self):
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(Caso2State)
        g.add_node("kpis", self._kpis)
        g.add_node("detectar", self._detectar)
        g.add_node("auditar", self._auditar)
        g.add_node("diagnosticar", self._diagnosticar)

        g.add_edge(START, "kpis")
        g.add_edge("kpis", "detectar")
        # Si no hay desvío negativo accionable -> END (sin LLM).
        g.add_conditional_edges("detectar", self._hay_desvio,
                                {"auditar": "auditar", "fin": END})
        # Si no se recuperó contexto del manual -> END (abstención, sin síntesis).
        # No se gatea por el veredicto de cumplimiento (cumple/no_cumple/sin_norma):
        # ese veredicto es frágil para hipótesis basadas en KPIs y abstiene de más.
        # Quien decide si hay acción fundada es la síntesis (fundada=true/false).
        g.add_conditional_edges("auditar", self._hay_contexto,
                                {"diagnosticar": "diagnosticar", "fin": END})
        g.add_edge("diagnosticar", END)
        return g.compile()

    # --- nodos (kpis y detectar: sin LLM) ---
    def _kpis(self, state: Caso2State) -> dict:
        k = compute_kpis(self.con, state["pdv"], state["periodo"])
        return {"kpis": k}

    def _detectar(self, state: Caso2State) -> dict:
        return {"desvio": detectar_desvio(state["kpis"])}

    # --- ruteo: ¿vale la pena invocar al LLM? ---
    def _hay_desvio(self, state: Caso2State) -> str:
        return "auditar" if state.get("desvio") is not None else "fin"

    def _hay_contexto(self, state: Caso2State) -> str:
        audit = state.get("audit")
        # Solo se intenta diagnosticar si hubo pasajes del manual que recuperar.
        # El veredicto se conserva como CONTEXTO de la síntesis, no como gate.
        return "diagnosticar" if (audit is not None and audit.passages) else "fin"

    # --- nodos RAG/LLM (solo si hubo desvío) ---
    def _auditar(self, state: Caso2State) -> dict:
        d: Desvio = state["desvio"]
        # La hipótesis operativa del desvío es la consulta normativa a auditar.
        claim = f"{d.detalle} {d.hipotesis}"
        res = self.auditor.audit(claim)
        return {"audit": res, "citas": list(res.citations)}

    def _diagnosticar(self, state: Caso2State) -> dict:
        d: Desvio = state["desvio"]
        audit: AuditResult = state["audit"]
        context = self.auditor.retriever.format_context(audit.passages)
        desvio_txt = "\n".join([d.detalle] + [f"Contexto: {c}" for c in d.contexto])
        out = _parse_json(self.llm.complete(DIAGNOSE_PROMPT.format(
            desvio=desvio_txt, veredicto=audit.verdict,
            justificacion=audit.justification, context=context)))

        if not out or out.get("fundada") is False:
            # El LLM no pudo fundamentar una acción en la norma -> abstención.
            motivo = ((out or {}).get("diagnostico")
                      or "El manual no permite fundamentar una acción para este desvío.")
            return {"abstuvo": True, "motivo": motivo}
        return {"abstuvo": False,
                "diagnostico": out.get("diagnostico", ""),
                "accion": out.get("accion", "")}

    # --- API pública ---
    def diagnosticar(self, pdv: str,
                     periodo: Periodo | str | None = None) -> Caso2Result:
        """Ejecuta el Caso 2 para una sucursal y un período.

        ``periodo`` admite un :class:`Periodo` (ventana de fechas), un 'YYYY-MM'
        (atajo de mes) o None. Si es None usa el último mes COMPLETO disponible
        (evita comparar contra un mes en curso).
        """
        periodo = (Periodo.coerce(periodo) if periodo is not None
                   else periodo_por_defecto(self.con))
        s = self.app.invoke({"pdv": pdv, "periodo": periodo})

        desvio = s.get("desvio")
        audit = s.get("audit")
        # Motivo de abstención según en qué punto terminó el grafo.
        if s.get("abstuvo", True):
            if desvio is None:
                motivo = ("No se detectó un desvío negativo relevante "
                          "(la sucursal está en o por encima de la media de la red).")
            elif audit is None or not audit.passages:
                motivo = ("Se detectó un desvío pero no se recuperó contexto del "
                          f"manual para auditarlo. Desvío: {desvio.detalle}")
            else:
                # La síntesis no pudo fundamentar una acción en el manual.
                motivo = s.get("motivo", "Abstención.")
        else:
            motivo = ""

        return Caso2Result(
            pdv=pdv, periodo=periodo.etiqueta, kpis=s.get("kpis"), desvio=desvio,
            diagnostico=s.get("diagnostico", ""), accion=s.get("accion", ""),
            citas=s.get("citas", []), abstuvo=s.get("abstuvo", True),
            motivo=motivo, verdict=(audit.verdict if audit else ""))
