"""Caso 5 — Plan mensual de diagnóstico (LangGraph que envuelve al Caso 2).

Itera el Caso 2 sobre todas las sucursales para un mismo mes, las rankea por
severidad del desvío y produce un reporte mensual en Markdown.

Grafo:  diagnosticos --> sintetizar --> END
  - diagnosticos: corre el Caso 2 por cada PDV (sin LLM si la sucursal no tiene
    desvío) y ordena los resultados por severidad.
  - sintetizar: UNA sola llamada al LLM que agrupa problemas comunes y prioriza
    acciones. Las cifras y las citas del reporte son DETERMINÍSTICAS (se arman en
    código a partir de los Caso2Result); el LLM solo redacta la narrativa.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from ..kpi import Periodo, list_pdvs, periodo_por_defecto
from .auditor_rag import _parse_json
from .caso2 import Caso2Result, DiagnosticadorPDV

SINTESIS_PROMPT = """Sos el supervisor regional de una cadena de heladerías. Tenés los
diagnósticos del período para cada sucursal (con su desvío de KPIs y la acción fundada en
el manual). Redactá la SÍNTESIS ejecutiva del período, BREVE y accionable.

Reglas:
- Basate SOLO en los diagnósticos provistos. No inventes cifras ni sucursales.
- Agrupá los problemas que se repiten en varias sucursales (máx. 3 temas comunes).
- Priorizá acciones por impacto (máx. 5, las sucursales más severas primero).
- Frases cortas. No repitas los diagnósticos completos.
Respondé SOLO JSON (sin texto extra):
{{"resumen": "<2-3 frases ejecutivas del estado del mes>",
  "temas_comunes": [{{"tema": "<patrón>", "sucursales": ["<id8>", ...],
                      "recomendacion": "<acción transversal breve>"}}],
  "prioridades": ["<acción prioritaria breve>", ...]}}

DIAGNÓSTICOS DEL PERÍODO ({periodo}):
{diagnosticos}
"""


class Caso5State(TypedDict, total=False):
    periodo: Periodo
    pdvs: list[str]
    resultados: list[Caso2Result]
    reporte: str


@dataclass
class PlanMensualResult:
    periodo: str               # etiqueta del período analizado ('YYYY-MM' o rango)
    resultados: list[Caso2Result] = field(default_factory=list)  # ya ordenados por severidad
    reporte: str = ""

    @property
    def accionables(self) -> list[Caso2Result]:
        return [r for r in self.resultados if not r.abstuvo]

    def __str__(self) -> str:
        return self.reporte


def _estado(r: Caso2Result) -> str:
    """Estado legible de una sucursal según dónde terminó su Caso 2."""
    if not r.abstuvo:
        return "Diagnosticado"
    if r.desvio is not None:
        return "Desvío sin norma aplicable"
    return "Sin desvío"


# Alcance del desvío por nivel de prioridad: explica por qué un desvío de tienda
# vs la red rankea sobre una caída de producto de % mayor pero impacto local.
_ALCANCE = {0: "Sucursal vs red", 1: "Interanual", 2: "Producto"}


def _alcance(r: Caso2Result) -> str:
    return _ALCANCE.get(r.desvio.prioridad, "—") if r.desvio else "—"


class PlanificadorMensual:
    """Caso 5: genera el plan mensual rankeado a partir del Caso 2 por sucursal."""

    def __init__(self, con, auditor, llm=None):
        self.con = con
        self.llm = llm or auditor.llm
        self.diag = DiagnosticadorPDV(con, auditor, self.llm)
        self.app = self._build()

    def _build(self):
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(Caso5State)
        g.add_node("diagnosticos", self._diagnosticos)
        g.add_node("sintetizar", self._sintetizar)
        g.add_edge(START, "diagnosticos")
        g.add_edge("diagnosticos", "sintetizar")
        g.add_edge("sintetizar", END)
        return g.compile()

    # --- nodos ---
    def _diagnosticos(self, state: Caso5State) -> dict:
        periodo = state["periodo"]
        pdvs = state.get("pdvs") or list_pdvs(self.con)
        resultados = [self.diag.diagnosticar(p, periodo) for p in pdvs]
        # Orden: desvíos primero (por prioridad/severidad), abstenciones al final.
        resultados.sort(key=lambda r: r.orden)
        return {"resultados": resultados, "pdvs": pdvs}

    def _sintetizar(self, state: Caso5State) -> dict:
        periodo: Periodo = state["periodo"]
        resultados: list[Caso2Result] = state["resultados"]
        accionables = [r for r in resultados if not r.abstuvo]

        # Síntesis con LLM SOLO sobre los diagnósticos accionables (1 llamada).
        # Input compacto: el LLM agrupa/prioriza, no necesita el diagnóstico completo.
        sintesis = {}
        if accionables:
            bloque = "\n".join(
                f"- {r.pdv[:8]} (sev {r.severidad:.0f}, {_alcance(r)}): "
                f"{r.desvio.detalle} Diagnóstico: {r.diagnostico[:220]}"
                for r in accionables)
            sintesis = _parse_json(self.llm.complete(
                SINTESIS_PROMPT.format(periodo=periodo.etiqueta, diagnosticos=bloque))) or {}

        # Fallback determinístico: si la síntesis (LLM) no trae prioridades, derivarlas
        # de las sucursales accionables en orden de severidad. El reporte nunca queda
        # sin una lista de acciones priorizadas, aun si la llamada al LLM falla/trunca.
        if accionables and not sintesis.get("prioridades"):
            sintesis["prioridades"] = [
                f"{r.pdv[:8]} ({r.desvio.titulo}): {r.accion[:160]}" for r in accionables]

        return {"reporte": self._render_markdown(periodo.etiqueta, resultados, sintesis)}

    # --- render determinístico (cifras y citas exactas) ---
    def _render_markdown(self, periodo: str, resultados: list[Caso2Result],
                         sintesis: dict) -> str:
        n_diag = sum(1 for r in resultados if not r.abstuvo)
        n_sin_norma = sum(1 for r in resultados if r.abstuvo and r.desvio is not None)
        n_ok = sum(1 for r in resultados if r.desvio is None)
        L: list[str] = []
        L.append(f"# Plan de diagnóstico operativo — {periodo}")
        L.append("")
        L.append(f"_{len(resultados)} sucursales · {n_diag} con acción · "
                 f"{n_sin_norma} con desvío sin norma · {n_ok} sin desvío_")
        L.append("")

        # Resumen ejecutivo (narrativa del LLM, opcional).
        if sintesis.get("resumen"):
            L.append("## Resumen ejecutivo")
            L.append(sintesis["resumen"])
            L.append("")

        # Ranking por severidad (determinístico).
        L.append("## Ranking de sucursales por severidad")
        L.append("")
        L.append("| # | Sucursal | Desvío | Alcance | Severidad | Estado |")
        L.append("|---|----------|--------|---------|-----------|--------|")
        for i, r in enumerate(resultados, 1):
            desv = r.desvio.detalle if r.desvio else "—"
            sev = f"{r.severidad:.0f}%" if r.desvio else "—"
            L.append(f"| {i} | {r.pdv[:8]} | {desv} | {_alcance(r)} | {sev} | "
                     f"{_estado(r)} |")
        L.append("")

        # Detalle por sucursal accionable (diagnóstico + acción + citas reales).
        accionables = [r for r in resultados if not r.abstuvo]
        if accionables:
            L.append("## Diagnósticos y acciones")
            L.append("")
            for r in accionables:
                L.append(f"### {r.pdv[:8]} — {r.desvio.titulo if r.desvio else ''}")
                if r.desvio:
                    L.append(f"- **Desvío:** {r.desvio.detalle}")
                L.append(f"- **Diagnóstico:** {r.diagnostico}")
                L.append(f"- **Acción:** {r.accion}")
                citas = "; ".join(r.citas) if r.citas else "—"
                L.append(f"- **Citas (manual):** {citas}")
                L.append("")

        # Temas comunes (narrativa del LLM).
        temas = sintesis.get("temas_comunes") or []
        if temas:
            L.append("## Temas comunes")
            L.append("")
            for t in temas:
                sucs = ", ".join(t.get("sucursales", []))
                L.append(f"- **{t.get('tema', '')}** ({sucs}): {t.get('recomendacion', '')}")
            L.append("")

        # Acciones prioritarias (narrativa del LLM).
        prioridades = sintesis.get("prioridades") or []
        if prioridades:
            L.append("## Acciones prioritarias")
            L.append("")
            for i, p in enumerate(prioridades, 1):
                L.append(f"{i}. {p}")
            L.append("")

        return "\n".join(L).rstrip() + "\n"

    # --- API pública ---
    def generar(self, periodo: Periodo | str | None = None,
                pdvs: list[str] | None = None) -> PlanMensualResult:
        """Genera el plan de diagnóstico para un período.

        ``periodo`` admite un :class:`Periodo` (ventana de fechas), un 'YYYY-MM'
        (atajo de mes) o None (último mes completo disponible).
        """
        periodo = (Periodo.coerce(periodo) if periodo is not None
                   else periodo_por_defecto(self.con))
        init: Caso5State = {"periodo": periodo}
        if pdvs:
            init["pdvs"] = pdvs
        s = self.app.invoke(init)
        return PlanMensualResult(periodo=periodo.etiqueta,
                                 resultados=s.get("resultados", []),
                                 reporte=s.get("reporte", ""))
