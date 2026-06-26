"""Caso 3 — Diagnóstico combinado por PDV: ventas + encuestas (LangGraph).

Cruza KPIs de ventas con las quejas de clientes de la misma sucursal para
producir un diagnóstico más rico: el desvío de datos (ventas) se correlaciona
con la voz del cliente (encuestas de mala experiencia) y se contrasta contra el
manual operativo.

Grafo:
    kpis ──────────────────────────────┐
                                       ▼
    encuestas ──────────→  fusionar ──→ (¿hay señal?) ──no──→ END  (abstención)
    (SQL + keyword)            │
                              sí
                               ▼
                           auditar ──→ (¿hay contexto?) ──no──→ END
                                               │
                                              sí
                                               ▼
                                         diagnosticar ──→ END

Eficiencia: `kpis` y `encuestas` son SQL + reglas (sin LLM). La clasificación
de quejas usa keyword matching (sin LLM). El LLM se invoca UNA sola vez en
`diagnosticar`, solo si hay señal y contexto normativo.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, TypedDict

from ..kpi import (
    Desvio, PDVKpis, Periodo, compute_kpis, detectar_desvio, periodo_por_defecto,
)
from .auditor_rag import AuditResult, _parse_json
from .caso1 import TEMAS

# Palabras clave por tema (raíces sin acento) para clasificación sin LLM.
_KEYWORDS: dict[str, list[str]] = {
    "cadena_frio":         ["derretid", "blando", "liquid", "caliente", "deshiel", "tibio"],
    "calidad_producto":    ["sabor", "textur", "crema", "feo", "insipid", "duro", "mal gusto"],
    "cantidad_peso":       ["cantidad", "peso", "porcion", "medio", "lleno", "escas", "poco", "vacio"],
    "atencion_cliente":    ["atencion", "trato", "personal", "amabilid", "maltrat", "grosero", "descort"],
    "limpieza_local":      ["limpiez", "higiene", "sucio", "sucied", "bano", "instalac"],
    "precios":             ["precio", "caro", "carest", "aument", "cobr", "costo"],
    "delivery_logistica":  ["delivery", "domicilio", "pedido", "demor", "entrega", "llego"],
    "fidelizacion_puntos": ["punto", "descuent", "promo", "app", "fideliz", "beneficio"],
}


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


def _classify(text: str) -> str:
    """Clasifica un texto en el tema de mayor coincidencia (o 'otros')."""
    n = _norm(text)
    hits = {tema: sum(1 for kw in kws if kw in n) for tema, kws in _KEYWORDS.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "otros"


DIAGNOSE_PROMPT = """Sos el supervisor de diagnóstico operativo de una cadena de heladerías.
Una sucursal presenta señales de alerta provenientes de DOS fuentes:
  1. KPIs de ventas (desempeño vs la red o interanual).
  2. Encuestas de mala experiencia de los clientes del mismo período.

El Auditor Normativo ya evaluó la norma aplicable (VEREDICTO) y recuperó el
CONTEXTO del manual operativo. Redactá un diagnóstico integrado y accionable
apoyado ÚNICAMENTE en ese contexto.

Reglas:
- Integrá ambas señales si están presentes; si solo hay una, diagnosticá sobre esa.
- Conectá el desvío observado y/o las quejas con la práctica del manual.
- La acción es concreta, verificable y ejecutable por el encargado del PDV.
- Si el contexto no alcanza para fundamentar una acción, abstenete (fundada=false).
Respondé SOLO JSON:
{{"fundada": true|false, "diagnostico": "<resumen breve>", "accion": "<acción concreta>"}}

SEÑAL DE VENTAS: {senal_ventas}
SEÑAL DE ENCUESTAS: {senal_encuestas}
VEREDICTO DE AUDITORÍA: {veredicto} — {justificacion}
CONTEXTO NORMATIVO (pasajes del manual con su cita entre corchetes):
{context}
"""


class Caso3State(TypedDict, total=False):
    pdv: str
    periodo: Periodo
    kpis: PDVKpis
    desvio: Optional[Desvio]
    n_quejas: int
    tema_quejas: Optional[str]
    audit: AuditResult
    diagnostico: str
    accion: str
    citas: list[str]
    abstuvo: bool
    motivo: str


@dataclass
class Caso3Result:
    """Salida del Caso 3: diagnóstico combinado ventas + encuestas por PDV."""
    pdv: str
    periodo: str
    kpis: Optional[PDVKpis] = None
    desvio: Optional[Desvio] = None
    n_quejas: int = 0
    tema_quejas: Optional[str] = None
    diagnostico: str = ""
    accion: str = ""
    citas: list[str] = field(default_factory=list)
    abstuvo: bool = True
    motivo: str = ""
    verdict: str = ""

    @property
    def severidad(self) -> float:
        return self.desvio.severidad if self.desvio else 0.0

    def __str__(self) -> str:
        cab = f"PDV {self.pdv[:8]} · {self.periodo}"
        if self.abstuvo:
            return f"{cab}\nABSTENCIÓN: {self.motivo}"
        cites = "\n".join(f"  · {c}" for c in self.citas) or "  —"
        desv = self.desvio.detalle if self.desvio else "Sin desvío relevante en ventas."
        enc = (f"{self.n_quejas} quejas sobre '{self.tema_quejas}'"
               if self.tema_quejas else
               (f"{self.n_quejas} quejas sin tema dominante." if self.n_quejas
                else "Sin quejas de mala experiencia en el período."))
        return (f"{cab}\n"
                f"VENTAS: {desv}\n"
                f"ENCUESTAS: {enc}\n"
                f"DIAGNÓSTICO: {self.diagnostico}\n"
                f"ACCIÓN: {self.accion}\n"
                f"CITAS (manual operativo):\n{cites}")


class DiagnosticadorCombinado:
    """Caso 3: diagnóstico por PDV cruzando KPIs de ventas y encuestas de mala experiencia.

    Args:
        con: conexión DuckDB (solo lectura).
        auditor: ``NormativeAuditor`` (RAG + CRAG) ya construido.
        llm: modelo para la síntesis final (por defecto reutiliza el del auditor).
    """

    def __init__(self, con, auditor, llm=None):
        self.con = con
        self.auditor = auditor
        self.llm = llm or auditor.llm
        self.app = self._build()

    def _build(self):
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(Caso3State)
        g.add_node("kpis", self._kpis)
        g.add_node("encuestas", self._encuestas)
        g.add_node("auditar", self._auditar)
        g.add_node("diagnosticar", self._diagnosticar)

        g.add_edge(START, "kpis")
        g.add_edge("kpis", "encuestas")
        g.add_conditional_edges("encuestas", self._hay_senal,
                                {"auditar": "auditar", "fin": END})
        g.add_conditional_edges("auditar", self._hay_contexto,
                                {"diagnosticar": "diagnosticar", "fin": END})
        g.add_edge("diagnosticar", END)
        return g.compile()

    # --- nodos sin LLM ---

    def _kpis(self, state: Caso3State) -> dict:
        k = compute_kpis(self.con, state["pdv"], state["periodo"])
        return {"kpis": k, "desvio": detectar_desvio(k)}

    def _encuestas(self, state: Caso3State) -> dict:
        """Carga y clasifica quejas de mala experiencia del PDV (keyword matching)."""
        row = self.con.execute(
            "SELECT DISTINCT numero FROM datos_ventas WHERE branchofficeid = ? LIMIT 1",
            [state["pdv"]],
        ).fetchone()
        if not row:
            return {"n_quejas": 0, "tema_quejas": None}

        numero = row[0]
        p: Periodo = state["periodo"]
        textos = self.con.execute(
            """SELECT origen_respuesta_texto
               FROM encuestas_mala_experiencia
               WHERE numero = ?
                 AND origen_fhrespuesta::DATE BETWEEN ? AND ?
                 AND origen_respuesta_texto IS NOT NULL
                 AND trim(origen_respuesta_texto) <> ''""",
            [numero, p.desde, p.hasta],
        ).fetchall()

        if not textos:
            return {"n_quejas": 0, "tema_quejas": None}

        conteo = Counter(_classify(t[0]) for t in textos)
        dominante, _ = conteo.most_common(1)[0]
        return {
            "n_quejas": len(textos),
            "tema_quejas": dominante if dominante != "otros" else None,
        }

    # --- routing ---

    def _hay_senal(self, state: Caso3State) -> str:
        if state.get("desvio") is not None or state.get("n_quejas", 0) > 0:
            return "auditar"
        return "fin"

    def _hay_contexto(self, state: Caso3State) -> str:
        audit = state.get("audit")
        return "diagnosticar" if (audit is not None and audit.passages) else "fin"

    # --- nodos RAG/LLM ---

    def _auditar(self, state: Caso3State) -> dict:
        claims = []
        d: Optional[Desvio] = state.get("desvio")
        if d:
            claims.append(f"{d.detalle} {d.hipotesis}")
        tema = state.get("tema_quejas")
        n = state.get("n_quejas", 0)
        if tema and n > 0:
            consulta_norma = TEMAS.get(tema, ("", ""))[1]
            if consulta_norma:
                claims.append(
                    f"{n} quejas de clientes sobre '{tema}': {consulta_norma}")
        claim = " | ".join(claims) if claims else "Revisión operativa general"
        res = self.auditor.audit(claim)
        return {"audit": res, "citas": list(res.citations)}

    def _diagnosticar(self, state: Caso3State) -> dict:
        d: Optional[Desvio] = state.get("desvio")
        tema = state.get("tema_quejas")
        n = state.get("n_quejas", 0)
        audit: AuditResult = state["audit"]
        context = self.auditor.retriever.format_context(audit.passages)

        senal_ventas = d.detalle if d else "Sin desvío relevante en ventas."
        if tema and n > 0:
            senal_encuestas = (f"{n} quejas sobre '{tema}' "
                               f"({TEMAS[tema][0]})")
        elif n > 0:
            senal_encuestas = f"{n} quejas sin tema dominante identificado."
        else:
            senal_encuestas = "Sin quejas de mala experiencia en el período."

        out = _parse_json(self.llm.complete(DIAGNOSE_PROMPT.format(
            senal_ventas=senal_ventas,
            senal_encuestas=senal_encuestas,
            veredicto=audit.verdict,
            justificacion=audit.justification,
            context=context,
        )))

        if not out or out.get("fundada") is False:
            motivo = ((out or {}).get("diagnostico")
                      or "El manual no permite fundamentar una acción para estas señales.")
            return {"abstuvo": True, "motivo": motivo}
        return {
            "abstuvo": False,
            "diagnostico": out.get("diagnostico", ""),
            "accion": out.get("accion", ""),
        }

    # --- API pública ---

    def diagnosticar(self, pdv: str,
                     periodo: Optional[Periodo] = None) -> Caso3Result:
        """Diagnóstico combinado (ventas + encuestas) para una sucursal y período."""
        periodo = (Periodo.coerce(periodo) if periodo is not None
                   else periodo_por_defecto(self.con))
        s = self.app.invoke({"pdv": pdv, "periodo": periodo})

        desvio = s.get("desvio")
        audit = s.get("audit")
        n_quejas = s.get("n_quejas", 0)

        if s.get("abstuvo", True):
            if desvio is None and n_quejas == 0:
                motivo = "Sin desvío de ventas ni quejas de clientes en el período."
            elif audit is None or not audit.passages:
                motivo = ("Se detectaron señales pero el manual no aporta "
                          "contexto normativo aplicable.")
            else:
                motivo = s.get("motivo", "Abstención.")
        else:
            motivo = ""

        return Caso3Result(
            pdv=pdv,
            periodo=periodo.etiqueta,
            kpis=s.get("kpis"),
            desvio=desvio,
            n_quejas=n_quejas,
            tema_quejas=s.get("tema_quejas"),
            diagnostico=s.get("diagnostico", ""),
            accion=s.get("accion", ""),
            citas=s.get("citas", []),
            abstuvo=s.get("abstuvo", True),
            motivo=motivo,
            verdict=(audit.verdict if audit else ""),
        )
