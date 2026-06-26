"""Caso 1 — Clasificador / router de encuestas (LangGraph).

Toma el texto libre de las encuestas de mala experiencia, clasifica cada queja en
un ÚNICO tema (taxonomía fija), agrega el volumen por tema (y por sucursal) y rutea
el tema DOMINANTE al manual operativo (NormativeAuditor) para producir una acción
correctiva citada — o se ABSTIENE si el tema no encuentra norma aplicable.

Grafo:

    cargar --> (¿hay quejas?) --no--> END   (sin datos, SIN LLM)
                   |
                  sí
                   v
              clasificar --> agregar --> (¿tema dominante auditable?) --no--> END
                                              |                         (solo conteos)
                                             sí
                                              v
                                          auditar --> (¿hay norma?) --no--> END
                                                            |          (conteos + sin acción)
                                                           sí
                                                            v
                                                       recomendar --> END

Eficiencia (igual criterio que Caso 2/5): `cargar` y `agregar` son SQL + reglas
(sin LLM). La clasificación se hace en LOTES (un puñado de llamadas, no una por
queja). La auditoría RAG + síntesis se invoca UNA sola vez, sobre el tema dominante.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TypedDict

from ..kpi import Periodo
from .auditor_rag import AuditResult, _parse_json

# --- Taxonomía (single-label). slug -> (descripción, consulta normativa al manual) ---
# El orden define el desempate determinístico ante frecuencias iguales.
TEMAS: dict[str, tuple[str, str]] = {
    "calidad_producto": (
        "sabor, textura o estado del helado (poco rico, mucha crema, congelado, feo)",
        "calidad y conservación del producto: sabor, textura y estado del helado servido"),
    "atencion_cliente": (
        "trato del personal: maltrato, mala atención, falta de amabilidad, deshonestidad",
        "atención al cliente y trato del personal en el salón"),
    "cantidad_peso": (
        "cantidad o peso entregado de menos: potes a medio llenar, porciones escasas",
        "control de peso y porciones servidas al cliente"),
    "fidelizacion_puntos": (
        "programa de puntos, descuentos, promociones o app de fidelización",
        "programa de fidelización: acumulación y canje de puntos y descuentos"),
    "limpieza_local": (
        "limpieza e higiene del local, mesas, baños o instalaciones",
        "higiene y limpieza del local y las instalaciones"),
    "precios": (
        "precios altos o aumentos de precio percibidos como excesivos",
        "política de precios y comunicación de aumentos al cliente"),
    "cadena_frio": (
        "helado derretido, blando o líquido — falla de cadena de frío",
        "mantenimiento de la cadena de frío y temperatura de conservación del helado"),
    "delivery_logistica": (
        "pedidos a domicilio: demoras, no entregados, marcados entregados sin llegar",
        "gestión de pedidos a domicilio y tiempos de entrega"),
    "otros": (
        "cualquier queja que no encaje en las categorías anteriores",
        ""),
}

CLASSIFY_PROMPT = """Sos un clasificador de quejas de clientes de una cadena de heladerías.
Clasificá CADA queja en UN ÚNICO tema de esta taxonomía (elegí el más representativo):
{taxonomia}

Reglas:
- Asigná exactamente un tema por queja (el dominante si toca varios).
- Usá los slugs EXACTOS de la lista (ej. "calidad_producto"). Si ninguno aplica, "otros".
Respondé SOLO JSON, un objeto que mapea el número de queja a su slug:
{{"1": "<slug>", "2": "<slug>", ...}}

QUEJAS:
{quejas}
"""

RECOMMEND_PROMPT = """Sos el supervisor de operaciones de una cadena de heladerías.
El tema MÁS frecuente en las quejas de clientes es "{tema}" ({descripcion}), con
{n} quejas ({pct}). El Auditor Normativo recuperó el CONTEXTO del manual operativo.
Redactá una recomendación accionable apoyada ÚNICAMENTE en ese contexto.

Reglas:
- Basate solo en el CONTEXTO normativo. No inventes normas ni cifras.
- La recomendación conecta el volumen de quejas con la práctica del manual.
- La acción es concreta, verificable y ejecutable por el encargado.
- Si el contexto no alcanza para fundamentar una acción, abstenete (fundada=false).
Respondé SOLO JSON:
{{"fundada": true|false, "recomendacion": "<resumen breve>", "accion": "<acción concreta>"}}

CONTEXTO NORMATIVO (pasajes del manual con su cita entre corchetes):
{context}
"""


class Caso1State(TypedDict, total=False):
    tabla: str                          # 'mala' | 'buena'
    periodo: Periodo | None
    limite: int | None
    quejas: list[tuple[str, str]]       # (numero_sucursal, texto)
    temas: list[str]                    # tema por queja (paralelo a `quejas`)
    conteo: dict[str, int]              # tema -> cantidad
    por_sucursal: dict[str, dict[str, int]]
    dominante: str
    audit: AuditResult
    recomendacion: str
    accion: str
    citas: list[str]
    abstuvo: bool
    motivo: str


@dataclass
class Caso1Result:
    """Salida del Caso 1: conteos por tema + acción citada para el tema dominante."""
    tabla: str
    periodo: str
    total: int = 0
    conteo: dict[str, int] = field(default_factory=dict)
    por_sucursal: dict[str, dict[str, int]] = field(default_factory=dict)
    dominante: str = ""
    recomendacion: str = ""
    accion: str = ""
    citas: list[str] = field(default_factory=list)
    abstuvo: bool = True
    motivo: str = ""
    verdict: str = ""

    def ranking(self) -> list[tuple[str, int]]:
        """Temas ordenados por frecuencia desc (desempate por orden de la taxonomía)."""
        orden = {t: i for i, t in enumerate(TEMAS)}
        return sorted(self.conteo.items(), key=lambda kv: (-kv[1], orden.get(kv[0], 99)))

    def __str__(self) -> str:
        cab = f"Encuestas '{self.tabla}' · {self.periodo} · {self.total} quejas clasificadas"
        if not self.total:
            return f"{cab}\n(no hay quejas para el período)"
        filas = []
        for tema, n in self.ranking():
            pct = f"{n / self.total:.0%}" if self.total else "—"
            filas.append(f"  {tema:22} {n:5}  {pct}")
        bloques = [cab, "DISTRIBUCIÓN POR TEMA:", *filas,
                   f"\nTEMA DOMINANTE: {self.dominante}"]
        if self.abstuvo:
            bloques.append(f"ABSTENCIÓN (acción): {self.motivo}")
        else:
            cites = "\n".join(f"  · {c}" for c in self.citas) or "  —"
            bloques += [f"RECOMENDACIÓN: {self.recomendacion}",
                        f"ACCIÓN: {self.accion}",
                        f"CITAS (manual operativo):\n{cites}"]
        return "\n".join(bloques)


class ClasificadorEncuestas:
    """Caso 1: clasifica encuestas por tema y rutea el dominante al manual.

    Args:
        con: conexión DuckDB (solo lectura) con las tablas de encuestas.
        auditor: ``NormativeAuditor`` (RAG + CRAG) para auditar el tema dominante.
        llm: modelo para clasificación y síntesis (por defecto, el del auditor).
        batch: cantidad de quejas por llamada de clasificación.
    """

    TABLAS = {"mala": "encuestas_mala_experiencia",
              "buena": "encuestas_buena_experiencia"}

    def __init__(self, con, auditor, llm=None, batch: int = 25):
        self.con = con
        self.auditor = auditor
        self.llm = llm or auditor.llm
        self.batch = max(1, batch)
        self.app = self._build()

    def _build(self):
        from langgraph.graph import StateGraph, START, END
        g = StateGraph(Caso1State)
        g.add_node("cargar", self._cargar)
        g.add_node("clasificar", self._clasificar)
        g.add_node("agregar", self._agregar)
        g.add_node("auditar", self._auditar)
        g.add_node("recomendar", self._recomendar)

        g.add_edge(START, "cargar")
        # Sin quejas -> END (sin gastar LLM).
        g.add_conditional_edges("cargar", self._hay_quejas,
                                {"clasificar": "clasificar", "fin": END})
        g.add_edge("clasificar", "agregar")
        # 'otros' no tiene norma que auditar -> termina con solo los conteos.
        g.add_conditional_edges("agregar", self._auditable,
                                {"auditar": "auditar", "fin": END})
        # Sin contexto del manual -> conteos sin acción.
        g.add_conditional_edges("auditar", self._hay_contexto,
                                {"recomendar": "recomendar", "fin": END})
        g.add_edge("recomendar", END)
        return g.compile()

    # --- nodos sin LLM ---
    def _cargar(self, state: Caso1State) -> dict:
        tabla = self.TABLAS[state["tabla"]]
        sql = (f'SELECT numero, origen_respuesta_texto FROM "{tabla}" '
               "WHERE origen_respuesta_texto IS NOT NULL "
               "AND length(trim(origen_respuesta_texto)) > 3")
        params: list = []
        periodo = state.get("periodo")
        if periodo is not None:
            sql += " AND CAST(origen_fhrespuesta AS DATE) BETWEEN ? AND ?"
            params += [periodo.desde, periodo.hasta]
        sql += " ORDER BY origen_fhrespuesta DESC"
        limite = state.get("limite")
        if limite:
            sql += f" LIMIT {int(limite)}"
        filas = self.con.execute(sql, params).fetchall()
        return {"quejas": [(num, txt.strip()) for num, txt in filas]}

    def _agregar(self, state: Caso1State) -> dict:
        temas = state["temas"]
        quejas = state["quejas"]
        conteo = Counter(temas)
        por_suc: dict[str, Counter] = defaultdict(Counter)
        for (num, _), tema in zip(quejas, temas):
            por_suc[num][tema] += 1
        # Dominante: mayor frecuencia; desempate por orden de la taxonomía.
        orden = {t: i for i, t in enumerate(TEMAS)}
        dominante = min(conteo, key=lambda t: (-conteo[t], orden.get(t, 99)))
        return {"conteo": dict(conteo),
                "por_sucursal": {k: dict(v) for k, v in por_suc.items()},
                "dominante": dominante}

    # --- ruteo ---
    def _hay_quejas(self, state: Caso1State) -> str:
        return "clasificar" if state.get("quejas") else "fin"

    def _auditable(self, state: Caso1State) -> str:
        # 'otros' es un cajón de sastre sin norma específica: no se audita.
        return "fin" if state.get("dominante") == "otros" else "auditar"

    def _hay_contexto(self, state: Caso1State) -> str:
        audit = state.get("audit")
        return "recomendar" if (audit is not None and audit.passages) else "fin"

    # --- clasificación (LLM, en lotes) ---
    def _clasificar(self, state: Caso1State) -> dict:
        quejas = state["quejas"]
        taxonomia = "\n".join(f"- {slug}: {desc}" for slug, (desc, _) in TEMAS.items())
        validos = set(TEMAS)
        temas: list[str] = []
        for ini in range(0, len(quejas), self.batch):
            lote = quejas[ini:ini + self.batch]
            listado = "\n".join(f"{i}. {txt}" for i, (_, txt) in enumerate(lote, 1))
            out = _parse_json(self.llm.complete(
                CLASSIFY_PROMPT.format(taxonomia=taxonomia, quejas=listado)))
            for i in range(1, len(lote) + 1):
                tema = (out.get(str(i)) or "").strip()
                temas.append(tema if tema in validos else "otros")
        return {"temas": temas}

    # --- auditoría + síntesis (LLM, una vez sobre el dominante) ---
    def _auditar(self, state: Caso1State) -> dict:
        dominante = state["dominante"]
        consulta = TEMAS[dominante][1]
        claim = (f"Quejas recurrentes de clientes sobre {consulta}. "
                 "¿Qué establece el manual operativo al respecto?")
        res = self.auditor.audit(claim)
        return {"audit": res, "citas": [p.citation for p in res.passages]}

    def _recomendar(self, state: Caso1State) -> dict:
        dominante = state["dominante"]
        audit: AuditResult = state["audit"]
        n = state["conteo"][dominante]
        total = len(state["quejas"])
        context = self.auditor.retriever.format_context(audit.passages)
        out = _parse_json(self.llm.complete(RECOMMEND_PROMPT.format(
            tema=dominante, descripcion=TEMAS[dominante][0], n=n,
            pct=f"{n / total:.0%}" if total else "—", context=context)))
        if not out or out.get("fundada") is False:
            motivo = ((out or {}).get("recomendacion")
                      or "El manual no permite fundamentar una acción para este tema.")
            return {"abstuvo": True, "motivo": motivo}
        return {"abstuvo": False,
                "recomendacion": out.get("recomendacion", ""),
                "accion": out.get("accion", "")}

    # --- API pública ---
    def clasificar(self, tabla: str = "mala",
                   periodo: Periodo | str | None = None,
                   limite: int | None = None) -> Caso1Result:
        """Clasifica las encuestas de una tabla y rutea el tema dominante al manual.

        Args:
            tabla: 'mala' (insatisfacción) o 'buena' (satisfacción).
            periodo: ``Periodo``, 'YYYY-MM' o None (sin filtro temporal).
            limite: tope de quejas a clasificar (las más recientes), para acotar costo.
        """
        if tabla not in self.TABLAS:
            raise ValueError(f"tabla inválida: {tabla!r} (usá 'mala' o 'buena').")
        per = Periodo.coerce(periodo) if periodo is not None else None
        s = self.app.invoke({"tabla": tabla, "periodo": per, "limite": limite})

        total = len(s.get("quejas", []))
        dominante = s.get("dominante", "")
        audit = s.get("audit")
        abstuvo = s.get("abstuvo", True)
        if abstuvo and not total:
            motivo = "No hay quejas para clasificar en el período."
        elif abstuvo and dominante == "otros":
            motivo = "El tema dominante es 'otros'; no hay una norma específica que auditar."
        elif abstuvo and (audit is None or not audit.passages):
            motivo = f"No se recuperó contexto del manual para el tema '{dominante}'."
        else:
            motivo = s.get("motivo", "") if abstuvo else ""

        return Caso1Result(
            tabla=tabla, periodo=(per.etiqueta if per else "todo el histórico"),
            total=total, conteo=s.get("conteo", {}),
            por_sucursal=s.get("por_sucursal", {}), dominante=dominante,
            recomendacion=s.get("recomendacion", ""), accion=s.get("accion", ""),
            citas=s.get("citas", []), abstuvo=abstuvo, motivo=motivo,
            verdict=(audit.verdict if audit else ""))
