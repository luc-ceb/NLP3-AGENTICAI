"""Front Streamlit del Gemelo Digital Operativo (demo).

Tabs:
  1. Diagnóstico operativo  — pregunta libre (Text-to-SQL + RAG)
  2. Encuestas por sucursal — Caso 1 filtrado por PDV
  3. Diagnóstico combinado  — Caso 3 (ventas + encuestas por PDV)
  4. Consultar el manual    — RAG puro sobre el manual operativo

Levantar:  streamlit run src/interface/app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv          # noqa: E402
load_dotenv(ROOT / ".env", override=True)

from src.interface.api import build_supervisor  # noqa: E402
from src.actions.notifier import LogNotifier    # noqa: E402

st.set_page_config(page_title="Gemelo Digital Operativo", page_icon="🍦", layout="centered")

EJEMPLOS = [
    "¿Cuántas quejas de mala experiencia mencionan que el helado estaba derretido o blando?",
    "¿Cuáles son los 5 productos más vendidos por cantidad?",
    "¿Qué día de la semana concentra más ventas?",
    "¿Qué sucursal concentra más quejas de helado derretido?",
]


# ── recursos compartidos ──────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelos e índices (solo la primera vez)…")
def get_engine():
    return build_supervisor(), LogNotifier()


@st.cache_resource(show_spinner="Cargando agentes de encuestas y diagnóstico combinado…")
def get_agentes():
    import duckdb
    from src.rag.retrieve import HybridRetriever
    from src.agents.auditor_rag import NormativeAuditor
    from src.agents.caso1 import ClasificadorEncuestas
    from src.agents.caso3 import DiagnosticadorCombinado
    from src.llm.clients import make_llm_with_fallback

    con = duckdb.connect(str(ROOT / "data" / "gdo.duckdb"), read_only=True)
    llm = make_llm_with_fallback(
        provider=os.getenv("FAST_PROVIDER", "groq"),
        fallback=os.getenv("FAST_FALLBACK_PROVIDER", "anthropic"),
        model=os.getenv("FAST_MODEL"),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
    )
    retriever = HybridRetriever.build_default(ROOT / "data" / "index")
    auditor = NormativeAuditor(retriever, llm)
    clf = ClasificadorEncuestas(con, auditor, llm)
    diag = DiagnosticadorCombinado(con, auditor, llm)
    return con, clf, diag


@st.cache_data(show_spinner=False)
def get_pdvs() -> list[tuple[str, str]]:
    """Devuelve lista de (branchofficeid, nombre_sucursal) ordenada por nombre."""
    import duckdb
    con = duckdb.connect(str(ROOT / "data" / "gdo.duckdb"), read_only=True)
    rows = con.execute(
        "SELECT DISTINCT branchofficeid, sucursal FROM datos_ventas ORDER BY sucursal"
    ).fetchall()
    con.close()
    return rows


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Gemelo Digital Operativo")
    st.caption("Contrasta los datos reales de las sucursales contra la norma del "
               "manual operativo y los recursos, con citas direccionables.")
    st.divider()
    st.caption("Flujo: Analista (SQL) → Auditor (RAG + CRAG) → Supervisor (reconcilia).")

st.title("🍦 Gemelo Digital Operativo")

tab_diag, tab_enc, tab_comb, tab_manual = st.tabs([
    "Diagnóstico operativo",
    "Encuestas por sucursal",
    "Diagnóstico combinado",
    "Consultar el manual",
])


# ── Tab 1: Diagnóstico operativo (pregunta libre) ─────────────────────────────

with tab_diag:
    st.caption("Realidad (datos) vs norma (manual) — con trazabilidad de fuentes.")
    ejemplo = st.selectbox("Ejemplos (opcional)", [""] + EJEMPLOS)
    pregunta = st.text_area("Pregunta operativa", value=ejemplo, height=80,
                            placeholder="Escribí una pregunta sobre la operación…")

    if st.button("Diagnosticar", type="primary", disabled=not pregunta.strip()):
        supervisor, notifier = get_engine()
        with st.spinner("Analizando datos y contrastando con la norma…"):
            res = supervisor.diagnose(pregunta.strip())
            notifier.send(res)

        st.subheader("Diagnóstico")
        st.markdown(res.diagnosis)

        if res.audits:
            st.subheader("Hallazgos")
            for a in res.audits:
                v = a.get("verdict", "")
                linea = f"**{a.get('claim','')}** — {a.get('justification','')}"
                if v == "no_cumple":
                    st.error(f"❌ NO CUMPLE · {linea}")
                elif v == "cumple":
                    st.success(f"✅ CUMPLE · {linea}")
                else:
                    st.info(f"➖ SIN NORMA · {linea}")

        if res.citations:
            st.subheader("Fuentes citadas")
            for c in res.citations:
                st.markdown(f"- `{c}`")

        with st.expander("Detalle técnico (SQL y datos)"):
            st.code(res.sql or "—", language="sql")
            st.text(res.facts or "—")

        st.caption("✓ Diagnóstico registrado automáticamente en el log.")


# ── Tab 2: Encuestas por sucursal (Caso 1 con PDV) ───────────────────────────

with tab_enc:
    st.caption("Clasifica las encuestas de una sucursal por tema (LLM) "
               "y audita el dominante contra el manual operativo.")

    pdvs = get_pdvs()
    pdv_labels = [f"{nombre}  ({guid[:8]})" for guid, nombre in pdvs]
    pdv_map = {f"{nombre}  ({guid[:8]})": guid for guid, nombre in pdvs}

    col1, col2, col3 = st.columns([3, 2, 2])
    with col1:
        sel_enc = st.selectbox("Sucursal", ["— elegí una sucursal —"] + pdv_labels,
                               key="enc_pdv")
    with col2:
        tipo_enc = st.radio("Tipo de encuesta", ["Mala experiencia", "Buena experiencia"],
                            key="enc_tipo", horizontal=True)
    with col3:
        mes_enc = st.text_input("Mes (YYYY-MM)", placeholder="ej. 2026-04", key="enc_mes")

    tabla_enc = "mala" if "Mala" in tipo_enc else "buena"
    pdv_enc = pdv_map.get(sel_enc)

    if st.button("Clasificar encuestas", type="primary",
                 disabled=pdv_enc is None, key="btn_enc"):
        _, clf, _ = get_agentes()
        periodo_enc = mes_enc.strip() if mes_enc.strip() else None
        with st.spinner("Clasificando encuestas con LLM…"):
            r = clf.clasificar(tabla=tabla_enc, periodo=periodo_enc, pdv=pdv_enc)

        if not r.total:
            st.info(f"No hay encuestas de '{tabla_enc}' para esta sucursal "
                    f"en el período seleccionado.")
        else:
            st.subheader(f"Distribución por tema — {r.total} encuestas")
            ranking = r.ranking()
            for tema, n in ranking:
                pct = n / r.total
                st.progress(pct, text=f"**{tema}** — {n} ({pct:.0%})")

            st.divider()
            st.markdown(f"**Tema dominante:** `{r.dominante}`")

            if r.abstuvo:
                st.warning(f"Abstención: {r.motivo}")
            else:
                st.subheader("Recomendación normativa")
                st.markdown(r.recomendacion)
                st.markdown(f"**Acción:** {r.accion}")
                if r.citas:
                    st.subheader("Fuentes citadas")
                    for c in r.citas:
                        st.markdown(f"- `{c}`")


# ── Tab 3: Diagnóstico combinado (Caso 3) ────────────────────────────────────

with tab_comb:
    st.caption("Cruza KPIs de ventas con quejas de clientes para una sucursal "
               "y produce un diagnóstico integrado contrastado contra el manual.")

    col1, col2 = st.columns([3, 2])
    with col1:
        sel_comb = st.selectbox("Sucursal", ["— elegí una sucursal —"] + pdv_labels,
                                key="comb_pdv")
    with col2:
        mes_comb = st.text_input("Mes (YYYY-MM)", placeholder="ej. 2026-04",
                                 key="comb_mes",
                                 help="Vacío = último mes completo disponible")

    pdv_comb = pdv_map.get(sel_comb)

    if st.button("Diagnosticar", type="primary",
                 disabled=pdv_comb is None, key="btn_comb"):
        _, _, diag_comb = get_agentes()
        periodo_comb = mes_comb.strip() if mes_comb.strip() else None
        with st.spinner("Analizando ventas y encuestas…"):
            r = diag_comb.diagnosticar(pdv_comb, periodo_comb)

        # Señales en dos columnas
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Ventas**")
            if r.desvio:
                st.error(r.desvio.detalle)
            else:
                st.success("Sin desvío relevante.")
        with c2:
            st.markdown("**Encuestas de mala experiencia**")
            if r.n_quejas:
                tema_label = f" · tema: `{r.tema_quejas}`" if r.tema_quejas else ""
                st.warning(f"{r.n_quejas} quejas{tema_label}")
            else:
                st.success("Sin quejas en el período.")

        st.divider()

        if r.abstuvo:
            st.info(f"**Abstención:** {r.motivo}")
        else:
            st.subheader("Diagnóstico integrado")
            st.markdown(r.diagnostico)
            st.markdown(f"**Acción:** {r.accion}")

            v = r.verdict
            if v == "no_cumple":
                st.error("❌ Incumplimiento normativo detectado")
            elif v == "cumple":
                st.success("✅ Opera dentro de la norma")
            elif v:
                st.info(f"➖ {v}")

            if r.citas:
                st.subheader("Fuentes citadas")
                for c in r.citas:
                    st.markdown(f"- `{c}`")


# ── Tab 4: Consultar el manual ────────────────────────────────────────────────

with tab_manual:
    st.caption("Consulta directa al manual operativo (RAG, sin SQL). Responde solo "
               "con lo que figura en el manual y cita la fuente; si no figura, lo aclara.")
    preg_manual = st.text_area(
        "¿Qué querés consultar del manual?", height=80,
        placeholder="Ej.: ¿cuál es el procedimiento de recepción de mercadería congelada?")

    if st.button("Consultar manual", disabled=not preg_manual.strip()):
        from src.rag.qa import consultar_norma
        supervisor, _ = get_engine()
        with st.spinner("Buscando en el manual…"):
            r = consultar_norma(supervisor.retriever, supervisor.llm, preg_manual.strip())

        st.subheader("Respuesta")
        st.markdown(r["respuesta"])
        if r["abstuvo"]:
            st.info("El manual no parece cubrir esta consulta.")
        if r["citas"]:
            st.subheader("Fuentes citadas")
            for c in r["citas"]:
                st.markdown(f"- `{c}`")
