"""Front Streamlit del Gemelo Digital Operativo (demo).

Reutiliza la misma construcción que la API (build_supervisor) y muestra el
diagnóstico de forma presentable. Dispara también la acción de log.

Levantar:  streamlit run src/interface/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.interface.api import build_supervisor  # noqa: E402
from src.actions.notifier import LogNotifier    # noqa: E402

st.set_page_config(page_title="Gemelo Digital Operativo", page_icon="🍦", layout="centered")

EJEMPLOS = [
    "¿Cuántas quejas de mala experiencia mencionan que el helado estaba derretido o blando?",
    "¿Cuáles son los 5 productos más vendidos por cantidad?",
    "¿Qué día de la semana concentra más ventas?",
    "¿Qué sucursal concentra más quejas de helado derretido?",
]


@st.cache_resource(show_spinner="Cargando modelos e índices (solo la primera vez)…")
def get_engine():
    return build_supervisor(), LogNotifier()


with st.sidebar:
    st.header("Gemelo Digital Operativo")
    st.caption("Contrasta los datos reales de las sucursales contra la norma del "
               "manual operativo y los recursos, con citas direccionables.")
    st.divider()
    st.caption("Flujo: Analista (SQL) → Auditor (RAG + CRAG) → Supervisor (reconcilia).")

st.title("🍦 Gemelo Digital Operativo")

tab_diag, tab_manual = st.tabs(["Diagnóstico operativo", "Consultar el manual"])

with tab_diag:
    st.caption("Realidad (datos) vs norma (manual) — con trazabilidad de fuentes.")
    ejemplo = st.selectbox("Ejemplos (opcional)", [""] + EJEMPLOS)
    pregunta = st.text_area("Pregunta operativa", value=ejemplo, height=80,
                            placeholder="Escribí una pregunta sobre la operación…")

    if st.button("Diagnosticar", type="primary", disabled=not pregunta.strip()):
        supervisor, notifier = get_engine()
        with st.spinner("Analizando datos y contrastando con la norma…"):
            res = supervisor.diagnose(pregunta.strip())
            notifier.send(res)  # acción automática

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