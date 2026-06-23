"""Chequeos livianos de comportamiento del sistema (spot checks del MVP).

No es un harness formal (Hit@k/MRR/nDCG viven en metrics.py). Verifica las
propiedades clave del Caso 2 y Caso 5:

  - Sin LLM: el ranking ordena todas las sucursales; la abstención (sin desvío)
    no invoca al LLM.
  - Con LLM (--with-llm): Caso 2 cita cuando actúa; la consulta normativa cita
    cuando el manual cubre y se abstiene cuando no; Caso 5 produce reporte.

Cada chequeo devuelve ``Check(nombre, ok, detalle)``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..kpi import compute_kpis, detectar_desvio, latest_month, list_pdvs


@dataclass
class Check:
    nombre: str
    ok: bool
    detalle: str = ""


# --- chequeos sin LLM (determinísticos, rápidos) ---

def check_ranking_pdvs(con, mes: str) -> Check:
    """El ranking de severidad cubre todas las sucursales y queda ordenado."""
    pdvs = list_pdvs(con)
    pares = [(p, detectar_desvio(compute_kpis(con, p, mes))) for p in pdvs]
    ordenados = sorted(pares, key=lambda t: t[1].orden if t[1] else (99, 0.0))
    claves = [t[1].orden if t[1] else (99, 0.0) for t in ordenados]
    monotono = all(claves[i] <= claves[i + 1] for i in range(len(claves) - 1))
    n = len(pdvs)
    ok = n >= 1 and monotono
    return Check("ranking_pdvs", ok,
                 f"{n} sucursales, orden {'monótono' if monotono else 'INCONSISTENTE'} "
                 f"por (prioridad, -severidad)")


def check_abstencion_sin_llm(con, mes: str) -> Check:
    """Una sucursal sin desvío se abstiene SIN invocar al LLM (tripwire)."""
    from ..agents.caso2 import DiagnosticadorPDV

    # Buscar una sucursal sin desvío en el mes; si no hay, el chequeo no aplica.
    sin_desvio = next((p for p in list_pdvs(con)
                       if detectar_desvio(compute_kpis(con, p, mes)) is None), None)
    if sin_desvio is None:
        return Check("abstencion_sin_llm", True, "no aplica (todas con desvío este mes)")

    class Tripwire:
        llm = None
        retriever = None
        def audit(self, claim):  # noqa: D401
            raise AssertionError("LLM invocado en abstención")
        def complete(self, prompt):
            raise AssertionError("LLM invocado en abstención")

    tw = Tripwire(); tw.llm = tw
    try:
        r = DiagnosticadorPDV(con, tw, tw).diagnosticar(sin_desvio, mes)
        ok = r.abstuvo
        return Check("abstencion_sin_llm", ok,
                     f"{sin_desvio[:8]} abstuvo sin tocar el LLM")
    except AssertionError as e:
        return Check("abstencion_sin_llm", False, str(e))


# --- chequeos con LLM (requieren red; opcionales) ---

def check_caso2_cita_cuando_actua(diag, con, mes: str) -> Check:
    """Si el Caso 2 emite diagnóstico (no se abstiene), debe traer citas."""
    # Tomar la sucursal con mayor severidad (la más propensa a accionar).
    pares = [(p, detectar_desvio(compute_kpis(con, p, mes))) for p in list_pdvs(con)]
    pares = [t for t in pares if t[1] is not None]
    if not pares:
        return Check("caso2_cita_cuando_actua", True, "no aplica (sin desvíos)")
    pdv = min(pares, key=lambda t: t[1].orden)[0]
    r = diag.diagnosticar(pdv, mes)
    if r.abstuvo:
        ok = bool(r.motivo)
        return Check("caso2_cita_cuando_actua", ok,
                     f"{pdv[:8]} se abstuvo con motivo (válido): {r.motivo[:60]}")
    ok = len(r.citas) > 0
    return Check("caso2_cita_cuando_actua", ok,
                 f"{pdv[:8]} diagnosticó con {len(r.citas)} cita(s)")


def check_norma_cita_y_abstiene(retriever, llm) -> Check:
    """La consulta normativa cita cuando el manual cubre y se abstiene cuando no."""
    from ..rag.qa import consultar_norma

    dentro = consultar_norma(retriever, llm,
                             "¿Qué dice el manual sobre la cadena de frío y la "
                             "conservación del helado?")
    fuera = consultar_norma(retriever, llm,
                            "¿Cuál es la capital de Francia y su población?")
    ok_dentro = (not dentro["abstuvo"]) and len(dentro["citas"]) > 0
    ok_fuera = fuera["abstuvo"]
    ok = ok_dentro and ok_fuera
    return Check("norma_cita_y_abstiene", ok,
                 f"en-manual: {'cita' if ok_dentro else 'FALLA'} "
                 f"({len(dentro['citas'])} citas) | "
                 f"fuera-manual: {'abstiene' if ok_fuera else 'FALLA'}")


def check_caso5_reporte(plan, mes: str) -> Check:
    """Caso 5 produce un reporte con todas las sucursales en el ranking."""
    res = plan.generar(mes)
    n = len(res.resultados)
    tiene_tabla = "Ranking de sucursales" in res.reporte
    ok = n == len(list_pdvs(plan.con)) and tiene_tabla and len(res.reporte) > 0
    return Check("caso5_reporte", ok,
                 f"{n} sucursales en el plan, reporte {'con' if tiene_tabla else 'SIN'} "
                 f"ranking ({len(res.reporte)} chars)")


def run_checks(con, mes: str | None = None, with_llm: bool = False,
               build_llm=None, build_retriever=None) -> list[Check]:
    """Corre los chequeos. Sin ``with_llm`` solo los determinísticos (sin red)."""
    mes = mes or latest_month(con, complete_only=True)
    checks = [check_ranking_pdvs(con, mes), check_abstencion_sin_llm(con, mes)]
    if with_llm:
        from ..agents.auditor_rag import NormativeAuditor
        from ..agents.caso2 import DiagnosticadorPDV
        from ..agents.caso5 import PlanificadorMensual

        llm = build_llm()
        retriever = build_retriever()
        auditor = NormativeAuditor(retriever, llm)
        diag = DiagnosticadorPDV(con, auditor, llm)
        plan = PlanificadorMensual(con, auditor, llm)
        checks += [
            check_caso2_cita_cuando_actua(diag, con, mes),
            check_norma_cita_y_abstiene(retriever, llm),
            check_caso5_reporte(plan, mes),
        ]
    return checks
