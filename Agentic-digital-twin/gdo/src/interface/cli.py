"""CLI del Gemelo Digital Operativo (GDO).

Comandos:
    diagnosticar <PDV> [--mes YYYY-MM | --desde D --hasta H] [--json]
    plan-mensual [--mes YYYY-MM | --desde D --hasta H] [--out f] [--json]
    eval [--mes YYYY-MM] [--with-llm]             chequeos livianos de comportamiento

El período de análisis es un mes (--mes YYYY-MM) o una ventana de fechas arbitraria
(--desde/--hasta en YYYY-MM-DD); en ambos casos se compara contra el mismo período
del año anterior. Por defecto, el último mes completo disponible.

El PDV admite el GUID completo o un prefijo único (p.ej. los primeros 8 chars).
LLM: Groq por defecto con respaldo automático a Anthropic ante rate-limit.
Recuperación densa: FAISS local (VECTOR_BACKEND=faiss por defecto en la CLI).

Uso:
    python -m src.interface.cli plan-mensual
    python -m src.interface.cli diagnosticar A9D75316 --mes 2026-04
    python -m src.interface.cli diagnosticar A9D75316 --desde 2026-01-01 --hasta 2026-05-31
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DB = ROOT / "data" / "gdo.duckdb"
INDEX = ROOT / "data" / "index"


def _connect():
    import duckdb
    if not DB.exists():
        raise SystemExit(f"No existe la base {DB}. Cargá los datos primero (ETL).")
    return duckdb.connect(str(DB), read_only=True)


def _resolve_pdv(con, pdv: str) -> str:
    """Resuelve un GUID completo o un prefijo único contra las sucursales."""
    from ..kpi import list_pdvs
    pdvs = list_pdvs(con)
    if pdv in pdvs:
        return pdv
    matches = [p for p in pdvs if p.lower().startswith(pdv.lower())]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        opciones = "\n".join(f"  - {p[:8]}  ({p})" for p in pdvs)
        raise SystemExit(f"PDV '{pdv}' no encontrado. Sucursales disponibles:\n{opciones}")
    amb = ", ".join(p[:8] for p in matches)
    raise SystemExit(f"Prefijo '{pdv}' es ambiguo: coincide con {amb}. Usá más caracteres.")


def _periodo_from_args(args):
    """Construye el Periodo desde --mes o --desde/--hasta (o None → por defecto)."""
    from ..kpi import Periodo
    desde = getattr(args, "desde", None)
    hasta = getattr(args, "hasta", None)
    mes = getattr(args, "mes", None)
    if desde or hasta:
        if not (desde and hasta):
            raise SystemExit("Indicá --desde y --hasta juntos (YYYY-MM-DD).")
        if mes:
            raise SystemExit("Usá --mes O --desde/--hasta, no ambos.")
        try:
            d0, d1 = date.fromisoformat(desde), date.fromisoformat(hasta)
        except ValueError as e:
            raise SystemExit(f"Fecha inválida (usá YYYY-MM-DD): {e}")
        try:
            return Periodo.de_rango(d0, d1)
        except ValueError as e:
            raise SystemExit(str(e))
    if mes:
        try:
            return Periodo.de_mes(mes)
        except (ValueError, IndexError):
            raise SystemExit(f"Mes inválido '{mes}' (usá YYYY-MM).")
    return None


def _build_llm():
    """LLM primario (Groq) con respaldo automático (Anthropic) ante rate-limit."""
    from ..llm.clients import make_llm_with_fallback
    # max_tokens holgado: la síntesis y los diagnósticos producen JSON extensos.
    return make_llm_with_fallback(max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")))


def _build_retriever():
    from ..rag.retrieve import HybridRetriever
    backend = os.getenv("VECTOR_BACKEND", "faiss")   # CLI prioriza FAISS local
    return HybridRetriever.build_default(INDEX, vector_backend=backend)


def _build_auditor(retriever, llm):
    from ..agents.auditor_rag import NormativeAuditor
    return NormativeAuditor(retriever, llm)


# --- comandos ---

def cmd_diagnosticar(args) -> int:
    from ..agents.caso2 import DiagnosticadorPDV
    con = _connect()
    pdv = _resolve_pdv(con, args.pdv)
    llm = _build_llm()
    diag = DiagnosticadorPDV(con, _build_auditor(_build_retriever(), llm), llm)
    r = diag.diagnosticar(pdv, _periodo_from_args(args))

    if args.json:
        print(json.dumps({
            "pdv": r.pdv, "periodo": r.periodo, "abstuvo": r.abstuvo, "motivo": r.motivo,
            "desvio": (r.desvio.detalle if r.desvio else None),
            "severidad": r.severidad, "verdict": r.verdict,
            "diagnostico": r.diagnostico, "accion": r.accion, "citas": r.citas,
        }, ensure_ascii=False, indent=2))
    else:
        print(r)
    return 0


def cmd_plan_mensual(args) -> int:
    from ..agents.caso5 import PlanificadorMensual
    con = _connect()
    llm = _build_llm()
    plan = PlanificadorMensual(con, _build_auditor(_build_retriever(), llm), llm)
    res = plan.generar(_periodo_from_args(args))

    if args.json:
        out = {"periodo": res.periodo, "ranking": [
            {"pdv": r.pdv, "desvio": (r.desvio.detalle if r.desvio else None),
             "severidad": r.severidad, "abstuvo": r.abstuvo,
             "diagnostico": r.diagnostico, "accion": r.accion, "citas": r.citas}
            for r in res.resultados]}
        texto = json.dumps(out, ensure_ascii=False, indent=2)
    else:
        texto = res.reporte

    if args.out:
        Path(args.out).write_text(texto, encoding="utf-8")
        print(f"Reporte escrito en {args.out}")
    else:
        print(texto)
    return 0


def cmd_eval(args) -> int:
    from ..eval.checks import run_checks
    con = _connect()
    checks = run_checks(con, mes=args.mes, with_llm=args.with_llm,
                        build_llm=_build_llm, build_retriever=_build_retriever)
    print(f"Chequeos GDO ({'con LLM' if args.with_llm else 'sin LLM'}):\n")
    todos_ok = True
    for c in checks:
        todos_ok = todos_ok and c.ok
        print(f"  [{'PASS' if c.ok else 'FAIL'}] {c.nombre}: {c.detalle}")
    print(f"\n{'Todos los chequeos pasaron.' if todos_ok else 'Hay chequeos en FALLA.'}")
    return 0 if todos_ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gdo", description="Gemelo Digital Operativo (GDO) — CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnosticar", help="Caso 2: diagnóstico de un PDV")
    d.add_argument("pdv", help="GUID de la sucursal o un prefijo único (ej. A9D75316)")
    d.add_argument("--mes", help="Mes 'YYYY-MM' (por defecto, último mes completo)")
    d.add_argument("--desde", help="Inicio de la ventana 'YYYY-MM-DD' (con --hasta)")
    d.add_argument("--hasta", help="Fin de la ventana 'YYYY-MM-DD' (con --desde)")
    d.add_argument("--json", action="store_true", help="Salida en JSON")
    d.set_defaults(func=cmd_diagnosticar)

    m = sub.add_parser("plan-mensual", help="Caso 5: plan de diagnóstico rankeado (Markdown)")
    m.add_argument("--mes", help="Mes 'YYYY-MM' (por defecto, último mes completo)")
    m.add_argument("--desde", help="Inicio de la ventana 'YYYY-MM-DD' (con --hasta)")
    m.add_argument("--hasta", help="Fin de la ventana 'YYYY-MM-DD' (con --desde)")
    m.add_argument("--out", help="Archivo de salida (por defecto, stdout)")
    m.add_argument("--json", action="store_true", help="Salida en JSON")
    m.set_defaults(func=cmd_plan_mensual)

    e = sub.add_parser("eval", help="Chequeos livianos de comportamiento")
    e.add_argument("--mes", help="Mes 'YYYY-MM' (por defecto, último mes completo)")
    e.add_argument("--with-llm", action="store_true",
                   help="Incluye chequeos que invocan al LLM (requiere red)")
    e.set_defaults(func=cmd_eval)
    return p


def main(argv: list[str] | None = None) -> int:
    import logging
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"),
                        format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
