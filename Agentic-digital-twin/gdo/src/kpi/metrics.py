"""Cálculo determinístico de KPIs por (sucursal, mes) sobre DuckDB.

Todas las consultas son SELECT parametrizadas (sin Text-to-SQL). Las métricas se
calculan para el mes objetivo y se comparan contra (a) el mismo mes del año
anterior (interanual / YoY) y (b) la media de la red en el mismo mes. La
comparación interanual cancela la estacionalidad (la demanda de helado es
fuertemente estacional), por lo que es más informativa que el mes anterior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb

TABLE = "datos_ventas"


def _same_month_last_year(mes: str) -> str:
    """'YYYY-MM' -> mismo mes del año anterior ('YYYY-MM')."""
    y, m = (int(x) for x in mes.split("-"))
    return f"{y - 1:04d}-{m:02d}"


def _pct(curr: float, base: float) -> float | None:
    """Variación porcentual (curr vs base). None si la base es 0/None."""
    if base in (None, 0) or curr is None:
        return None
    return round((curr - base) / base * 100, 1)


def list_pdvs(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Sucursales (branchofficeid) presentes en ventas, orden estable."""
    return [r[0] for r in con.execute(
        f"SELECT DISTINCT branchofficeid FROM {TABLE} ORDER BY branchofficeid"
    ).fetchall()]


def latest_month(con: duckdb.DuckDBPyConnection, complete_only: bool = False) -> str:
    """Último mes con datos ('YYYY-MM').

    Con ``complete_only`` descarta el mes más reciente si está incompleto (su
    último día es anterior al fin de mes), útil para no comparar meses parciales.
    """
    months = [r[0] for r in con.execute(
        f"SELECT DISTINCT strftime(fecha, '%Y-%m') m FROM {TABLE} ORDER BY m"
    ).fetchall()]
    if not months:
        raise ValueError(f"{TABLE} no tiene datos de fecha")
    if not complete_only:
        return months[-1]
    last = months[-1]
    max_day, = con.execute(
        f"SELECT MAX(fecha) FROM {TABLE} WHERE strftime(fecha,'%Y-%m') = ?", [last]
    ).fetchone()
    y, m = (int(x) for x in last.split("-"))
    eom = (date(y + (m == 12), (m % 12) + 1, 1)).toordinal() - 1
    if max_day is not None and date.fromordinal(eom) > max_day.date():
        return months[-2] if len(months) > 1 else last
    return last


@dataclass
class ProductoKpi:
    productid: int
    descripcion: str
    kilos: float
    unidades: int
    kilos_yoy: float = 0.0
    var_yoy: float | None = None  # variación % de kilos vs el mismo mes del año anterior


@dataclass
class PDVKpis:
    """Snapshot de KPIs de una sucursal en un mes, con comparativas."""
    pdv: str
    mes: str
    mes_yoy: str               # mismo mes del año anterior ('YYYY-MM')

    # Volumen y tráfico (mes objetivo)
    kilos: float = 0.0
    unidades: int = 0
    visitas: int = 0           # proxy de ticket = distinct (customerid, fecha)
    clientes: int = 0          # clientes únicos = distinct customerid
    kilos_por_visita: float = 0.0
    unidades_por_visita: float = 0.0

    # Comparativa interanual (YoY): mismo mes del año anterior. Cancela la
    # estacionalidad, por lo que un descenso aquí es señal de un problema real.
    kilos_yoy: float = 0.0
    visitas_yoy: int = 0
    clientes_yoy: int = 0
    kilos_var_yoy: float | None = None
    visitas_var_yoy: float | None = None
    clientes_var_yoy: float | None = None
    kilos_por_visita_yoy: float = 0.0
    kilos_por_visita_var_yoy: float | None = None

    # Comparativa contra la red (media de sucursales en el mismo mes).
    # Aísla el desempeño propio del efecto estacional que afecta a toda la red.
    kilos_red_media: float = 0.0
    kilos_var_red: float | None = None
    visitas_red_media: float = 0.0
    visitas_var_red: float | None = None
    clientes_red_media: float = 0.0
    clientes_var_red: float | None = None

    # Mix de producto
    top_productos: list[ProductoKpi] = field(default_factory=list)
    caidas_productos: list[ProductoKpi] = field(default_factory=list)

    def resumen(self) -> str:
        """Texto compacto de los KPIs (para prompts / logs)."""
        def f(v: float | None, suf: str = "%") -> str:
            return "s/d" if v is None else f"{v:+.1f}{suf}"
        top = ", ".join(p.descripcion for p in self.top_productos[:3]) or "—"
        caida = "; ".join(
            f"{p.descripcion} ({f(p.var_yoy)})" for p in self.caidas_productos[:3]) or "—"
        return (
            f"PDV {self.pdv[:8]} · {self.mes}\n"
            f"- Kilos: {self.kilos:.1f} (interanual {f(self.kilos_var_yoy)}, "
            f"vs red {f(self.kilos_var_red)})\n"
            f"- Visitas: {self.visitas} "
            f"(interanual {f(self.visitas_var_yoy)}, vs red {f(self.visitas_var_red)})\n"
            f"- Clientes únicos: {self.clientes} "
            f"(interanual {f(self.clientes_var_yoy)}, vs red {f(self.clientes_var_red)})\n"
            f"- Kilos/visita: {self.kilos_por_visita:.2f} "
            f"(interanual {f(self.kilos_por_visita_var_yoy)})\n"
            f"- Top productos: {top}\n"
            f"- Caídas relevantes: {caida}"
        )


# Agregado de volumen/tráfico por sucursal para un mes dado.
_AGG_SQL = f"""
SELECT
    branchofficeid                              AS pdv,
    COALESCE(SUM(kilos), 0)                      AS kilos,
    COALESCE(SUM(cantidad), 0)                   AS unidades,
    COUNT(DISTINCT (customerid || '|' || CAST(fecha AS DATE))) AS visitas,
    COUNT(DISTINCT customerid)                   AS clientes
FROM {TABLE}
WHERE strftime(fecha, '%Y-%m') = ?
GROUP BY branchofficeid
"""

# Productos de una sucursal en un mes (kilos y unidades por producto).
_PROD_SQL = f"""
SELECT productid,
       any_value(descripcion)        AS descripcion,
       COALESCE(SUM(kilos), 0)        AS kilos,
       COALESCE(SUM(cantidad), 0)     AS unidades
FROM {TABLE}
WHERE branchofficeid = ? AND strftime(fecha, '%Y-%m') = ?
GROUP BY productid
"""


def _agg_by_pdv(con: duckdb.DuckDBPyConnection, mes: str) -> dict[str, dict]:
    df = con.execute(_AGG_SQL, [mes]).fetchdf()
    return {row["pdv"]: row.to_dict() for _, row in df.iterrows()}


def _productos(con: duckdb.DuckDBPyConnection, pdv: str, mes: str) -> dict[int, dict]:
    df = con.execute(_PROD_SQL, [pdv, mes]).fetchdf()
    return {int(row["productid"]): row.to_dict() for _, row in df.iterrows()}


def compute_kpis(con: duckdb.DuckDBPyConnection, pdv: str, mes: str,
                 top_n: int = 5, drop_n: int = 5) -> PDVKpis:
    """KPIs determinísticos de una sucursal en un mes, con interanual y vs red.

    Args:
        con: conexión DuckDB (solo lectura).
        pdv: branchofficeid objetivo.
        mes: 'YYYY-MM'.
        top_n: cantidad de productos top por kilos a reportar.
        drop_n: cantidad de mayores caídas de producto (interanual) a reportar.
    """
    mes_yoy = _same_month_last_year(mes)
    cur = _agg_by_pdv(con, mes)
    prev = _agg_by_pdv(con, mes_yoy)

    if pdv not in cur:
        # Sucursal sin ventas en el mes: snapshot vacío pero válido.
        return PDVKpis(pdv=pdv, mes=mes, mes_yoy=mes_yoy)

    c = cur[pdv]
    p = prev.get(pdv, {})
    kilos = float(c["kilos"])
    unidades = int(c["unidades"])
    visitas = int(c["visitas"])
    clientes = int(c["clientes"])
    kilos_yoy = float(p.get("kilos", 0) or 0)
    visitas_yoy = int(p.get("visitas", 0) or 0)
    clientes_yoy = int(p.get("clientes", 0) or 0)

    kpv = kilos / visitas if visitas else 0.0
    kpv_yoy = kilos_yoy / visitas_yoy if visitas_yoy else 0.0

    # Media de la red (todas las sucursales con ventas en el mes objetivo).
    n_red = len(cur)
    red_kilos = sum(float(r["kilos"]) for r in cur.values()) / n_red
    red_visitas = sum(int(r["visitas"]) for r in cur.values()) / n_red
    red_clientes = sum(int(r["clientes"]) for r in cur.values()) / n_red

    k = PDVKpis(
        pdv=pdv, mes=mes, mes_yoy=mes_yoy,
        kilos=round(kilos, 1), unidades=unidades, visitas=visitas, clientes=clientes,
        kilos_por_visita=round(kpv, 2),
        unidades_por_visita=round(unidades / visitas, 2) if visitas else 0.0,
        kilos_yoy=round(kilos_yoy, 1), visitas_yoy=visitas_yoy,
        clientes_yoy=clientes_yoy,
        kilos_var_yoy=_pct(kilos, kilos_yoy),
        visitas_var_yoy=_pct(visitas, visitas_yoy),
        clientes_var_yoy=_pct(clientes, clientes_yoy),
        kilos_por_visita_yoy=round(kpv_yoy, 2),
        kilos_por_visita_var_yoy=_pct(kpv, kpv_yoy),
        kilos_red_media=round(red_kilos, 1),
        kilos_var_red=_pct(kilos, red_kilos),
        visitas_red_media=round(red_visitas, 1),
        visitas_var_red=_pct(visitas, red_visitas),
        clientes_red_media=round(red_clientes, 1),
        clientes_var_red=_pct(clientes, red_clientes),
    )

    # Mix de producto: top por kilos y mayores caídas interanuales.
    prod_cur = _productos(con, pdv, mes)
    prod_yoy = _productos(con, pdv, mes_yoy)
    productos: list[ProductoKpi] = []
    for pid, row in prod_cur.items():
        kp = float(prod_yoy.get(pid, {}).get("kilos", 0) or 0)
        productos.append(ProductoKpi(
            productid=pid, descripcion=str(row["descripcion"]),
            kilos=round(float(row["kilos"]), 1), unidades=int(row["unidades"]),
            kilos_yoy=round(kp, 1), var_yoy=_pct(float(row["kilos"]), kp)))

    k.top_productos = sorted(productos, key=lambda x: x.kilos, reverse=True)[:top_n]
    # Caídas: solo productos con volumen interanual significativo y baja real.
    caidas = [x for x in productos if x.var_yoy is not None and x.var_yoy < 0
              and x.kilos_yoy >= 5.0]
    k.caidas_productos = sorted(caidas, key=lambda x: x.var_yoy)[:drop_n]
    return k
