"""Cálculo determinístico de KPIs por (sucursal, mes) sobre DuckDB.

Todas las consultas son SELECT parametrizadas (sin Text-to-SQL). Los datos son
agregados diarios por (sucursal, producto): kilos vendidos, precio de lista y %
en promoción (no hay clientes/tickets/unidades). Por eso los KPIs son de:

  - **volumen** (kilos), y
  - **facturación estimada** (Σ kilos × precio — proxy; el precio es por unidad).

Cada métrica se compara contra (a) el mismo mes del año anterior (interanual /
YoY) y (b) la media de la red en el mismo mes. La facturación SOLO se compara vs
red: su variación temporal está dominada por la inflación (el precio ~10×'ó en 4
años), así que el eje temporal se mide con kilos (volumen real, sin inflación).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

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
    # `fecha` es DATE (DuckDB devuelve date); tolera también datetime por las dudas.
    md = max_day.date() if isinstance(max_day, datetime) else max_day
    if md is not None and date.fromordinal(eom) > md:
        return months[-2] if len(months) > 1 else last
    return last


@dataclass
class ProductoKpi:
    producto: str              # nombre del producto (la fuente no trae id)
    kilos: float
    kilos_yoy: float = 0.0
    var_yoy: float | None = None  # variación % de kilos vs el mismo mes del año anterior


@dataclass
class PDVKpis:
    """Snapshot de KPIs de una sucursal en un mes, con comparativas."""
    pdv: str
    mes: str
    mes_yoy: str               # mismo mes del año anterior ('YYYY-MM')

    # Volumen y facturación (mes objetivo)
    kilos: float = 0.0
    facturacion: float = 0.0   # Σ kilos × precio (proxy de facturación)
    pct_promocion: float = 0.0  # % de kilos en promoción (ponderado por kilos)

    # Comparativa interanual (YoY): mismo mes del año anterior. Cancela la
    # estacionalidad; se aplica al VOLUMEN (kilos), libre de inflación.
    kilos_yoy: float = 0.0
    kilos_var_yoy: float | None = None

    # Comparativa contra la red (media de sucursales en el mismo mes). Aísla el
    # desempeño propio del efecto estacional. La facturación se compara solo aquí
    # (vs red), no en el tiempo, porque la inflación domina su variación temporal.
    kilos_red_media: float = 0.0
    kilos_var_red: float | None = None
    facturacion_red_media: float = 0.0
    facturacion_var_red: float | None = None
    pct_promocion_red_media: float = 0.0

    # Mix de producto
    top_productos: list[ProductoKpi] = field(default_factory=list)
    caidas_productos: list[ProductoKpi] = field(default_factory=list)

    def resumen(self) -> str:
        """Texto compacto de los KPIs (para prompts / logs)."""
        def f(v: float | None, suf: str = "%") -> str:
            return "s/d" if v is None else f"{v:+.1f}{suf}"
        top = ", ".join(p.producto for p in self.top_productos[:3]) or "—"
        caida = "; ".join(
            f"{p.producto} ({f(p.var_yoy)})" for p in self.caidas_productos[:3]) or "—"
        return (
            f"PDV {self.pdv[:8]} · {self.mes}\n"
            f"- Kilos: {self.kilos:.1f} (interanual {f(self.kilos_var_yoy)}, "
            f"vs red {f(self.kilos_var_red)})\n"
            f"- Facturación est.: {self.facturacion:,.0f} "
            f"(vs red {f(self.facturacion_var_red)})\n"
            f"- En promoción: {self.pct_promocion:.1f}% de kilos "
            f"(red {self.pct_promocion_red_media:.1f}%)\n"
            f"- Top productos: {top}\n"
            f"- Caídas relevantes (interanual): {caida}"
        )


# Agregado de volumen/facturación por sucursal para un mes dado. La promoción se
# pondera por kilos (no es un simple promedio de filas).
_AGG_SQL = f"""
SELECT
    branchofficeid                                       AS pdv,
    COALESCE(SUM(kilos), 0)                              AS kilos,
    COALESCE(SUM(facturacion), 0)                        AS facturacion,
    COALESCE(SUM(kilos * pct_promocion) / NULLIF(SUM(kilos), 0), 0) AS pct_promocion
FROM {TABLE}
WHERE strftime(fecha, '%Y-%m') = ?
GROUP BY branchofficeid
"""

# Kilos por producto de una sucursal en un mes.
_PROD_SQL = f"""
SELECT producto,
       COALESCE(SUM(kilos), 0) AS kilos
FROM {TABLE}
WHERE branchofficeid = ? AND strftime(fecha, '%Y-%m') = ?
GROUP BY producto
"""


def _agg_by_pdv(con: duckdb.DuckDBPyConnection, mes: str) -> dict[str, dict]:
    df = con.execute(_AGG_SQL, [mes]).fetchdf()
    return {row["pdv"]: row.to_dict() for _, row in df.iterrows()}


def _productos(con: duckdb.DuckDBPyConnection, pdv: str, mes: str) -> dict[str, float]:
    df = con.execute(_PROD_SQL, [pdv, mes]).fetchdf()
    return {str(row["producto"]): float(row["kilos"]) for _, row in df.iterrows()}


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
    facturacion = float(c["facturacion"])
    pct_promocion = float(c["pct_promocion"])
    kilos_yoy = float(p.get("kilos", 0) or 0)

    # Media de la red (todas las sucursales con ventas en el mes objetivo).
    n_red = len(cur)
    red_kilos = sum(float(r["kilos"]) for r in cur.values()) / n_red
    red_fact = sum(float(r["facturacion"]) for r in cur.values()) / n_red
    red_promo = sum(float(r["pct_promocion"]) for r in cur.values()) / n_red

    k = PDVKpis(
        pdv=pdv, mes=mes, mes_yoy=mes_yoy,
        kilos=round(kilos, 1), facturacion=round(facturacion, 1),
        pct_promocion=round(pct_promocion, 1),
        kilos_yoy=round(kilos_yoy, 1),
        kilos_var_yoy=_pct(kilos, kilos_yoy),
        kilos_red_media=round(red_kilos, 1),
        kilos_var_red=_pct(kilos, red_kilos),
        facturacion_red_media=round(red_fact, 1),
        facturacion_var_red=_pct(facturacion, red_fact),
        pct_promocion_red_media=round(red_promo, 1),
    )

    # Mix de producto: top por kilos y mayores caídas interanuales.
    prod_cur = _productos(con, pdv, mes)
    prod_yoy = _productos(con, pdv, mes_yoy)
    productos: list[ProductoKpi] = []
    for nombre, kg in prod_cur.items():
        kp = float(prod_yoy.get(nombre, 0) or 0)
        productos.append(ProductoKpi(
            producto=nombre, kilos=round(kg, 1),
            kilos_yoy=round(kp, 1), var_yoy=_pct(kg, kp)))

    k.top_productos = sorted(productos, key=lambda x: x.kilos, reverse=True)[:top_n]
    # Caídas: solo productos con volumen interanual significativo y baja real.
    caidas = [x for x in productos if x.var_yoy is not None and x.var_yoy < 0
              and x.kilos_yoy >= 5.0]
    k.caidas_productos = sorted(caidas, key=lambda x: x.var_yoy)[:drop_n]
    return k
