"""Cálculo determinístico de KPIs por (sucursal, período) sobre DuckDB.

Todas las consultas son SELECT parametrizadas (sin Text-to-SQL). Los datos son
agregados diarios por (sucursal, producto): kilos vendidos, precio de lista y %
en promoción (no hay clientes/tickets/unidades). Por eso los KPIs son de:

  - **volumen** (kilos), y
  - **facturación estimada** (Σ kilos × precio — proxy; el precio es por unidad).

El **período** de análisis es una ventana de fechas arbitraria ``[desde, hasta]``
(un mes es solo el caso particular del primer al último día del mes). Cada métrica
se compara contra (a) el **mismo período del año anterior** (interanual / YoY) y
(b) la media de la red en el mismo período. La facturación SOLO se compara vs red:
su variación temporal está dominada por la inflación (el precio ~10×'ó en 4 años),
así que el eje temporal se mide con kilos (volumen real, sin inflación).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import duckdb

TABLE = "datos_ventas"


def _shift_year_back(d: date) -> date:
    """Misma fecha un año antes (29-feb cae a 28-feb)."""
    try:
        return d.replace(year=d.year - 1)
    except ValueError:  # 29 de febrero en año no bisiesto previo
        return d.replace(year=d.year - 1, day=28)


@dataclass(frozen=True)
class Periodo:
    """Ventana de análisis ``[desde, hasta]`` (ambos inclusive) con etiqueta legible."""
    desde: date
    hasta: date
    etiqueta: str

    @classmethod
    def de_mes(cls, mes: str) -> "Periodo":
        """Construye el período que cubre un mes completo ('YYYY-MM')."""
        y, m = (int(x) for x in mes.split("-"))
        d0 = date(y, m, 1)
        d1 = date(y + (m == 12), (m % 12) + 1, 1) - timedelta(days=1)
        return cls(d0, d1, mes)

    @classmethod
    def de_rango(cls, desde: date, hasta: date) -> "Periodo":
        """Construye un período a partir de dos fechas ISO (desde ≤ hasta)."""
        if hasta < desde:
            raise ValueError(f"'desde' ({desde}) no puede ser posterior a 'hasta' ({hasta}).")
        return cls(desde, hasta, f"{desde.isoformat()} a {hasta.isoformat()}")

    @classmethod
    def coerce(cls, p: "Periodo | str") -> "Periodo":
        """Acepta un Periodo o un 'YYYY-MM' (atajo de mes) y devuelve un Periodo."""
        if isinstance(p, Periodo):
            return p
        if isinstance(p, str):
            return cls.de_mes(p)
        raise TypeError(f"Período inválido: {p!r} (esperaba Periodo o 'YYYY-MM').")

    def yoy(self) -> "Periodo":
        """El mismo período corrido un año hacia atrás (baseline interanual)."""
        d0, d1 = _shift_year_back(self.desde), _shift_year_back(self.hasta)
        return Periodo(d0, d1, f"{self.etiqueta} (año anterior)")

    def __str__(self) -> str:
        return self.etiqueta


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


def _as_date(x) -> date | None:
    """DuckDB devuelve date para columnas DATE; tolera datetime/None por las dudas."""
    if x is None:
        return None
    return x.date() if isinstance(x, datetime) else x


def data_span(con: duckdb.DuckDBPyConnection) -> tuple[date | None, date | None]:
    """Rango de fechas con datos en la tabla de ventas (min, max)."""
    lo, hi = con.execute(f"SELECT MIN(fecha), MAX(fecha) FROM {TABLE}").fetchone()
    return _as_date(lo), _as_date(hi)


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
    md = _as_date(max_day)
    if md is not None and date.fromordinal(eom) > md:
        return months[-2] if len(months) > 1 else last
    return last


def periodo_por_defecto(con: duckdb.DuckDBPyConnection) -> Periodo:
    """Período por defecto: el último mes COMPLETO disponible."""
    return Periodo.de_mes(latest_month(con, complete_only=True))


@dataclass
class ProductoKpi:
    producto: str              # nombre del producto (la fuente no trae id)
    kilos: float
    kilos_yoy: float = 0.0
    var_yoy: float | None = None  # variación % de kilos vs el mismo período del año anterior


@dataclass
class PDVKpis:
    """Snapshot de KPIs de una sucursal en un período, con comparativas."""
    pdv: str
    periodo: str               # etiqueta del período objetivo
    periodo_yoy: str           # etiqueta del mismo período del año anterior

    # Volumen y facturación (período objetivo)
    kilos: float = 0.0
    facturacion: float = 0.0   # Σ kilos × precio (proxy de facturación)
    pct_promocion: float = 0.0  # % de kilos en promoción (ponderado por kilos)

    # Comparativa interanual (YoY): mismo período del año anterior. Cancela la
    # estacionalidad; se aplica al VOLUMEN (kilos), libre de inflación. Es None si
    # la fuente no cubre por completo la ventana del año anterior.
    kilos_yoy: float = 0.0
    kilos_var_yoy: float | None = None

    # Comparativa contra la red (media de sucursales en el mismo período). Aísla el
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
            f"PDV {self.pdv[:8]} · {self.periodo}\n"
            f"- Kilos: {self.kilos:.1f} (interanual {f(self.kilos_var_yoy)}, "
            f"vs red {f(self.kilos_var_red)})\n"
            f"- Facturación est.: {self.facturacion:,.0f} "
            f"(vs red {f(self.facturacion_var_red)})\n"
            f"- En promoción: {self.pct_promocion:.1f}% de kilos "
            f"(red {self.pct_promocion_red_media:.1f}%)\n"
            f"- Top productos: {top}\n"
            f"- Caídas relevantes (interanual): {caida}"
        )


# Agregado de volumen/facturación por sucursal en una ventana de fechas. La
# promoción se pondera por kilos (no es un simple promedio de filas).
_AGG_SQL = f"""
SELECT
    branchofficeid                                       AS pdv,
    COALESCE(SUM(kilos), 0)                              AS kilos,
    COALESCE(SUM(facturacion), 0)                        AS facturacion,
    COALESCE(SUM(kilos * pct_promocion) / NULLIF(SUM(kilos), 0), 0) AS pct_promocion
FROM {TABLE}
WHERE fecha BETWEEN ? AND ?
GROUP BY branchofficeid
"""

# Kilos por producto de una sucursal en una ventana de fechas.
_PROD_SQL = f"""
SELECT producto,
       COALESCE(SUM(kilos), 0) AS kilos
FROM {TABLE}
WHERE branchofficeid = ? AND fecha BETWEEN ? AND ?
GROUP BY producto
"""


def _agg_by_pdv(con: duckdb.DuckDBPyConnection, p: Periodo) -> dict[str, dict]:
    df = con.execute(_AGG_SQL, [p.desde, p.hasta]).fetchdf()
    return {row["pdv"]: row.to_dict() for _, row in df.iterrows()}


def _productos(con: duckdb.DuckDBPyConnection, pdv: str, p: Periodo) -> dict[str, float]:
    df = con.execute(_PROD_SQL, [pdv, p.desde, p.hasta]).fetchdf()
    return {str(row["producto"]): float(row["kilos"]) for _, row in df.iterrows()}


def compute_kpis(con: duckdb.DuckDBPyConnection, pdv: str, periodo: Periodo | str,
                 top_n: int = 5, drop_n: int = 5) -> PDVKpis:
    """KPIs determinísticos de una sucursal en un período, con interanual y vs red.

    Args:
        con: conexión DuckDB (solo lectura).
        pdv: branchofficeid objetivo.
        periodo: :class:`Periodo` (ventana de fechas) o un 'YYYY-MM' (atajo de mes).
        top_n: cantidad de productos top por kilos a reportar.
        drop_n: cantidad de mayores caídas de producto (interanual) a reportar.

    La comparativa interanual se SUPRIME (queda en None) si la fuente no cubre por
    completo la ventana del año anterior; en ese caso el desvío cae al vs-red.
    """
    periodo = Periodo.coerce(periodo)
    yoy = periodo.yoy()
    lo, hi = data_span(con)
    # YoY solo es comparable si la fuente cubre TODA la ventana del año anterior;
    # de lo contrario compararíamos contra datos parciales (engañoso).
    yoy_cubierto = lo is not None and lo <= yoy.desde and yoy.hasta <= hi

    cur = _agg_by_pdv(con, periodo)
    if pdv not in cur:
        # Sucursal sin ventas en el período: snapshot vacío pero válido.
        return PDVKpis(pdv=pdv, periodo=periodo.etiqueta, periodo_yoy=yoy.etiqueta)

    prev = _agg_by_pdv(con, yoy) if yoy_cubierto else {}
    c = cur[pdv]
    p = prev.get(pdv, {})
    kilos = float(c["kilos"])
    facturacion = float(c["facturacion"])
    pct_promocion = float(c["pct_promocion"])
    kilos_yoy = float(p.get("kilos", 0) or 0)

    # Media de la red (todas las sucursales con ventas en el período objetivo).
    n_red = len(cur)
    red_kilos = sum(float(r["kilos"]) for r in cur.values()) / n_red
    red_fact = sum(float(r["facturacion"]) for r in cur.values()) / n_red
    red_promo = sum(float(r["pct_promocion"]) for r in cur.values()) / n_red

    k = PDVKpis(
        pdv=pdv, periodo=periodo.etiqueta, periodo_yoy=yoy.etiqueta,
        kilos=round(kilos, 1), facturacion=round(facturacion, 1),
        pct_promocion=round(pct_promocion, 1),
        kilos_yoy=round(kilos_yoy, 1),
        kilos_var_yoy=_pct(kilos, kilos_yoy) if yoy_cubierto else None,
        kilos_red_media=round(red_kilos, 1),
        kilos_var_red=_pct(kilos, red_kilos),
        facturacion_red_media=round(red_fact, 1),
        facturacion_var_red=_pct(facturacion, red_fact),
        pct_promocion_red_media=round(red_promo, 1),
    )

    # Mix de producto: top por kilos y mayores caídas interanuales.
    prod_cur = _productos(con, pdv, periodo)
    prod_yoy = _productos(con, pdv, yoy) if yoy_cubierto else {}
    productos: list[ProductoKpi] = []
    for nombre, kg in prod_cur.items():
        kp = float(prod_yoy.get(nombre, 0) or 0)
        productos.append(ProductoKpi(
            producto=nombre, kilos=round(kg, 1),
            kilos_yoy=round(kp, 1),
            var_yoy=_pct(kg, kp) if yoy_cubierto else None))

    k.top_productos = sorted(productos, key=lambda x: x.kilos, reverse=True)[:top_n]
    # Caídas: solo productos con volumen interanual significativo y baja real (si no
    # hay YoY comparable, var_yoy es None y no se reportan caídas de producto).
    caidas = [x for x in productos if x.var_yoy is not None and x.var_yoy < 0
              and x.kilos_yoy >= 5.0]
    k.caidas_productos = sorted(caidas, key=lambda x: x.var_yoy)[:drop_n]
    return k
