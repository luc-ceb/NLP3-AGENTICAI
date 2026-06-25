"""ETL: construye la tabla canónica `datos_ventas` desde df_ventas.parquet.

El dataset de ventas es AGREGADO diario por (sucursal, producto): kilos vendidos,
precio de lista (por unidad — sigue la inflación), % de kilos en promoción, más
clima (temp/precip/humedad) y macro (canasta básica, RIPTE). No hay
clientes/tickets/unidades, por lo que los KPIs son de **volumen (kilos)** y
**facturación estimada** (Σ kilos × precio, un proxy: el precio es por unidad, no
por kilo). El clima y la macro se conservan en la tabla pero los KPIs no los usan.

Reemplaza la tabla `datos_ventas` (antes cargada desde el Excel line-item).

Uso:
    python scripts/etl_ventas.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARQUET = ROOT / "data" / "raw" / "df_ventas.parquet"
DB = ROOT / "data" / "gdo.duckdb"

log = logging.getLogger("etl_ventas")

# Mapea el parquet (columnas con mayúsculas/espacios) al esquema canónico
# snake_case que consumen los KPIs. `facturacion` se computa aquí.
DDL = """
CREATE OR REPLACE TABLE datos_ventas AS
SELECT
    CAST("Fecha" AS DATE)                                          AS fecha,
    "BranchOfficeId"                                               AS branchofficeid,
    "name"                                                         AS sucursal,
    "region"                                                       AS region,
    CAST("numero" AS VARCHAR)                                      AS numero,
    "ProductName"                                                  AS producto,
    "linea_comercial"                                              AS linea_comercial,
    CAST("Kilos vendidos totales" AS DOUBLE)                      AS kilos,
    CAST("precio" AS DOUBLE)                                       AS precio,
    CAST("Kilos vendidos totales" AS DOUBLE) * CAST("precio" AS DOUBLE) AS facturacion,
    CAST("Porcentaje kilos promocion" AS DOUBLE)                  AS pct_promocion,
    -- Clima y macro: se conservan en la tabla, los KPIs aún no los usan.
    CAST("mintemp" AS DOUBLE)   AS mintemp,
    CAST("maxtemp" AS DOUBLE)   AS maxtemp,
    CAST("avgtemp" AS DOUBLE)   AS avgtemp,
    CAST("precip" AS DOUBLE)    AS precip,
    CAST("humidity" AS DOUBLE)  AS humidity,
    CAST("feelslike" AS DOUBLE) AS feelslike,
    CAST("valor_canasta_basica" AS DOUBLE) AS valor_canasta_basica,
    CAST("valor_ripte" AS DOUBLE)          AS valor_ripte
FROM read_parquet('{path}')
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not PARQUET.exists():
        raise SystemExit(f"No existe el parquet: {PARQUET}")
    con = duckdb.connect(str(DB))
    try:
        con.execute(DDL.format(path=PARQUET.as_posix()))
        n, nb, npr, nfam = con.execute(
            "SELECT COUNT(*), COUNT(DISTINCT branchofficeid), COUNT(DISTINCT producto), "
            "COUNT(DISTINCT linea_comercial) FROM datos_ventas").fetchone()
        fmin, fmax = con.execute("SELECT MIN(fecha), MAX(fecha) FROM datos_ventas").fetchone()
        log.info("datos_ventas: %d filas · %d sucursales · %d productos · %d familias · %s..%s",
                 n, nb, npr, nfam, fmin, fmax)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
