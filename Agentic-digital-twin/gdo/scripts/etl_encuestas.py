"""ETL: construye las tablas canónicas de encuestas desde los CSV de data/raw.

Reemplaza las tablas:
  - `encuestas_buena_experiencia`  ← encuestas_satisfaccion.csv
  - `encuestas_mala_experiencia`   ← encuestas_insatisfaccion.csv

Las encuestas son texto libre del cliente (`origen_respuesta_texto`) con fecha de
respuesta y, opcionalmente, email. La novedad respecto de versiones previas es la
columna **`numero`**: el código de sucursal (los mismos 14 valores que
`datos_ventas.numero`), que permite atribuir cada encuesta a su sucursal con un
join directo —sin depender del email, nulo en ~50% de los casos—.

Las columnas conservan los nombres snake_case ya conocidos por el analista; se
agrega `numero` y se tipa `origen_fhrespuesta` como TIMESTAMP.

Uso:
    python scripts/etl_encuestas.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw"
DB = ROOT / "data" / "gdo.duckdb"

log = logging.getLogger("etl_encuestas")

# (tabla destino, archivo fuente). El CSV trae un BOM en el header que DuckDB
# descarta solo; `numero` se castea a VARCHAR para alinear con datos_ventas.numero.
SOURCES = [
    ("encuestas_buena_experiencia", "encuestas_satisfaccion.csv"),
    ("encuestas_mala_experiencia", "encuestas_insatisfaccion.csv"),
]

DDL = """
CREATE OR REPLACE TABLE "{table}" AS
SELECT
    CAST("numero" AS VARCHAR)            AS numero,
    CAST("origen_fhRespuesta" AS TIMESTAMP) AS origen_fhrespuesta,
    "email"                              AS email,
    "origen_respuesta_texto"             AS origen_respuesta_texto
FROM read_csv_auto('{path}')
"""


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    missing = [f for _, f in SOURCES if not (RAW / f).exists()]
    if missing:
        raise SystemExit(f"Faltan archivos en {RAW}: {', '.join(missing)}")
    con = duckdb.connect(str(DB))
    try:
        for table, fname in SOURCES:
            con.execute(DDL.format(table=table, path=(RAW / fname).as_posix()))
            n, nb, fmin, fmax = con.execute(
                f'SELECT COUNT(*), COUNT(DISTINCT numero), '
                f'MIN(origen_fhrespuesta), MAX(origen_fhrespuesta) FROM "{table}"'
            ).fetchone()
            log.info("%s: %d filas · %d sucursales · %s..%s", table, n, nb, fmin, fmax)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
