"""Carga agnóstica al esquema de los archivos de data/raw a una base DuckDB.

Lee cualquier .xlsx (todas las hojas), .xls o .csv que encuentre, limpia los
nombres de columna a snake_case sin acentos, y crea una tabla por archivo/hoja.
La base se persiste en disco para poder reabrirla en modo solo-lectura.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


def slug(name: str) -> str:
    """Normaliza un identificador: sin acentos, snake_case, solo [a-z0-9_]."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w]+", "_", s.strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    cols: list[str] = []
    for c in df.columns:
        base = slug(c)
        if base in seen:
            seen[base] += 1
            base = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        cols.append(base)
    df = df.copy()
    df.columns = cols
    return df


def load_sources(raw_dir: str | Path, db_path: str | Path) -> dict[str, list[str]]:
    """Construye la base DuckDB a partir de los archivos de raw_dir.

    Devuelve un catálogo {nombre_tabla: [columnas]}.
    """
    raw = Path(raw_dir)
    catalog: dict[str, list[str]] = {}
    con = duckdb.connect(str(db_path))
    try:
        files = sorted(
            p for p in raw.glob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv")
        )
        if not files:
            log.warning("No se encontraron archivos en %s", raw)
        for f in files:
            try:
                if f.suffix.lower() == ".csv":
                    sheets = {f.stem: pd.read_csv(f)}
                else:
                    sheets = pd.read_excel(f, sheet_name=None)  # todas las hojas
                multi = len(sheets) > 1
                for sheet, df in sheets.items():
                    if df is None or df.empty:
                        continue
                    df = _clean_columns(df)
                    tname = slug(f"{f.stem}_{sheet}" if multi else f.stem)
                    con.register("_tmp_df", df)
                    con.execute(f'CREATE OR REPLACE TABLE "{tname}" AS SELECT * FROM _tmp_df')
                    con.unregister("_tmp_df")
                    catalog[tname] = list(df.columns)
                    log.info("Cargada tabla '%s' (%d filas)", tname, len(df))
            except Exception as e:  # noqa: BLE001
                log.error("No se pudo cargar %s: %s", f.name, e)
    finally:
        con.close()
    return catalog
