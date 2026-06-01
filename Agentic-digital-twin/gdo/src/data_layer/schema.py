"""Construye el contexto de esquema que se le pasa al LLM para Text-to-SQL."""
from __future__ import annotations

import duckdb


def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in con.execute("SHOW TABLES").fetchall()]


def get_schema_context(con: duckdb.DuckDBPyConnection, sample_rows: int = 3) -> str:
    """Texto compacto: por tabla, columnas+tipos, conteo de filas y una muestra."""
    parts: list[str] = []
    for t in list_tables(con):
        info = con.execute(f'PRAGMA table_info("{t}")').fetchdf()
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        cols = ", ".join(f"{r['name']} ({r['type']})" for _, r in info.iterrows())
        sample = con.execute(f'SELECT * FROM "{t}" LIMIT {sample_rows}').fetchdf()
        parts.append(
            f'Tabla "{t}" ({n} filas)\n'
            f"  Columnas: {cols}\n"
            f"  Muestra:\n{sample.to_string(index=False)}"
        )
    return "\n\n".join(parts)
