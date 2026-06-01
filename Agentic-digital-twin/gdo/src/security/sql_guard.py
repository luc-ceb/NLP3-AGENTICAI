"""Validación de SQL: solo lectura, una sentencia, tablas en allow-list.

Es la primera línea de defensa (§6 del diseño). La segunda es abrir la conexión
DuckDB en modo read_only, con lo cual cualquier escritura es rechazada por el motor.
"""
from __future__ import annotations

import re

FORBIDDEN = {
    "insert", "update", "delete", "drop", "alter", "create", "attach", "detach",
    "copy", "install", "load", "pragma", "export", "import", "replace",
    "truncate", "grant", "revoke", "call", "vacuum", "merge",
}


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", "", sql)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
    return sql


def validate_sql(sql: str, allowed_tables: list[str] | None = None) -> tuple[bool, str]:
    """Devuelve (ok, motivo)."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return False, "SQL vacío"

    s_nc = _strip_comments(s).strip()
    if ";" in s_nc.rstrip(";"):
        return False, "Solo se permite una sentencia"

    low = s_nc.lower()
    first = re.match(r"\s*(\w+)", low)
    if not first or first.group(1) not in ("select", "with"):
        return False, "Solo se permiten consultas SELECT/WITH"

    for kw in FORBIDDEN:
        if re.search(rf"\b{kw}\b", low):
            return False, f"Palabra clave no permitida: {kw.upper()}"

    if allowed_tables is not None:
        refs = set(re.findall(r'(?:from|join)\s+["`]?(\w+)', low))
        ctes = set(re.findall(r'(\w+)\s+as\s*\(', low))  # nombres de CTE
        allowed = {t.lower() for t in allowed_tables}
        unknown = refs - allowed - ctes
        if unknown:
            return False, f"Tabla(s) no permitida(s): {sorted(unknown)}"

    return True, "ok"
