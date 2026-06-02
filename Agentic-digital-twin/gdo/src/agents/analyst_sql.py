"""Agente Analista: Text-to-SQL sobre DuckDB con validación de seguridad."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import duckdb

from ..data_layer.schema import get_schema_context, list_tables
from ..security.sql_guard import validate_sql

log = logging.getLogger(__name__)

SQL_PROMPT = """Sos un analista de datos experto en SQL (dialecto DuckDB).
Generá UNA sola consulta SQL de tipo SELECT que responda la pregunta del usuario.

Reglas:
- Usá solo las tablas y columnas del esquema. Respetá los nombres EXACTOS.
- Nunca uses INSERT/UPDATE/DELETE/DROP ni otras sentencias de escritura.
- Si la pregunta es ambigua, elegí la interpretación más razonable.
- Devolvé SOLO la consulta SQL, sin explicaciones ni backticks.
{hints}
Esquema:
{schema}

Pregunta: {question}
SQL:"""


def _extract_sql(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    return t.strip().rstrip(";").strip()


@dataclass
class SQLResult:
    question: str
    sql: str
    ok: bool
    reason: str = "ok"
    rows: Any = None  # pandas.DataFrame

    def __str__(self) -> str:
        head = f"Q: {self.question}\nSQL: {self.sql}\n"
        if self.ok and self.rows is not None:
            return head + self.rows.to_string(index=False)
        return head + f"[FALLÓ] {self.reason}"


class TextToSQLAnalyst:
    """Traduce preguntas en lenguaje natural a SQL, valida y ejecuta."""

    def __init__(self, con: duckdb.DuckDBPyConnection, llm, max_retries: int = 1,
                 hints: str = ""):
        self.con = con
        self.llm = llm
        self.max_retries = max_retries
        self.hints = hints
        self.allowed = list_tables(con)
        self.schema = get_schema_context(con)

    def ask(self, question: str) -> SQLResult:
        sql, last_reason = "", ""
        for attempt in range(self.max_retries + 1):
            prompt = SQL_PROMPT.format(schema=self.schema, question=question, hints=self.hints)
            if attempt > 0:
                prompt += f"\n(El intento anterior fue inválido: {last_reason}. Corregilo.)"
            sql = _extract_sql(self.llm.complete(prompt))

            ok, reason = validate_sql(sql, self.allowed)
            if not ok:
                last_reason = reason
                log.warning("SQL inválido (intento %d): %s", attempt, reason)
                continue
            try:
                df = self.con.execute(sql).fetchdf()
                return SQLResult(question, sql, True, "ok", df)
            except Exception as e:  # noqa: BLE001
                last_reason = f"error de ejecución: {e}"
                log.warning(last_reason)
        return SQLResult(question, sql, False, last_reason, None)
