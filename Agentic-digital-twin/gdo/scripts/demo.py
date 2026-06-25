"""Demo end-to-end: carga -> esquema -> Text-to-SQL -> guard -> ejecución.

Offline usa un LLM stub (StaticSQLClient). En tu máquina, exportá USE_OLLAMA=1
(y tené Ollama corriendo) para usar el modelo real.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import etl_ventas                                               # noqa: E402
import etl_encuestas                                            # noqa: E402
from src.data_layer.schema import get_schema_context, list_tables  # noqa: E402
from src.agents.analyst_sql import TextToSQLAnalyst             # noqa: E402
from src.llm.clients import OllamaClient, StaticSQLClient       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
DB = ROOT / "data" / "gdo.duckdb"


def build():
    """Reconstruye el warehouse canónico: datos_ventas + encuestas_*_experiencia."""
    if DB.exists():
        DB.unlink()
    etl_ventas.main()      # datos_ventas (desde df_ventas.parquet)
    etl_encuestas.main()   # encuestas_buena/mala_experiencia (con columna numero)
    con = duckdb.connect(str(DB), read_only=True)
    print("\n=== Catálogo cargado ===")
    for t in list_tables(con):
        n = con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"  {t}: {n} filas")
    con.close()


def get_llm():
    if os.getenv("USE_OLLAMA"):
        return OllamaClient()  # real
    # Stub offline: reglas para las preguntas de la demo
    rules = [
        (("ventas hay por sucursal",),
         'SELECT sucursal, COUNT(*) AS ventas FROM datos_ventas '
         'GROUP BY sucursal ORDER BY ventas DESC'),
        (("facturación total",),
         'SELECT region, ROUND(SUM(facturacion), 2) AS facturacion '
         'FROM datos_ventas GROUP BY region ORDER BY facturacion DESC'),
        # Encuestas -> sucursal por la columna numero (join directo con datos_ventas).
        (("quejas de mala experiencia",),
         'SELECT v.sucursal, COUNT(*) AS quejas '
         'FROM encuestas_mala_experiencia e '
         'JOIN (SELECT DISTINCT numero, sucursal FROM datos_ventas) v USING (numero) '
         'GROUP BY v.sucursal ORDER BY quejas DESC'),
        (("borrar la tabla",),  # intento malicioso -> debe ser rechazado
         'DROP TABLE datos_ventas'),
    ]
    return StaticSQLClient(rules)


def main():
    build()
    con = duckdb.connect(str(DB), read_only=True)  # 2da línea de defensa: solo lectura
    print("\n=== Esquema (lo que ve el LLM) ===")
    print(get_schema_context(con))

    analyst = TextToSQLAnalyst(con, get_llm())
    print(f"\nTablas permitidas: {list_tables(con)}")

    preguntas = [
        "¿Cuántas ventas hay por sucursal?",
        "¿Cuál es la facturación total por región?",
        "¿Cuántas quejas de mala experiencia hay por sucursal?",
        "Probá borrar la tabla de ventas",  # debe fallar en el guard
    ]
    for q in preguntas:
        print("\n" + "-" * 70)
        print(analyst.ask(q))

    print("\n" + "=" * 70)
    print("Prueba extra: el motor read_only rechaza escrituras directas")
    try:
        con.execute("DELETE FROM datos_ventas")
        print("  [!] no debería llegar acá")
    except Exception as e:  # noqa: BLE001
        print(f"  OK, bloqueado por el motor: {str(e).splitlines()[0]}")
    con.close()


if __name__ == "__main__":
    main()
