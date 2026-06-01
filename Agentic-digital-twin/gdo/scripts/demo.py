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

from src.data_layer.duckdb_loader import load_sources          # noqa: E402
from src.data_layer.schema import get_schema_context, list_tables  # noqa: E402
from src.agents.analyst_sql import TextToSQLAnalyst             # noqa: E402
from src.llm.clients import OllamaClient, StaticSQLClient       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
DB = ROOT / "data" / "gdo.duckdb"


def build():
    if DB.exists():
        DB.unlink()
    catalog = load_sources(ROOT / "data" / "raw", DB)
    print("\n=== Catálogo cargado ===")
    for t, cols in catalog.items():
        print(f"  {t}: {cols}")
    return catalog


def get_llm():
    if os.getenv("USE_OLLAMA"):
        return OllamaClient()  # real
    # Stub offline: reglas para las preguntas de la demo
    rules = [
        (("ventas hay por sucursal",),
         'SELECT sucursal, COUNT(*) AS ventas FROM datos_ventas '
         'GROUP BY sucursal ORDER BY ventas DESC'),
        (("tiempo promedio de ticket",),
         'SELECT franja_horaria, ROUND(AVG(ticket_duration_min), 2) AS prom_ticket_min '
         'FROM datos_ventas GROUP BY franja_horaria ORDER BY prom_ticket_min DESC'),
        (("nps promedio",),
         'SELECT sucursal, ROUND(AVG(nps), 2) AS nps_prom '
         'FROM encuestas_mala_experiencia GROUP BY sucursal ORDER BY nps_prom ASC'),
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
        "¿Cuál es el tiempo promedio de ticket por franja horaria?",
        "¿Cuál es el NPS promedio por sucursal según las encuestas malas?",
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
