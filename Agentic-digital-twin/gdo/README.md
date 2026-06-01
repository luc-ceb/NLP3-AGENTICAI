# GDO — Capa de datos + Analista Text-to-SQL

Primera fase del Gemelo Digital Operativo: carga de los Excel a DuckDB y un
agente que responde preguntas en lenguaje natural traduciéndolas a SQL, con
validación de seguridad (solo lectura).

## Instalación
```bash
pip install -r requirements.txt
```

## Uso con datos sintéticos (prueba rápida, sin Ollama)
```bash
python scripts/make_sample_data.py   # genera data/raw/*.xlsx de ejemplo
python scripts/demo.py               # carga + esquema + Text-to-SQL (stub offline)
```

## Uso con tus datos reales + Ollama
1. Poné tus archivos en `data/raw/` (ej.: `datos-ventas.xlsx`,
   `encuestas-buena-experiencia.xlsx`, `encuestas-mala-experiencia.xlsx`).
   El loader es **agnóstico al esquema**: lee todas las hojas y limpia los
   nombres de columna automáticamente. No hace falta tocar el código.
2. Asegurate de tener Ollama corriendo con un modelo de código, p. ej.:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```
3. Corré la demo apuntando al modelo real:
   ```bash
   USE_OLLAMA=1 python scripts/demo.py
   ```

## Uso programático
```python
import duckdb
from src.data_layer.duckdb_loader import load_sources
from src.agents.analyst_sql import TextToSQLAnalyst
from src.llm.clients import OllamaClient

load_sources("data/raw", "data/gdo.duckdb")
con = duckdb.connect("data/gdo.duckdb", read_only=True)
analyst = TextToSQLAnalyst(con, OllamaClient())
print(analyst.ask("¿Qué sucursal tuvo el ticket más lento los sábados?"))
```

## Seguridad (dos capas)
- `src/security/sql_guard.py`: rechaza todo lo que no sea una única consulta
  SELECT/WITH y limita las tablas a las cargadas (allow-list).
- La conexión del analista se abre en `read_only=True`: el motor DuckDB
  rechaza cualquier escritura aunque se cuele.

## Estructura
```
src/data_layer/   carga (duckdb_loader) e introspección de esquema (schema)
src/security/     validación de SQL (sql_guard)
src/agents/       analista Text-to-SQL (analyst_sql)
src/llm/          clientes LLM (Ollama / stub)
scripts/          make_sample_data, demo
```
