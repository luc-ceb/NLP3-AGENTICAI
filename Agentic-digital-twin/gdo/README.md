# GDO — Sistema de diagnóstico agéntico

**Gemelo Digital Operativo (GDO)** es un sistema de diagnóstico agéntico para una
cadena minorista de heladerías. Cruza los KPIs de ventas de cada sucursal contra
el manual operativo de la empresa y produce, por sucursal, un **diagnóstico + una
acción correctiva citada** del manual — o se **abstiene** honestamente cuando
ninguna norma aplica.

El razonamiento se organiza en grafos de agentes (LangGraph). Las métricas son
SQL determinístico (sin Text-to-SQL); el LLM solo interviene para auditar contra
la norma y redactar el diagnóstico, siempre con citas reales.

## Qué hace

- **Caso 2 — Diagnóstico de una sucursal.** Calcula KPIs de volumen/tráfico,
  detecta el desvío más relevante con reglas, recupera las secciones pertinentes
  del manual (RAG) y redacta diagnóstico + acción con cita, o se abstiene.
- **Caso 5 — Plan mensual.** Itera el Caso 2 sobre las 6 sucursales, las rankea
  por severidad y, con **una** llamada de síntesis, agrupa temas comunes y
  prioriza. Salida: un reporte Markdown.

## Cómo razona

- **KPIs determinísticos (SQL fijo).** Kilos, unidades, visitas (ticket proxy =
  `distinct(customerid, fecha)`) y clientes únicos por `(sucursal, mes)`. No hay
  columnas de precio/ingreso, así que las métricas son de volumen y tráfico.
- **Dos comparaciones, ambas libres de estacionalidad:**
  - **vs red** — la sucursal contra la media de la red en el **mismo mes**. Aísla
    el desempeño propio del efecto estacional que mueve a toda la red. *Señal
    primaria.*
  - **interanual (YoY)** — contra el **mismo mes del año anterior**. Detecta el
    deterioro propio aunque la red entera haya caído. *Fallback / contexto.*
- **Prioridad de desvíos:** `vs red (0) < interanual (1) < producto (2)`; dentro
  de cada nivel, mayor magnitud primero. Este orden rige tanto la selección por
  sucursal (Caso 2) como el ranking entre sucursales (Caso 5).
- **Eficiencia.** Los nodos de KPIs y detección son SQL + reglas, sin LLM. Si no
  hay desvío negativo relevante, el grafo termina en abstención **sin gastar una
  sola llamada al LLM**. Solo un desvío real dispara la auditoría RAG + LLM.
- **Grounding.** La auditoría normativa (CRAG sobre el manual) entrega pasajes
  reales; la síntesis cita esos pasajes o se autoabstiene (`fundada=false`).

## Instalación

```bash
pip install -r requirements.txt
# Stack agéntico (LLM + RAG), si no está ya instalado:
pip install groq anthropic langgraph faiss-cpu sentence-transformers fastapi uvicorn streamlit
```

## Configuración

Copiá `.env.example` a `.env` y completá las claves. El LLM usa **Groq por
defecto con respaldo automático a Anthropic** ante rate-limit (Groq tiene un cupo
diario de 100k tokens; al agotarse, la sesión conmuta al respaldo).

| Variable | Default | Descripción |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | Proveedor primario (`groq` \| `anthropic` \| `ollama`) |
| `LLM_FALLBACK_PROVIDER` | `anthropic` | Respaldo ante rate-limit del primario |
| `LLM_MAX_TOKENS` | `2048` | Holgura para JSON de síntesis/diagnóstico |
| `GROQ_API_KEY` / `GROQ_MODEL` | — / `llama-3.3-70b-versatile` | Credencial y modelo Groq |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-sonnet-4-6` | Credencial y modelo Anthropic |
| `VECTOR_BACKEND` | `faiss` | Backend denso de recuperación (la CLI prioriza FAISS local) |

## Uso (CLI)

Entrada principal. Vía módulo o el wrapper `scripts/gdo.py`. El PDV admite el
GUID completo o un **prefijo único** (p. ej. `A9D75316`); el mes por defecto es el
último mes completo.

```bash
# Caso 2 — diagnóstico de una sucursal
python -m src.interface.cli diagnosticar A9D75316 --mes 2026-04
python -m src.interface.cli diagnosticar A9D75316 --json

# Caso 5 — plan mensual rankeado (Markdown)
python -m src.interface.cli plan-mensual --out plan_2026-04.md
python -m src.interface.cli plan-mensual --json

# Chequeos livianos de comportamiento
python -m src.interface.cli eval            # sin LLM (ranking, abstención)
python -m src.interface.cli eval --with-llm # incluye chequeos que invocan al LLM
```

## Uso programático

```python
import duckdb
from src.agents.caso5 import PlanificadorMensual
from src.agents.auditor_rag import NormativeAuditor
from src.rag.retrieve import HybridRetriever
from src.llm.clients import make_llm_with_fallback

con = duckdb.connect("data/gdo.duckdb", read_only=True)
llm = make_llm_with_fallback(max_tokens=2048)
auditor = NormativeAuditor(HybridRetriever.build_default("data/index"), llm)

plan = PlanificadorMensual(con, auditor, llm)
print(plan.generar("2026-04").reporte)
```

## Otras interfaces

- **API REST (FastAPI):** `uvicorn src.interface.api:create_app --factory`
  (`POST /diagnose`, `POST /consultar`, `GET /health`; autenticación opcional por
  header `x-api-key`).
- **UI (Streamlit):** `streamlit run src/interface/app.py`.

## Preparación de datos (ETL e índice)

La CLI espera `data/gdo.duckdb` (hechos de ventas) y `data/index/` (índice del
manual: `meta.jsonl`, `bm25.pkl`, `dense.faiss`, `dense.json`).

- **Datos:** `datos-ventas.xlsx` (line-item, 6 sucursales, ~190 productos,
  abr-2025 → may-2026) cargados a DuckDB. *La comparación interanual existe para
  2026-01…05; meses/sucursales sin año previo caen al comparativo vs red.*
- **Manual:** `manual_operativo_completo.md` chunkeado por Tema (con metadatos de
  SECCIÓN) e indexado. Si el denso FAISS quedara desincronizado con
  `meta.jsonl`/`bm25.pkl`, reconstruilo con:
  ```bash
  python scripts/rebuild_faiss.py
  ```

## Estructura

```
src/kpi/         KPIs determinísticos (metrics) y detección de desvíos (deviations)
src/agents/      caso2 (DiagnosticadorPDV), caso5 (PlanificadorMensual), auditor_rag (CRAG)
src/rag/         recuperación híbrida (BM25 + denso FAISS), fusión, reranker
src/llm/         clientes LLM (Groq/Anthropic/Ollama) + respaldo automático
src/interface/   cli (entrada principal), api (FastAPI), app (Streamlit)
src/eval/        chequeos livianos (checks) y métricas IR (metrics)
src/data_layer/  carga a DuckDB e introspección de esquema
src/security/    validación de SQL (solo lectura, allow-list)
scripts/         gdo (wrapper CLI), rebuild_faiss, demos de Caso 2 / Caso 5
```

## Evaluación

Chequeos livianos de comportamiento (`src/eval/checks.py`):

- **Sin LLM (default):** `ranking_6_pdvs` (6 sucursales, orden monótono por
  prioridad/severidad) y `abstencion_sin_llm` (tripwire: sin desvío → sin LLM).
- **Con `--with-llm`:** cita cuando hay norma, abstiene cuando no la hay, y genera
  el reporte de Caso 5.

Métricas IR formales (Hit@k, MRR, nDCG) quedan en `src/eval/metrics.py`.

## Decisiones de diseño

- **KPIs por SQL fijo, no Text-to-SQL** — más robusto para la demo.
- **Un solo modelo LLM** (sin split FAST/REASON).
- **vs red** como señal primaria; **interanual** y caídas de producto como
  contexto/fallback.

## Fuera de alcance (fases siguientes)

Encuestas de clientes, clasificador de temas / router (Caso 1), Text-to-SQL y
tiers FAST/REASON.
