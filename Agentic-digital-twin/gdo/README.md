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

- **Caso 1 — Clasificador / router de encuestas.** Clasifica el texto libre de
  las encuestas en un **único tema** (taxonomía fija de 9 categorías), agrega el
  volumen por tema (y por sucursal) y **rutea el tema dominante al manual**
  (NormativeAuditor) para producir una acción correctiva citada — o se abstiene si
  el tema es `otros` o ninguna norma aplica. La clasificación se hace en lotes
  (pocas llamadas, no una por queja) y la auditoría RAG corre una sola vez.
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
  - **vs red** — la sucursal contra la media de la red en el **mismo período**.
    Aísla el desempeño propio del efecto estacional que mueve a toda la red.
    *Señal primaria.*
  - **interanual (YoY)** — contra el **mismo período del año anterior**. Detecta
    el deterioro propio aunque la red entera haya caído. *Fallback / contexto.*
- **Prioridad de desvíos:** `vs red (0) < interanual (1) < familia (2)`; dentro
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

## Puesta en marcha en una PC nueva (clon)

Al clonar el repo, el **código y las recetas** vienen; los **datos** y los
**artefactos derivados** (base DuckDB, índice RAG) **no** — toda la carpeta
`gdo/data/` está en `.gitignore`. Hay que regenerarlos en destino.

Regla mental: del repo sale código; los datos y la base/índice los reconstruís vos.

| Paso | ¿Viene en el repo? | Acción |
|------|--------------------|--------|
| Código (`src/`, `scripts/`) | sí | nada |
| Dependencias | sí (`requirements.txt`) | `pip install` |
| Claves de API | no (solo `.env.example`) | crear `.env` |
| **Datos crudos** (`data/raw/`) | **no — gitignored** | **copiar a mano** |
| Base DuckDB + índice | no — gitignored | **regenerar** (abajo) |
| Modelos HuggingFace (embeddings, reranker) | no | se descargan solos la 1ª vez |

**1. Entorno** (probado con Python 3.12):

```bash
cd gdo
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Claves:** `cp .env.example .env` y completá `GROQ_API_KEY` / `ANTHROPIC_API_KEY`.

**3. Traer los datos crudos** (lo que más se olvida — no viaja en el repo). Copiá a
`gdo/data/raw/`:

```
data/raw/
├── df_ventas.parquet                 # ventas (agregado diario por sucursal·producto)
├── encuestas_satisfaccion.csv        # encuestas "buena experiencia"
├── encuestas_insatisfaccion.csv      # encuestas "mala experiencia"
└── manual-operativo-general/         # manual operativo (Markdown) + recursos
```

**4. Regenerar base e índice** (los bloques A y B son independientes; necesitás ambos):

```bash
# (A) Hechos → DuckDB (data/gdo.duckdb)
python scripts/etl_ventas.py        # tabla datos_ventas
python scripts/etl_encuestas.py     # encuestas_buena/mala_experiencia

# (B) Conocimiento → índice RAG (data/index/)
python scripts/ingest_kb.py         # data/raw → data/processed/chunks.jsonl
python scripts/build_index.py       # chunks.jsonl → FAISS + BM25
```

> `rebuild_faiss.py` **no** es para esto: solo resincroniza el denso si quedó
> desfasado del `meta.jsonl`. La 1ª corrida del RAG descarga `all-MiniLM-L6-v2` y el
> Cross-Encoder (~80 MB) desde HuggingFace (requiere internet esa vez).

**5. Verificar:**

```bash
python -m src.interface.cli eval                                   # chequeos sin LLM
VECTOR_BACKEND=faiss python -m src.interface.cli clasificar-encuestas --limite 20
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
GUID completo o un **prefijo único** (p. ej. `0008C682`).

El **período de análisis** es un mes (`--mes YYYY-MM`) o una **ventana de fechas
arbitraria** (`--desde YYYY-MM-DD --hasta YYYY-MM-DD`, ambos inclusive); en los dos
casos se compara contra el **mismo período del año anterior**. Por defecto, el
último mes completo. *Si la fuente no cubre por completo la ventana del año
anterior, el interanual se suprime y el desvío cae al comparativo vs red.*

```bash
# Caso 1 — clasificar encuestas por tema y rutear el dominante al manual
python -m src.interface.cli clasificar-encuestas --tabla mala
python -m src.interface.cli clasificar-encuestas --limite 100 --mes 2026-04
python -m src.interface.cli clasificar-encuestas --json

# Caso 2 — diagnóstico de una sucursal
python -m src.interface.cli diagnosticar 0008C682 --mes 2026-04
python -m src.interface.cli diagnosticar 0008C682 --desde 2026-01-01 --hasta 2026-05-31
python -m src.interface.cli diagnosticar 0008C682 --json

# Caso 5 — plan de diagnóstico rankeado (Markdown), por mes o por ventana
python -m src.interface.cli plan-mensual --out plan_2026-04.md
python -m src.interface.cli plan-mensual --desde 2026-01-01 --hasta 2026-05-31
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

- **Ventas:** `df_ventas.parquet` (agregado diario por sucursal·producto) cargado
  a la tabla canónica `datos_ventas` con `python scripts/etl_ventas.py`. *La
  comparación interanual existe para 2026-01…05; meses/sucursales sin año previo
  caen al comparativo vs red.*
- **Encuestas:** `encuestas_satisfaccion.csv` / `encuestas_insatisfaccion.csv`
  (texto libre del cliente) cargadas a `encuestas_buena_experiencia` /
  `encuestas_mala_experiencia` con `python scripts/etl_encuestas.py`. Traen la
  columna `numero` (código de sucursal) que une directo con `datos_ventas.numero`;
  el `email` también está pero es nulo en ~50% de los casos.
- **Manual:** `manual_operativo_completo.md` chunkeado por Tema (con metadatos de
  SECCIÓN) e indexado. Si el denso FAISS quedara desincronizado con
  `meta.jsonl`/`bm25.pkl`, reconstruilo con:
  ```bash
  python scripts/rebuild_faiss.py
  ```

## Estructura

```
src/kpi/         KPIs determinísticos (metrics) y detección de desvíos (deviations)
src/agents/      caso1 (ClasificadorEncuestas), caso2 (DiagnosticadorPDV), caso5 (PlanificadorMensual), auditor_rag (CRAG)
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
- **vs red** como señal primaria; **interanual** y caídas por familia (línea
  comercial) como contexto/fallback.

## Fuera de alcance (fases siguientes)

Text-to-SQL en el camino de KPIs y tiers FAST/REASON. *El Caso 1 (clasificador /
router de encuestas) ya está implementado; el path Text-to-SQL existe como demo
(`scripts/diagnose_demo.py`) pero aún no está integrado a la CLI.*

---

# Apéndice — Diseño original y consigna (PLN III)

Esta sección preserva el diseño y el contexto académico del proyecto
(**Procesamiento de Lenguaje Natural III — FIUBA / CEIA**). Describe la *visión*
completa del sistema; donde difiere de lo implementado, se aclara qué es **MVP**
(en la CLI) y qué quedó como **diseño/diferido**.

## Objetivo y concepto de "gemelo digital"

Construir un **gemelo digital operativo (GDO)** por punto de venta (PDV) que cruce
*qué ocurrió en la realidad* (hechos: ventas y encuestas) contra *qué dice la norma*
(base de conocimiento: manual operativo + `recursos/`), produciendo un **diagnóstico
fundamentado con citación direccionable** y disparando una **acción automática**.

El gemelo se materializa como un **objeto de estado por PDV** que combina capas:

| Capa | Fuente | Naturaleza |
|---|---|---|
| Perfil estructural | diseño funcional de tiendas, capacidad de frío | Estático (configuración del local) |
| Hechos dinámicos | `df_ventas.parquet`, encuestas +/− | Transaccional / serie temporal |
| Norma aplicable | `manual_operativo_completo.md`, `recursos/` | Conocimiento gobernante (RAG) |

Pregunta típica: *"¿Por qué cayó la satisfacción en una sucursal?"* → detecta el
desvío en los datos, lo contrasta con el protocolo, y devuelve causa raíz + acción
correctiva citada.

## Arquitectura multi-agente (visión)

Patrón **ReAct sobre LangGraph** con un Supervisor que orquesta dos agentes
especializados y reconcilia realidad vs norma:

| Agente | Rol | Entrada | Salida | Mecanismo |
|---|---|---|---|---|
| Supervisor | Planificador / reconciliador | Consulta + estado del twin | Plan + diagnóstico final | ReAct + reflexión (CRAG) |
| Analista | Cuantificar la realidad | Pregunta NL | Tabla de hechos + SQL ejecutado | Text-to-SQL con validación de schema |
| Auditor Normativo | Recuperar y verificar la norma | Sub-pregunta + hechos | Protocolo + cita + veredicto cumple/no | RAG híbrido + grader |

**Comunicación:** estado compartido en LangGraph + mensajes estructurados
(`claim`, `evidence`, `source`, `confidence`). **Auto-evaluación:** grader de
recuperación + nodo de reflexión que verifica que cada afirmación del reporte esté
respaldada por evidencia citada (estilo Self-RAG).

> **MVP vs diseño.** La CLI implementa el camino **determinístico** (KPIs por SQL
> fijo → detección de desvíos por reglas → Auditor RAG → síntesis citada) para los
> Casos 1, 2 y 5. El **Supervisor + Analista Text-to-SQL** con reconciliación y
> ruteo de consulta vive como demo (`scripts/diagnose_demo.py`), aún no integrado a
> la CLI. El `TwinState` Pydantic y las señales de un modelo predictivo
> (churn/ventas) quedaron **diferidos**.

## RAG avanzado (Auditor Normativo)

Pipeline pre-retrieval → retrieval → post-retrieval:

1. **Ingesta/indexación:** chunking por estructura (headers markdown) con metadatos
   (`doc_id`, `section/tema`, `source`, `tipo_doc`). Índice **doble**: denso
   (`all-MiniLM-L6-v2`, coseno, FAISS local — Pinecone opcional) + **BM25** sobre los
   mismos chunks.
2. **Recuperación híbrida:** BM25 y denso en paralelo (top-k cada uno).
3. **Fusión RRF:** Reciprocal Rank Fusion `1/(k+rank)` sin normalizar puntajes.
4. **Re-ranking:** **Cross-Encoder** (MS MARCO MiniLM) sobre pares `(query, chunk)`.
5. **Diversidad:** `per_doc_cap` + dedup por `doc_id`.
6. **Contexto citable:** cada pasaje se arma con `[source, sección]` → **citación
   direccionable**.

**Variante:** **Agentic RAG + Corrective RAG (CRAG)** — el agente decide *cuándo*
recuperar y un grader clasifica el contexto `correcto / ambiguo / incorrecto`;
si es flojo, reescribe la query y reintenta.

## Seguridad

- **Control de acceso:** API key por header en FastAPI; tokens desde `.env`, sin hardcode.
- **Filtrado de inputs:** Text-to-SQL **read-only** sobre DuckDB + allow-list de
  tablas (previene inyección y operaciones destructivas) — ver `src/security/`.
- **Validación de outputs:** SQL validado contra el schema antes de ejecutar; salidas
  estructuradas en JSON; chequeo de "toda afirmación citada".
- **Auditoría:** logging estructurado de query, fuente recuperada y acción disparada.

## Optimización de costos y latencia

- **Model routing:** consultas simples → LLM liviano; reconciliación compleja →
  modelo mayor (en la práctica, Groq primario con respaldo Anthropic ante rate-limit).
- **Caching** de embeddings y respuestas frecuentes; **compresión de contexto**
  (resumen de chunks); **batching** de embeddings en la ingesta; control de
  `temperature`/`token_limit` por agente.
- **Ruteo eficiente del grafo:** los nodos de KPIs y detección son SQL+reglas; sin
  desvío relevante el grafo abstiene **sin gastar una llamada al LLM**.

## Acción final automática

Actuador de salida (`src/actions/notifier.py`): `LogNotifier` registra cada
diagnóstico como línea JSONL en un log externo. Extensible a **Email** (SMTP/Gmail) o
**Slack/Webhook** implementando la misma interfaz.

## Evaluación — taxonomía de métricas (referencia)

Más allá de los chequeos livianos (`src/eval/checks.py`) y las métricas IR
(`src/eval/metrics.py`), la batería completa de referencia abarca:

- **Retrieval:** Recall@k (meta ≥0.85@20), nDCG@k, MRR, Hit@k, diversidad/dedup, latencia p50/p95.
- **Contexto:** Context Precision/Recall, % de citación direccionable, compresión efectiva.
- **Generación:** EM/F1 sobre ground truth, Faithfulness/Attribution, hallucination rate, abstención correcta, % de SQL válido.
- **Re-rank/Fusión:** nDCG lift del Cross-Encoder; ganancia del híbrido (RRF) vs cada señal.
- **Agentes:** coherencia de decisiones, calidad del diálogo, redundancia evitada, latencia.
- **Robustez/seguridad:** caída ante paráfrasis/typos, OOD, filtrado de PII, resistencia a jailbreak.
- **Operación (SLOs):** latencia end-to-end p50/p95, costo por respuesta (tokens/$), tasa de errores.

*Set sugerido:* ~20-30 preguntas operativas con ground truth (chunk correcto +
respuesta esperada), generadas semiautomáticamente del manual y revisadas a mano.

## Stack tecnológico

| Componente | Herramienta |
|---|---|
| Orquestación de agentes | LangGraph / LangChain |
| LLM | Groq (primario) + Anthropic (respaldo) · Ollama (local) |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace) |
| Vector store | FAISS (local) · Pinecone (opcional) |
| Sparse retrieval | BM25 (`rank_bm25`) |
| Re-ranker | Cross-Encoder MS MARCO MiniLM |
| Datos / SQL | DuckDB + pandas |
| API / Front | FastAPI + Streamlit |
| Tests | pytest |

## Mapeo requisito → componente (consigna)

| Requisito de la consigna | Dónde se cumple | Estado |
|---|---|---|
| API funcional (FastAPI/Streamlit) | `src/interface/` | OK |
| ≥2 agentes con comunicación dinámica + auto-evaluación | Supervisor + Analista + Auditor; Casos 1/2/5 | OK |
| RAG (retriever-ranker, relevancia, trazabilidad) | `src/rag/` + Auditor (CRAG) | OK |
| Medidas de seguridad | `src/security/` (read-only SQL, allow-list, auth) | OK |
| Modelo preexistente para inferencia | Cross-Encoder de re-ranking (modelo preentrenado) · predictivo churn/ventas | Diferido / a confirmar |
| Flujo modular completo y monitoreable | Arquitectura + estructura de código | OK |
| Optimización de costos y latencia | Model routing, caching, abstención sin LLM | OK |
| Acción final automática | `src/actions/notifier.py` (LogNotifier) | OK |
| Evaluación y métricas | `src/eval/` (checks + métricas IR) | Parcial |
