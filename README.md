# NLP3-AgenticAI

Repositorio personal con los ejercicios prácticos vistos en clase del curso **Procesamiento de Lenguaje Natural 3 (NLP3)** de la **Maestría en Inteligencia Artificial (CEIA)** — FIUBA, UBA.

Cada subcarpeta corresponde a una clase del curso e incluye su propio `README.md` (o notebook) con la guía de configuración y ejecución específica.

---

## Proyectos

### `rag-C1/` — Pipeline RAG híbrido (Clase 1)

Sistema de **Recuperación Aumentada con Generación (RAG)** que combina:

- **Búsqueda léxica** con BM25.
- **Búsqueda semántica vectorial** con Pinecone.
- **Fusión de rankings** mediante Reciprocal Rank Fusion (RRF).
- **Re-ranqueo neural** con Cross-Encoder.

Flujo completo:

> Documentos → Chunks → Embeddings → Índices (BM25 + Pinecone) → Fusión RRF → Re-ranqueo Cross-Encoder → Contexto con citas → LLM

Ver [`rag-C1/README.md`](./rag-C1/README.md) para configuración (`.env`, corpus, namespaces de Pinecone) y comandos de ejecución.

### `guardrails-C2/` — Agente con guardrails (Clase 2)

Ejemplo didáctico de un **agente de extracción de datos estructurados** con defensa en profundidad:

- LLM local vía **Ollama** (`qwen3:8b` por defecto, con `format="json"`).
- Búsqueda web con **Tavily**.
- **Sanitización de inputs** y separación de instrucciones vs. contexto no confiable (patrón anti-prompt-injection con bloque `UNTRUSTED CONTEXT`).
- **Validación de salida** con `guardrails-ai` (`ValidJson` sobre un JSON Schema + `RegexMatch`, ambos `OnFailAction.EXCEPTION`).

Ver [`guardrails-C2/README.md`](./guardrails-C2/README.md) para detalle de las capas de defensa y configuración.

### `lang-C3/` — LangChain + LangGraph + LangSmith (Clase 3)

Notebook introductorio al stack **LangChain / LangGraph / LangSmith** con tools reales. Caso práctico: un asistente para vendedores que responde consultas combinando **Wikipedia** (explicar conceptos) con un **CSV interno de inventario** (precio, stock, disponibilidad).

Primero muestra el *loop* manual de tool-calling para entender la mecánica, y luego lo reorganiza como grafo en LangGraph (estado, nodos, ruteo y ciclos controlados), sin RAG ni vector stores.

Ver [`lang-C3/Lang_Snippets/lang_basicos_langchain_langgraph_langsmith.ipynb`](./lang-C3/Lang_Snippets/lang_basicos_langchain_langgraph_langsmith.ipynb).

### `Multiagent-C4/` — Sistema multiagente con LangGraph (Clase 4)

Sistema multiagente colaborativo (patrón **supervisor + workers**) para búsqueda y resumen de papers científicos, corriendo **all-Ollama**:

- **Supervisor** (`qwen3:14b`, reasoning ON): orquesta el flujo y decide qué agente invocar.
- **ResearchAssistant** (`qwen3:8b`, reasoning OFF): tools de **ArXiv** y **Wikipedia**.
- **SummarizerAgent**: resume el contenido recuperado.
- **SummarizationNode** (`langmem`): comprime el historial cuando supera un umbral de tokens.

Memoria de corto plazo con `InMemorySaver` / `InMemoryStore` por `thread_id`. Entrypoint interactivo (REPL) en `main.py`.

Ver [`Multiagent-C4/README.md`](./Multiagent-C4/README.md) para arquitectura, prerequisitos de Ollama y notas operativas (VRAM, cold start).

---

## Entorno

Cada proyecto puede correr con su propio virtualenv, o compartir el venv ubicado fuera del repo:

```
~/Documents/CEIA/PLN3/pln3
```

Activación desde cualquier directorio:

```bash
source ~/Documents/CEIA/PLN3/pln3/bin/activate
```

Los proyectos que declaran dependencias en un `requirements.txt` propio (por ejemplo `rag-C1`) se instalan con el entorno ya activado:

```bash
cd <proyecto>
pip install -r requirements.txt
```

`Multiagent-C4` usa un venv aislado (`.venv`) por sus dependencias de LangGraph/Ollama; ver su README.

> **Nota:** si dos proyectos requieren versiones incompatibles de una misma librería, conviene crear venvs separados por proyecto.

---

## Ejecución

Los scripts que forman parte de un paquete se invocan como **módulo** (`python -m ...`) desde la raíz del proyecto correspondiente, no como archivo suelto. Ejemplo:

```bash
cd ~/Documents/CEIA/PLN3/NLP3-AgenticAI/rag-C1
python -m main_test_scripts.rag_demo_pinecone
```

Esto es necesario para que Python resuelva correctamente los imports internos de cada paquete (`raglib`, `src`, etc.).

---

## Estructura

```
NLP3-AgenticAI/
├── README.md                       # este archivo
├── rag-C1/                         # Clase 1 — Pipeline RAG híbrido
│   ├── README.md
│   ├── requirements.txt
│   ├── corpus/
│   ├── data/
│   ├── raglib/
│   └── main_test_scripts/
├── guardrails-C2/                  # Clase 2 — Agente con guardrails
│   ├── README.md
│   └── agente_ejemplo_con_guardarails.py
├── lang-C3/                        # Clase 3 — LangChain/LangGraph/LangSmith
│   └── Lang_Snippets/
│       ├── lang_basicos_langchain_langgraph_langsmith.ipynb
│       └── inventario_productos.csv
└── Multiagent-C4/                  # Clase 4 — Sistema multiagente
    ├── README.md
    ├── main.py
    ├── agent_arch.png
    └── src/
        ├── agent/                  # supervisor, workers, prompts, estado
        └── tools/                  # arxiv, wikipedia
```

---

## Curso

- **Carrera:** Maestría en Inteligencia Artificial (CEIA)
- **Institución:** Facultad de Ingeniería — Universidad de Buenos Aires (FIUBA, UBA)
- **Materia:** Procesamiento de Lenguaje Natural 3 (NLP3)
