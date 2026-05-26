# Multiagent C4 — Sistema Multiagente con LangGraph

Implementación de un sistema multiagente colaborativo basado en **LangGraph** para búsqueda y resumen de papers científicos.

---

## Arquitectura

![Arquitectura del sistema multiagente](Codigo/agent_arch.png)

El sistema implementa un patrón **supervisor + workers**:

- **Supervisor**: orquesta el flujo, decide qué agente invocar según la query del usuario, y consolida los resultados. Utiliza un modelo de razonamiento (`qwen3:14b`).
- **ResearchAssistant**: agente equipado con tools de **ArXiv** y **Wikipedia** para recuperar información actualizada.
- **SummarizerAgent**: agente especializado en resumir contenido recibido del Research Assistant.
- **SummarizationNode**: comprime el historial conversacional cuando supera un umbral de tokens, para mantener el contexto manejable.

La memoria de corto plazo se mantiene con `InMemorySaver` y `InMemoryStore`, identificada por un `thread_id` por sesión.

**Justificación de los modelos elegidos:**

- `qwen3:14b` en el supervisor: el rol de coordinación requiere razonamiento explícito para decidir delegación. Con 16 GB de VRAM, usar 14b (~9 GB) permite que ambos modelos (supervisor + worker) convivan sin swapping, mejorando latencia. Alternativa (más potencia, más lentitud): `qwen3:30b-a3b` requiere ~18 GB y causa swapping frecuente entre supervisor y workers.
- `qwen3:8b` en los workers: los agentes que ejecutan tools o tareas concretas funcionan mejor sin thinking — los tokens de razonamiento intercalados pueden interferir con el parsing de tool calls.
- Thinking explícitamente desactivado en workers con doble candado (`reasoning=False` + `model_kwargs={"think": False}`) por el [bug conocido](https://github.com/langchain-ai/langchain/issues/33993) donde Qwen3 ignora `reasoning=False` en `langchain-ollama`.

---

## Prerequisitos

- **Python 3.11+**
- **Ollama** corriendo en `localhost:11434`
- Modelos descargados:
  ```bash
  ollama pull qwen3:14b
  ollama pull qwen3:8b
  ```
- GPU recomendada con ≥12 GB VRAM (ambos modelos entran sin swapping: 14b ≈9 GB + 8b ≈3-5 GB). En su defecto, ver [notas operativas](#notas-operativas).

---

## Setup

```bash
cd Codigo

# Crear entorno virtual aislado
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -U pip
pip install -r requirements.txt
```

> **Nota:** `requirements.txt` incluye `appnope` (macOS-only) que en Linux genera un warning pero no es bloqueante. Si alguna dep falla, las críticas para el funcionamiento son: `langgraph`, `langgraph-supervisor`, `langchain`, `langchain-community`, `langchain-ollama`, `langmem`, `python-dotenv`, `arxiv`, `wikipedia`.

---

## Ejecución

Asegurate de que Ollama esté corriendo:

```bash
systemctl --user status ollama   # o `ollama serve` manualmente
```

Desde `Codigo/` con el venv activado:

```bash
python main.py
```
Verás un prompt interactivo:

```
Research Assistant (LangGraph + Ollama)
  Supervisor: qwen3:14b (reasoning ON)
  Workers:    qwen3:8b (reasoning OFF)
Type 'exit' to quit.

Query>
```

### Queries de ejemplo

| Caso | Query | Agentes invocados |
|---|---|---|
| Concept lookup | `What is reinforcement learning from human feedback?` | Wikipedia |
| Búsqueda de papers | `Find recent papers about mixture of experts language models` | ArXiv |
| Flujo completo | `Find the latest papers on retrieval augmented generation and summarize the main approaches` | ArXiv → Summarizer |
| Memoria conversacional | (1) `Find papers about BG/NBD models` → (2) `Summarize the most cited one` | ArXiv → Summarizer con contexto persistente |

---

## Estructura del proyecto

```
Multiagent-C4/
├── README.md
└── Codigo/
    ├── main.py                  # Entrypoint (REPL interactivo)
    ├── requirements.txt
    ├── agent_arch.png           # Diagrama de arquitectura
    ├── dockerfile               # (opcional, no usado en setup local)
    ├── dockerfile.uv            # (opcional, variante con uv)
    └── src/
        ├── __init__.py
        ├── agent/
        │   ├── __init__.py
        │   ├── agent.py         # Definición de supervisor y workers
        │   ├── prompt.py        # System prompts
        │   ├── state.py         # Schema del estado de LangGraph
        │   └── .env             # GROQ_API_KEY (no usada en esta versión)
        └── tools/
            ├── __init__.py
            ├── arxiv/           # Tool de búsqueda en ArXiv
            └── wikipedia/       # Tool de búsqueda en Wikipedia
```

---

## Notas operativas

### VRAM y swapping de modelos

Con la configuración actual (**qwen3:14b** supervisor + **qwen3:8b** workers), ambos modelos caben en VRAM sin swapping (~14 GB total en GPU de 16 GB), evitando latencia por recargas.

### Cold start vs warm

La primera query tras arrancar Ollama tiene una pausa de 10-30 s mientras los pesos se cargan a VRAM. Una vez warm, las queries siguientes tienen solo la latencia de inferencia. Por defecto Ollama mantiene un modelo cargado 5 min tras la última invocación.

### Inspeccionar el reasoning del supervisor

Con `reasoning=True`, el chain-of-thought no aparece en el stream principal — queda en `additional_kwargs["reasoning_content"]` de los chunks. Para verlo en consola, ampliar el handler en `query()` de `agent.py`:

```python
if kind == "on_chat_model_stream":
    chunk = event["data"].get("chunk")
    if chunk:
        reasoning = chunk.additional_kwargs.get("reasoning_content")
        if reasoning:
            print(f"[think] {reasoning}", end="", flush=True)
        if chunk.content:
            yield chunk.content
```

---

## Referencias

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Repositorio oficial PLN3 - CEIA FIUBA](https://github.com/FIUBA-Posgrado-Inteligencia-Artificial/PLN3)
- [Ollama Thinking API](https://docs.ollama.com/capabilities/thinking)
- [langchain-ollama ChatOllama reference](https://reference.langchain.com/python/langchain-ollama/chat_models/ChatOllama)

---

## Licencia y créditos

Material académico basado en el contenido de la cátedra NLP3 — CEIA, Facultad de Ingeniería, Universidad de Buenos Aires. Las modificaciones son para uso personal en el marco de la cursada.
