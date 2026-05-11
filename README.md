# NLP3-AgenticAI

Repositorio personal con los ejercicios prácticos vistos en clase del curso **Procesamiento de Lenguaje Natural 3 (NLP3)** de la **Maestría en Inteligencia Artificial (CEIA)** — FIUBA, UBA.

Cada subcarpeta corresponde a una clase del curso e incluye su propio `README.md` con la guía de configuración y ejecución específica.

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
- **Validación de salida** con `guardrails-ai` (JSON Schema + `RegexMatch`).

Ver [`guardrails-C2/README.md`](./guardrails-C2/README.md) para detalle de las capas de defensa y configuración.

---

## Entorno

Todos los proyectos de este repositorio comparten el mismo virtualenv ubicado fuera del repo:

```
~/Documents/CEIA/PLN3/pln3
```

Activación desde cualquier directorio:

```bash
source ~/Documents/CEIA/PLN3/pln3/bin/activate
```

Cada proyecto declara sus dependencias en su propio `requirements.txt`. Para incorporarlas al venv compartido, con el entorno ya activado:

```bash
cd <proyecto>
pip install -r requirements.txt
```

> **Nota:** si dos proyectos requieren versiones incompatibles de una misma librería, conviene crear venvs separados por proyecto.

---

## Ejecución

Los scripts de cada proyecto se invocan como **módulo** (`python -m ...`) desde la raíz del proyecto correspondiente, no como archivo suelto. Ejemplo:

```bash
cd ~/Documents/CEIA/PLN3/NLP3-AgenticAI/rag-C1
python -m main_test_scripts.rag_demo_pinecone
```

Esto es necesario para que Python resuelva correctamente los imports internos de cada paquete (`raglib`, etc.).

---

## Estructura

```
NLP3-AgenticAI/
├── README.md              # este archivo
├── rag-C1/                # Clase 1 — Pipeline RAG híbrido
│   ├── README.md
│   ├── requirements.txt
│   ├── corpus/
│   ├── data/
│   ├── raglib/
│   └── main_test_scripts/
└── guardrails-C2/         # Clase 2 — Agente con guardrails
    ├── README.md
    ├── requirements.txt
    └── agente_ejemplo_con_guardarails.py
```

---

## Curso

- **Carrera:** Maestría en Inteligencia Artificial (CEIA)
- **Institución:** Facultad de Ingeniería — Universidad de Buenos Aires (FIUBA, UBA)
- **Materia:** Procesamiento de Lenguaje Natural 3 (NLP3)
