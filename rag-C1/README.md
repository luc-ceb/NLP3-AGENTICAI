# Guía de uso — Pipeline RAG híbrido (BM25 + Pinecone + Cross-Encoder)

Este proyecto implementa un **sistema de Recuperación Aumentada con Generación (RAG)**.
El objetivo es enriquecer a un LLM (modelo de lenguaje) con información proveniente de un corpus propio, combinando **búsqueda léxica**, **búsqueda semántica vectorial** y **re-ranqueo neural**.

---

## 1. Conceptos básicos

* **RAG (Retrieval-Augmented Generation):** técnica que mejora la generación de un LLM al proveerle *contexto externo* (documentos relevantes).
* **BM25:** algoritmo clásico de recuperación léxica basado en frecuencia de términos y normalización por longitud.
* **Embeddings vectoriales:** representaciones densas de texto en un espacio numérico; permiten búsquedas semánticas (similitud de significado).
* **Pinecone:** base de datos vectorial en la nube, diseñada para almacenar embeddings y recuperar vecinos cercanos.
* **Cross-Encoder:** modelo que evalúa pares *(query, documento)* para reordenar candidatos y mejorar precisión.
* **RRF (Reciprocal Rank Fusion):** técnica simple y robusta para fusionar listas ordenadas de distintos motores de búsqueda.

El flujo completo es:

> **Documentos → Chunks → Embeddings → Índices (BM25 + Pinecone) → Fusión RRF → Re-ranqueo Cross-Encoder → Contexto con citas → LLM**

---

## 2. Requisitos e instalación

### 2.1 Dependencias

Puede instalar librerías directamente:

```bash
pip install sentence-transformers rank-bm25 pinecone pdfplumber torch pandas scikit-learn python-dotenv
```

O usar el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

> Recomendado: use un entorno virtual (`venv`, `conda`) para aislar dependencias.

---

## 3. Configuración de credenciales

Cree un archivo `.env` en la raíz del proyecto:

```env
PINECONE_API_KEY=tu_api_key
PINECONE_INDEX=pln3-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
OPENAI_API_KEY=tu_api_key_openai
```

* **PINECONE_INDEX:** si no existe, se creará automáticamente.
* **OPENAI_API_KEY:** necesario solo para resúmenes (`rag_summary.py`).

> ⚠️ Asegúrese de que `.env` esté incluido en `.gitignore` para no versionar credenciales.

---

## 4. Preparación del corpus

Coloque sus PDFs en `./corpus/`.
Cada **página** se procesa como un `Document` y luego se divide en *chunks* de ~200 tokens con solapamiento (~60 tokens).

---

## 5. Ingesta y construcción del índice vectorial

> Si de entrada se busca trabajar con un namespace específico, hace falta agregarlo (por ejemplo: `"v2-200tok"`, que es el que contiene el siguiente código).

Ejecute desde la raíz del proyecto:

```bash
python -m main_test_scripts.build_pinecone_index
```

Este script:

1. Lee PDFs de `./corpus`.
2. Limpia texto y normaliza caracteres.
3. Divide en *chunks* manejables.
4. Inserta embeddings en Pinecone con metadatos (`source`, `page`).

---

## 6. Consulta híbrida con re-ranqueo

> Si se usa el espacio ya generado, se requiere comentar el `clear_namespace()`. Este código contiene un namespace.

Ejemplo de búsqueda:

```bash
python -m main_test_scripts.rag_demo_pinecone
```

Flujo:

1. Recupera candidatos por **BM25** (léxico) y **Pinecone** (vectorial).
2. Combina resultados con **RRF**.
3. Re-ranquea candidatos con un **Cross-Encoder**.
4. Devuelve los fragmentos más relevantes con citas `[source, p. X]`.
5. Construye un contexto que puede alimentarse directamente a un LLM (ej. GPT-4).

---

## 7. Evaluación del rendimiento

Si dispone de juicios de relevancia:

```bash
python -m main_test_scripts.evaluate_retrieval --docs data/docs_sample_en.jsonl --qrels data/qrels_sample_en.csv
```

* **docs.jsonl:** corpus serializado (un documento por línea).
* **qrels.csv:** archivo de evaluación con columnas `query, doc_id, label`.

Métricas calculadas:

* **Precision@k:** fracción de resultados relevantes en el top-k.
* **Recall@k:** fracción de documentos relevantes recuperados.
* **nDCG:** ganancia acumulada normalizada (prioriza orden).
* **MRR:** reciprocidad de la primera respuesta correcta.

---

## 8. Estructura del proyecto

```
project/
├─ requirements.txt
├─ .env
│
├─ corpus/                      # PDFs fuente
├─ data/                        # Corpus serializado y qrels
│
├─ raglib/                      # Librería del pipeline
│  ├─ __init__.py
│  ├─ documents.py              # Document + tokenización + chunking
│  ├─ bm25_index.py             # Índice BM25
│  ├─ vector_pinecone.py        # Conexión a Pinecone
│  ├─ fusion.py                 # RRF
│  ├─ reranker.py               # Cross-Encoder
│  ├─ pipeline.py               # Orquesta todo el flujo
│  ├─ rag_summary.py            # Generación de resumen vía OpenAI
│  ├─ metrics.py                # Métricas IR
│  ├─ io_utils.py               # Entrada/salida JSONL y qrels
│  └─ loader_pdfs.py            # Loader de PDFs
│
└─ main_test_scripts/           # Scripts ejecutables
   ├─ __init__.py
   ├─ build_pinecone_index.py   # Ingesta a Pinecone
   ├─ rag_demo_pinecone.py      # Demo de consulta híbrida
   └─ evaluate_retrieval.py     # Evaluación
```

> **Nota sobre la ejecución:** los scripts se invocan con `python -m main_test_scripts.<nombre_script>` (sintaxis de módulo) en lugar de `python main_test_scripts/<nombre_script>.py`. Esto es necesario para que Python resuelva correctamente el import de `raglib`.
>
> Requisito: los directorios `raglib/` y `main_test_scripts/` deben contener un archivo `__init__.py` (puede estar vacío) para ser reconocidos como paquetes.

---

## 9. Explicación de módulos

* **BM25 (`bm25_index.py`):** búsqueda rápida por coincidencia de términos.
* **Pinecone (`vector_pinecone.py`):** búsqueda semántica por similitud de embeddings.
* **RRF (`fusion.py`):** fusión robusta de rankings, evita depender de un solo motor.
* **Cross-Encoder (`reranker.py`):** ajusta la lista final con precisión neural. Ver: https://www.sbert.net/
* **Pipeline (`pipeline.py`):** une todas las piezas y construye el contexto para el LLM.
* **Resúmenes (`rag_summary.py`):** opcional, genera resúmenes citados con OpenAI.
* **Métricas (`metrics.py`):** permite comparar distintas configuraciones y medir mejora tras el re-rankeo.

---

## 10. Diagnóstico rápido

* **Corpus vacío:** asegúrese de que `./corpus` contiene PDFs con texto (no solo escaneados como imagen).
* **Error de Pinecone (`UnauthorizedException 401`):** revise API Key, región e índice en `.env`. Si la variable `PINECONE_API_KEY` está exportada en el sistema, puede pisar la del `.env` — verifique con `echo $PINECONE_API_KEY` y use `unset PINECONE_API_KEY` si es necesario.
* **`ModuleNotFoundError: No module named 'raglib'`:** verifique que ejecuta los scripts con `python -m main_test_scripts.<script>` desde la raíz del proyecto, y que `raglib/__init__.py` existe.
* **Resultados pobres:** ajuste tamaño de chunk y solapamiento; incremente `top_k`.
* **Demasiada latencia:** reduzca `top_retrieve` o use un Cross-Encoder más ligero.
* **Costos elevados en Pinecone:** deshabilite almacenar `text` completo en metadatos y use solo previews.
