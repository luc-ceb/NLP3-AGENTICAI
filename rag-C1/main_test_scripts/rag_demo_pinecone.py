# 1) AÑADIR RAÍZ DEL PROYECTO AL PATH ANTES DE IMPORTAR raglib
import sys, os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2) DEPENDENCIAS
from dotenv import load_dotenv
from raglib import RagPipeline, PineconeSearcher
from raglib.loader_pdfs import folder_pdfs_to_documents, documents_to_chunks

# 3) CARGA .env DESDE LA RAÍZ (para PINECONE_API_KEY)
# Si lo movés a otra carpeta fija, solo cambiás el path antes de .env. 
# Ejemplo: si lo tenés en ./config/.env: ENV_PATH = Path(__file__).resolve().parents[1] / "config" / ".env"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

def main():
    # Configuración básica
    INDEX_NAME = os.getenv("PINECONE_INDEX", "pln3-index-rag-pdfs")
    CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    REGION = os.getenv("PINECONE_REGION", "us-east-1")
    MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    API_KEY = os.getenv("PINECONE_API_KEY")

    if not API_KEY:
        raise RuntimeError("No encontré PINECONE_API_KEY. Revisá tu .env en la raíz del proyecto.")

    # 1) Cargar PDFs desde ./corpus
    corpus_dir = Path("./corpus")
    if not corpus_dir.exists():
        raise FileNotFoundError("No existe la carpeta ./corpus. Creala y poné tus PDFs adentro.")

    docs = folder_pdfs_to_documents(corpus_dir, recursive=True)
    if not docs:
        raise RuntimeError("No encontré texto en PDFs dentro de ./corpus. Verificá que tengan texto seleccionable.")

    print(f"[INFO] Documentos (páginas con texto): {len(docs)}")
    # opcional: mostrar un resumen
    # for d in docs[:3]:
    #     print(" -", d.id, d.source, d.page, len(d.text), "chars")

    # 2) Chunking (usa el mismo que tu pipeline)
    chunks_map = documents_to_chunks(docs, max_tokens_chunk=300, overlap=80)
    total_chunks = sum(len(v) for v in chunks_map.values())
    print(f"[INFO] Total de chunks: {total_chunks}")

    # 3) Pinecone (sube embeddings + metadatos)
    searcher = PineconeSearcher(
        index_name=INDEX_NAME,
        model_name=MODEL,
        cloud=CLOUD,
        region=REGION,
        api_key=API_KEY,
        namespace="v2-200tok"  # cualquier nombre válido (minúsculas, dígitos, '-')-->Sugerencia: guardá el namespace en .env para no tocar código
                               # Si cambiás de modelo de embeddings (dimensión de vector), cambiá de índice (no solo de namespace).
    )

    ## NO CONCATENA, borra todo lo que hay en ese namespace de Pinecone y lo deja vacío, comentar si se busca concatenar y perservar. 
    searcher.clear_namespace()  #=========> IMPORTANTE

    docs_meta = {d.id: {"source": d.source, "page": d.page} for d in docs}
    searcher.upsert_chunks(chunks_map, docs_meta)


    print(f"[INFO] Upsert en Pinecone completado -> índice '{INDEX_NAME}'.")

    # 4) Armar Pipeline HÍBRIDO y consultar 
    pipeline = RagPipeline(
        docs=docs,
        pinecone_searcher=searcher,
        max_tokens_chunk=300,
        overlap=80,
        ce_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    # Escribí acá tu consulta (o reemplazala por input()).
    query = "How does ARENA implement decision traceability in a RAG pipeline?"
    print(f"\n[QUERY] {query}\n")

    results = pipeline.retrieve_and_rerank(query, top_retrieve=20, top_final=5)

    print("\n=== RESUMEN RAG ===")
    summary = pipeline.build_summary_context(results)
    print(summary)


    print("=== TOP CONTEXTOS CON CITAS (RERANKED) ===")
    for rank, (doc_id, chunk, meta, score) in enumerate(results, 1):
        cite = pipeline.format_citation(meta)
        print(f"{rank:02d}. {doc_id}  | score_ce={score:.4f}  | cite={cite}")
        print(f"    {chunk[:140]}...")

    print("\n=== CONTEXTO PARA PROMPT ===")
    print(pipeline.build_cited_context(results))

if __name__ == "__main__":
    main()







