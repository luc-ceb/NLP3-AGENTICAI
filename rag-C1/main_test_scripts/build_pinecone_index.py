# Ingesta de PDFs → chunking → upsert a Pinecone
import os
from pathlib import Path
from dotenv import load_dotenv
from raglib import Document
from raglib.loader_pdfs import folder_pdfs_to_documents, documents_to_chunks
from raglib.vector_pinecone import PineconeSearcher

if __name__ == "__main__":
    load_dotenv(override=True)  # opcional
    # Config
    INDEX_NAME = os.getenv("PINECONE_INDEX", "pln3-index")
    CLOUD = os.getenv("PINECONE_CLOUD", "aws")
    REGION = os.getenv("PINECONE_REGION", "us-east-1")
    MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    corpus_dir = Path("./corpus")  # coloca tus PDFs aquí
    docs = folder_pdfs_to_documents(corpus_dir, recursive=True)
    print(f"Docs (páginas con texto): {len(docs)}")

    chunks_map = documents_to_chunks(docs, max_tokens_chunk=300, overlap=80)
    total_chunks = sum(len(v) for v in chunks_map.values())
    print(f"Total chunks: {total_chunks}")

    # armar meta por doc
    docs_meta = {d.id: {"source": d.source, "page": d.page} for d in docs}

    # construir searcher y upsert
    searcher = PineconeSearcher(index_name=INDEX_NAME, model_name=MODEL, cloud=CLOUD, region=REGION)
    searcher.upsert_chunks(chunks_map, docs_meta)  #(upsert = update + insert)--->Pinecone solo pisa datos si el "id" coincide
    print("Upsert a Pinecone completado.")
