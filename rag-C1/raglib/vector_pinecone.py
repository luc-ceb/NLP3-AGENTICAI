from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from typing import Dict, List, Tuple, Optional




# Convención de IDs de chunk: f"{doc_id}::chunk_{local_idx}"
def make_chunk_id(doc_id: str, local_idx: int) -> str:
    return f"{doc_id}::chunk_{local_idx}"

def parse_chunk_id(chunk_id: str) -> Tuple[str, int]:
    doc_id, rest = chunk_id.split("::", 1)
    i = int(rest.replace("chunk_", ""))
    return doc_id, i

def ensure_pinecone_index(
    pc: Pinecone,
    name: str,
    dim: int,
    cloud: str = "aws",
    region: str = "us-east-1",
    metric: str = "cosine",
):
    idxs = {i["name"]: i for i in pc.list_indexes().get("indexes", [])}
    if name not in idxs:
        pc.create_index(
            name=name,
            dimension=dim,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

@dataclass
class PineconeSearcher:
    """
    Maneja embeddings + upsert + query sobre Pinecone.
    Guarda registro local (chunk_id -> meta) para mapear resultados a texto y metadatos.
    """
    index_name: str
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    cloud: str = "aws"
    region: str = "us-east-1"
    api_key: Optional[str] = None
    namespace: str = "default"  # configurable

    def __post_init__(self):
        key = self.api_key or os.getenv("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("Falta PINECONE_API_KEY en entorno o parámetro api_key.")
        self.pc = Pinecone(api_key=key)
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_sentence_embedding_dimension()
        ensure_pinecone_index(self.pc, self.index_name, dim, cloud=self.cloud, region=self.region)
        self.index = self.pc.Index(self.index_name)
        # registro local: chunk_id -> dict(text, doc_id, local_idx, source, page)
        self.registry: Dict[str, Dict] = {}

    def _ns_vector_count(self) -> int:
        """Cantidad de vectores en la namespace actual."""
        try:
            stats = self.index.describe_index_stats()
            ns = getattr(stats, "namespaces", None) or stats.get("namespaces", {})
            if isinstance(ns, dict):
                node = ns.get(self.namespace, {})
                # Pinecone v3: {"vectorCount": N}
                if isinstance(node, dict):
                    return int(node.get("vectorCount", 0))
        except Exception as e:
            print(f"[WARN] describe_index_stats falló: {e}")
        return 0

    # método de instancia
    def clear_namespace(self):
        """
        Borra TODO en esta namespace si existe.
        Evita 404 'Namespace not found' en primeras corridas.
        """
        try:
            # describe_index_stats devuelve un dict con 'namespaces'
            stats = self.index.describe_index_stats()
            ns = getattr(stats, "namespaces", None) or stats.get("namespaces", {})
            if isinstance(ns, dict) and self.namespace in ns:
                self.index.delete(deleteAll=True, namespace=self.namespace)
            else:
                # No hay nada que borrar (namespace aún no creada)
                print(f"[INFO] Namespace '{self.namespace}' no existe todavía; skip clear.")
        except Exception as e:
            # No detengas la ejecución por esto; sólo avisa
            print(f"[WARN] clear_namespace saltado: {e}")

    def upsert_chunks(self, chunks_per_doc: Dict[str, List[str]], docs_meta: Dict[str, Dict]):
        vectors = []
        for doc_id, chunks in chunks_per_doc.items():
            if not chunks:
                continue
            embs = self.model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
            for i, (ch, v) in enumerate(zip(chunks, embs)):
                chunk_id = make_chunk_id(doc_id, i)
                meta = {
                    "doc_id": doc_id,
                    "local_idx": i,
                    "source": (docs_meta.get(doc_id, {}) or {}).get("source", ""),
                    "page": (docs_meta.get(doc_id, {}) or {}).get("page", None),
                    "text": ch,
                }
                self.registry[chunk_id] = meta
                vectors.append({"id": chunk_id, "values": v.tolist(), "metadata": meta})

        print(f"[UPSERT] index={self.index_name} ns={self.namespace} vectors={len(vectors)} (antes={self._ns_vector_count()})")
        B = 100
        for i in range(0, len(vectors), B):
            self.index.upsert(vectors=vectors[i:i+B], namespace=self.namespace)
        print(f"[UPSERT] después={self._ns_vector_count()} en ns={self.namespace}")


    def search(self, query: str, top_k: int = 50, meta_filter: Optional[dict] = None) -> List[Tuple[str, float, Dict]]:
        """
        Devuelve [(chunk_id, score, meta)], score mayor = más similar
        """
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        res = self.index.query(
            vector=q,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,     # usamos namespace
            filter=meta_filter,           # opcional: filtrar por source/page/etc.
        )
        out = []
        for m in res.matches:
            meta = self.registry.get(m.id) or (m.metadata or {})
            out.append((m.id, float(m.score), meta))
        return out
