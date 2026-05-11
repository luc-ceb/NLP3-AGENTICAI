from typing import Dict, List, Tuple, Optional
from .documents import Document, chunk_text
from .bm25_index import BM25Index
from .reranker import CrossEncoderReranker
from .fusion import rrf_combine
from .vector_pinecone import PineconeSearcher, make_chunk_id, parse_chunk_id
from .rag_summary import generar_rag_summary


class RagPipeline:
    """
    Pipeline híbrido: BM25 local + (opcional) Pinecone vectorial + CrossEncoder (re-ranking).
    """

    def __init__(
        self,
        docs: List[Document],
        pinecone_searcher: Optional[PineconeSearcher] = None,
        max_tokens_chunk: int = 400,
        overlap: int = 100,
        ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        do_upsert: bool = True,  # si hay pinecone_searcher=True, controla si se suben los chunks
    ):
        # Mapa rápido por id
        self.docs = {d.id: d for d in docs}
        self.doc_list = docs

        # Chunking con fallback (no perder páginas cortas)
        self.chunks_per_doc: Dict[str, List[str]] = {}
        for d in docs:
            chunks = chunk_text(d.text, max_tokens_chunk, overlap)
            if not chunks:
                txt = " ".join((d.text or "").split())
                if txt:
                    words = txt.split()
                    if len(words) > max_tokens_chunk:
                        words = words[:max_tokens_chunk]
                    chunks = [" ".join(words)]
                else:
                    chunks = []
            self.chunks_per_doc[d.id] = chunks

        # Índice BM25 (sobre los mismos chunks)
        self.bm25 = BM25Index(docs, self.chunks_per_doc)

        # Vector search 
        self.vec = pinecone_searcher
        if self.vec is not None and do_upsert:
            docs_meta = {d.id: {"source": d.source, "page": d.page} for d in docs}
            self.vec.upsert_chunks(self.chunks_per_doc, docs_meta)

        # Re-ranker
        self.reranker = CrossEncoderReranker(model_name=ce_model, device=device)

        # Índices globales (para mapear BM25 -> (doc_id, idx_local))
        self.global_chunks: List[str] = []
        self.global_map: List[Tuple[str, int]] = []
        for d in docs:
            for i, ch in enumerate(self.chunks_per_doc[d.id]):
                self.global_chunks.append(ch)
                self.global_map.append((d.id, i))

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 50,
        meta_filter: Optional[dict] = None,
    ) -> List[str]:
        """
        Devuelve lista de chunk_ids (doc_id::chunk_i) por ranking fusionado.
        """
        # BM25
        bm25_hits: List[str] = []
        for gi, _score in self.bm25.search(query, top_k=top_k):
            doc_id, local_i = self.global_map[gi]
            bm25_hits.append(make_chunk_id(doc_id, local_i))

        # Vector
        vec_hits: List[str] = []
        if self.vec is not None:
            vec_res = self.vec.search(query, top_k=top_k, meta_filter=meta_filter)
            vec_hits = [cid for (cid, _s, _m) in vec_res]

        # Fusión (si no hay vector, usa solo BM25)
        if vec_hits:
            combined = rrf_combine(bm25_hits, vec_hits)
        else:
            combined = bm25_hits

        return combined[:top_k]

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 20,
        per_doc_cap: int = 2,
        meta_filter: Optional[dict] = None,
    ) -> List[Tuple[str, str, Dict]]:
        """
        Devuelve [(doc_id, chunk_text, meta)] con límite por documento para favorecer diversidad.
        """
        cids = self.retrieve_hybrid(query, top_k=top_k * 3, meta_filter=meta_filter)
        out: List[Tuple[str, str, Dict]] = []
        seen: Dict[str, int] = {}
        for cid in cids:
            doc_id, local_i = parse_chunk_id(cid)
            seen[doc_id] = seen.get(doc_id, 0)
            if seen[doc_id] >= per_doc_cap:
                continue
            seen[doc_id] += 1

            ch = self.chunks_per_doc[doc_id][local_i]
            # Metadatos: primero del registro local (si hay vector); si no, a partir del doc
            if self.vec is not None:
                meta = self.vec.registry.get(cid, {})
            else:
                d = self.docs[doc_id]
                meta = {"doc_id": doc_id, "local_idx": local_i, "source": d.source, "page": d.page}

            out.append((doc_id, ch, meta))
            if len(out) >= top_k:
                break
        return out

    def retrieve_and_rerank(self, query: str, top_retrieve: int = 30, top_final: int = 5):
        cand = self.retrieve_with_metadata(query, top_k=top_retrieve)
        reranked = self.reranker.rerank(query, cand)
        return reranked[:top_final]

    def build_summary_context(self, reranked: List[Tuple[str, str, Dict, float]]) -> str:
        docs = []
        for _, chunk, meta, _ in reranked:
            docs.append(
                {
                    "source": meta.get("source") or meta.get("doc_id"),
                    "page": meta.get("page", 1),
                    "text": chunk,
                }
            )
        # Intentar resumen; si falla (sin API key), devolver concatenación
        try:
            return generar_rag_summary(docs)
        except Exception:
            return "\n\n---\n\n".join(
                f"{d['text']}\n[{d['source']}" + (f", p. {d['page']}]" if d.get("page") else "]")
                for d in docs
            )

    @staticmethod
    def build_cited_context(reranked: List[Tuple[str, str, Dict, float]]) -> str:
        lines = []
        for _, chunk, meta, _ in reranked:
            src = meta.get("source") or meta.get("doc_id")
            page = meta.get("page")
            cite = f"[{src}" + (f", p. {page}]" if page is not None else "]")
            lines.append(f"{chunk}\n{cite}")
        return "\n\n---\n\n".join(lines)

    @staticmethod
    def format_citation(meta: Dict) -> str:
        src = meta.get("source") or meta.get("doc_id")
        page = meta.get("page")
        return f"[{src}" + (f", p. {page}]" if page is not None else "]")
