"""Reconstruye el índice denso FAISS local desde los chunks YA indexados.

El `dense.faiss` quedó desincronizado con `meta.jsonl`/`bm25.pkl` (vectores de
una corrida anterior con otra cantidad de chunks). Para garantizar sincronía sin
re-chunkear, este script reconstruye SOLO `dense.faiss` + `dense.json` a partir
de las filas exactas de `meta.jsonl` (la fuente de verdad que comparten BM25 y el
denso). No toca `meta.jsonl` ni `bm25.pkl`.

Uso:
    python scripts/rebuild_faiss.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.index import build_dense_index  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
INDEX = ROOT / "data" / "index"


def main() -> int:
    meta_path = INDEX / "meta.jsonl"
    chunks = [json.loads(line) for line in
              meta_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(chunks)
    print(f"Reconstruyendo denso desde {meta_path} ({n} chunks)...")

    # Reconstruye dense.faiss + dense.json en el MISMO orden que meta (idx i <-> meta[i]).
    build_dense_index(chunks, INDEX)

    # Verificación de sincronía: ntotal del FAISS == filas de meta.
    import faiss
    idx = faiss.read_index(str(INDEX / "dense.faiss"))
    cfg = json.loads((INDEX / "dense.json").read_text())
    ok = idx.ntotal == n
    print(f"FAISS ntotal={idx.ntotal} | meta={n} | dim={cfg['dim']} | "
          f"modelo={cfg['model_name']}")
    print("OK: índice denso sincronizado." if ok else "ERROR: sigue desincronizado.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
