"""Construye los chunks de la base de conocimiento y los guarda en JSONL.

Uso: python scripts/ingest_kb.py
Lee data/raw/ (recursivo), ignora xlsx/csv, y escribe data/processed/chunks.jsonl
"""
from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.ingest import build_chunks, save_jsonl  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    raw = ROOT / "data" / "raw"
    out = ROOT / "data" / "processed" / "chunks.jsonl"
    chunks = build_chunks(raw)
    save_jsonl(chunks, out)

    print(f"\n=== Resumen de ingesta ===")
    print(f"Total chunks: {len(chunks)}  ->  {out.relative_to(ROOT)}")
    if not chunks:
        return
    by_tipo = Counter(c.tipo_doc for c in chunks)
    by_src = Counter(c.source for c in chunks)
    avg = sum(c.n_chars for c in chunks) / len(chunks)
    print(f"Caracteres promedio por chunk: {avg:.0f}")
    print("\nPor tipo_doc:")
    for t, n in by_tipo.most_common():
        print(f"  {t:22s} {n}")
    print("\nPor archivo fuente:")
    for s, n in by_src.most_common():
        print(f"  {n:5d}  {s}")

    print("\n=== Ejemplo de chunk (manual) ===")
    sample = next((c for c in chunks if c.tipo_doc == "manual_operativo"), chunks[0])
    print(f"  chunk_id : {sample.chunk_id}")
    print(f"  source   : {sample.source}")
    print(f"  section  : {sample.section}")
    print(f"  tema     : {sample.tema}")
    print(f"  ts       : {sample.ts_start} - {sample.ts_end}")
    print(f"  texto    : {sample.text[:300]}...")


if __name__ == "__main__":
    main()
