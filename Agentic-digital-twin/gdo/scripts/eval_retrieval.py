"""Evaluación de retrieval: Hit@k, MRR, nDCG@k, P@k sobre el ground truth.

Barato (no usa LLM): solo el HybridRetriever. Uso:
    python scripts/eval_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.metrics import (hit_at_k, is_relevant, mrr, ndcg_at_k,  # noqa: E402
                              precision_at_k)

K = 10
METRICS = ["hit@5", "hit@10", "mrr", "ndcg@10", "p@5"]


def run_eval(retriever, items: list[dict], k: int = K) -> list[dict]:
    rows = []
    for g in items:
        passages = retriever.retrieve(g["question"], top_n=k, per_doc_cap=k)
        rels = [1 if is_relevant(p.chunk, g["relevant"]) else 0 for p in passages]
        rows.append({"id": g["id"], "category": g.get("category", ""),
                     "hit@5": hit_at_k(rels, 5), "hit@10": hit_at_k(rels, 10),
                     "mrr": mrr(rels), "ndcg@10": ndcg_at_k(rels, 10),
                     "p@5": precision_at_k(rels, 5)})
    return rows


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return {m: round(sum(r[m] for r in rows) / len(rows), 3) for m in METRICS}


def main():
    from src.rag.retrieve import HybridRetriever
    gt = [json.loads(l) for l in (ROOT / "eval" / "ground_truth.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    items = [g for g in gt if g.get("relevant")]
    retriever = HybridRetriever.build_default(ROOT / "data" / "index")
    rows = run_eval(retriever, items)

    print(f"\n=== Retrieval ({len(rows)} preguntas, k={K}) ===")
    print(f"{'id':6} {'cat':12} " + " ".join(f"{m:>8}" for m in METRICS))
    for r in rows:
        print(f"{r['id']:6} {r['category']:12} " + " ".join(f"{r[m]:8.3f}" for m in METRICS))
    agg = aggregate(rows)
    print("-" * 60)
    print(f"{'PROMEDIO':19} " + " ".join(f"{agg[m]:8.3f}" for m in METRICS))

    out = ROOT / "eval" / "results_retrieval.json"
    out.write_text(json.dumps({"per_question": rows, "aggregate": agg},
                              ensure_ascii=False, indent=2))
    print(f"\nGuardado en {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
