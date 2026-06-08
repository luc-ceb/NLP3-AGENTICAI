"""Evaluación end-to-end: exactitud de veredicto, abstención correcta y cita.

Usa LLM (cuesta API). Limitá con EVAL_N. Uso:
    EVAL_N=6 python scripts/eval_system.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_system_eval(supervisor, items: list[dict], verbose: bool = True) -> list[dict]:
    rows = []
    for i, g in enumerate(items, 1):
        if verbose:
            print(f"[{i}/{len(items)}] {g['id']}: {g['question'][:55]}...", flush=True)
        t0 = time.time()
        res = supervisor.diagnose(g["question"])
        dt = time.time() - t0
        verdicts = [a["verdict"] for a in res.audits] or ["sin_norma"]
        exp = g["expected_verdict"]
        if exp == "sin_norma":
            verdict_ok = all(v == "sin_norma" for v in verdicts)   # abstención correcta
        else:
            verdict_ok = exp in verdicts
        cite_ok = True
        if g.get("expected_source"):
            cite_ok = any(g["expected_source"].lower() in c.lower() for c in res.citations)
        if verbose:
            print(f"    -> {verdicts} ({dt:.1f}s)", flush=True)
        rows.append({"id": g["id"], "expected": exp, "got": verdicts,
                     "verdict_ok": verdict_ok, "cite_ok": cite_ok})
    return rows


def main():
    from src.interface.api import build_supervisor
    gt = [json.loads(l) for l in (ROOT / "eval" / "ground_truth.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    items = [g for g in gt if g.get("expected_verdict")]
    n = int(os.getenv("EVAL_N", str(len(items))))
    items = items[:n]

    supervisor = build_supervisor()
    rows = run_system_eval(supervisor, items)

    abst = [r for r in rows if r["expected"] == "sin_norma"]
    norm = [r for r in rows if r["expected"] != "sin_norma"]
    print(f"\n=== Sistema end-to-end ({len(rows)} preguntas) ===")
    for r in rows:
        print(f"  {r['id']}: esperado={r['expected']:10} got={r['got']} "
              f"veredicto={'OK' if r['verdict_ok'] else 'X'} cita={'OK' if r['cite_ok'] else 'X'}")
    print("-" * 60)
    if norm:
        print(f"Exactitud de veredicto: {sum(r['verdict_ok'] for r in norm)}/{len(norm)}")
        print(f"Cita correcta:          {sum(r['cite_ok'] for r in norm)}/{len(norm)}")
    if abst:
        print(f"Abstención correcta:    {sum(r['verdict_ok'] for r in abst)}/{len(abst)}")

    out = ROOT / "eval" / "results_system.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nGuardado en {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
