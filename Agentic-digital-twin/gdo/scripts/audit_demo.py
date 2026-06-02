"""Demo del Auditor Normativo (producción: usa índices + Ollama).

Uso (en tu máquina, con índices construidos y Ollama corriendo):
    USE_OLLAMA=1 OLLAMA_MODEL=qwen3:14b python scripts/audit_demo.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.retrieve import HybridRetriever      # noqa: E402
from src.agents.auditor_rag import NormativeAuditor  # noqa: E402
from src.llm.clients import OllamaClient           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

HECHOS = [
    "En hora pico la espera promedio en la sucursal Centro es de 6 minutos.",
    "La chocolatera se mantiene a 35 grados centígrados.",
    "Las bateas de helado permanecen abiertas alrededor de 60 segundos.",
]

if __name__ == "__main__":
    retriever = HybridRetriever.build_default(ROOT / "data" / "index")
    auditor = NormativeAuditor(retriever, OllamaClient(), max_retries=1)
    for hecho in HECHOS:
        print("\n" + "=" * 80)
        print(auditor.audit(hecho))
