"""Wrapper de conveniencia para la CLI del GDO.

Equivale a `python -m src.interface.cli ...` pero ejecutable como script.

Ejemplos:
    python scripts/gdo.py plan-mensual
    python scripts/gdo.py diagnosticar A9D75316 --mes 2026-04
    python scripts/gdo.py eval --with-llm
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interface.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
