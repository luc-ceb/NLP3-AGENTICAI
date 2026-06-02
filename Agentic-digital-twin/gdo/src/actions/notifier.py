"""Acción de salida automática del sistema.

Por ahora: LogNotifier (registra el diagnóstico en un log externo JSONL).
Extensible: agregar EmailNotifier / SlackNotifier implementando la misma interfaz.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


class Notifier:
    def send(self, result) -> dict:
        raise NotImplementedError


class LogNotifier(Notifier):
    """Registra cada diagnóstico como una línea JSON en un log externo."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or os.getenv("GDO_LOG", "data/logs/diagnoses.jsonl"))
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, result) -> dict:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "question": result.question,
            "diagnosis": result.diagnosis,
            "sql": getattr(result, "sql", ""),
            "audits": getattr(result, "audits", []),
            "citations": getattr(result, "citations", []),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("Diagnóstico registrado en %s", self.path)
        return record
