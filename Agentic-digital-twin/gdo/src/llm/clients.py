"""Clientes LLM intercambiables para el Analista.

- OllamaClient: para uso real con tu Ollama local (default).
- StaticSQLClient: stub determinista para demo/tests offline.
"""
from __future__ import annotations

import json
import os
import urllib.request


class BaseLLM:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaClient(BaseLLM):
    """Llama al endpoint /api/generate de Ollama (http://localhost:11434)."""

    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.0):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]


class StaticSQLClient(BaseLLM):
    """Stub para demo offline: mapea (palabras clave) -> SQL por reglas."""

    def __init__(self, rules: list[tuple[tuple[str, ...], str]]):
        self.rules = rules

    def complete(self, prompt: str) -> str:
        q = prompt.lower()
        for keys, sql in self.rules:
            if all(k in q for k in keys):
                return sql
        return "SELECT 'sin regla para esta pregunta en el stub' AS nota"
