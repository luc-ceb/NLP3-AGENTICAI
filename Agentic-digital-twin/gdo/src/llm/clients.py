"""Clientes LLM intercambiables para los agentes.

- OllamaClient: uso real con Ollama local. Desactiva 'thinking' por defecto
  (clave para qwen3: evita cadenas de razonamiento largas que disparan timeouts)
  y usa un timeout amplio configurable (OLLAMA_TIMEOUT, default 300s).
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
    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.0, timeout: int | None = None,
                 think: bool | None = False, num_predict: int | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "300"))
        self.think = think           # False corta el <think> de qwen3 (más rápido)
        self.num_predict = num_predict

    def complete(self, prompt: str) -> str:
        options = {"temperature": self.temperature}
        if self.num_predict:
            options["num_predict"] = self.num_predict
        payload = {"model": self.model, "prompt": prompt, "stream": False, "options": options}
        if self.think is not None:
            payload["think"] = self.think
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
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
