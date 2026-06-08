"""Clientes LLM intercambiables para los agentes.

- AnthropicClient: inferencia online vía la API de Anthropic (Messages).
- GroqClient: inferencia online vía Groq.
- OllamaClient: inferencia local vía Ollama (fallback de desarrollo).
- StaticSQLClient: stub determinista para tests offline.
- make_llm(): factory que elige proveedor (anthropic|groq|ollama).
"""
from __future__ import annotations

import json
import os
import urllib.request


class BaseLLM:
    def complete(self, prompt: str) -> str:
        raise NotImplementedError


class AnthropicClient(BaseLLM):
    """Inferencia online vía la API de Anthropic (Messages)."""

    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 max_tokens: int | None = None, api_key: str | None = None):
        from anthropic import Anthropic  # carga perezosa
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.temperature = temperature
        self.max_tokens = max_tokens or int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024"))
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def complete(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,        # requerido por la API de Anthropic
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


class GroqClient(BaseLLM):
    """Inferencia online vía Groq (chat completions)."""

    def __init__(self, model: str | None = None, temperature: float = 0.0,
                 max_tokens: int | None = None, api_key: str | None = None):
        from groq import Groq  # carga perezosa
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = temperature
        self.max_tokens = max_tokens or int(os.getenv("GROQ_MAX_TOKENS", "1024"))
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def complete(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""


class OllamaClient(BaseLLM):
    """Inferencia local vía Ollama. think=False corta el <think> de qwen3."""

    def __init__(self, model: str | None = None, host: str | None = None,
                 temperature: float = 0.0, timeout: int | None = None,
                 think: bool | None = False, num_predict: int | None = None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "300"))
        self.think = think
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


def make_llm(model: str | None = None, provider: str | None = None, **kw) -> BaseLLM:
    provider = (provider or os.getenv("LLM_PROVIDER", "groq")).lower()
    if provider == "anthropic":
        return AnthropicClient(model=model, **kw)
    if provider == "groq":
        return GroqClient(model=model, **kw)
    if provider == "ollama":
        return OllamaClient(model=model, **kw)
    raise ValueError(f"proveedor desconocido: {provider!r} (usá 'anthropic', 'groq' u 'ollama')")
