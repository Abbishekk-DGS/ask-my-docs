"""Thin clients for the local Ollama server (chat + embeddings)."""

from __future__ import annotations

import json
import os
import zlib
from collections.abc import Iterator
from typing import Protocol

import httpx
import numpy as np

_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = _host if _host.startswith("http") else f"http://{_host}"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


class ChatModel(Protocol):
    def stream(self, messages: list[dict]) -> Iterator[str]: ...


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", url: str = OLLAMA_URL, batch: int = 32):
        self.model, self.url, self.batch = model, url, batch

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        with httpx.Client(timeout=120) as client:
            for i in range(0, len(texts), self.batch):
                resp = client.post(
                    f"{self.url}/api/embed",
                    json={"model": self.model, "input": texts[i : i + self.batch]},
                )
                resp.raise_for_status()
                vectors.extend(resp.json()["embeddings"])
        return normalize(np.asarray(vectors, dtype=np.float32))


class OllamaChat:
    def __init__(self, model: str = "qwen2.5:3b", url: str = OLLAMA_URL, temperature: float = 0.1):
        self.model, self.url, self.temperature = model, url, temperature

    def stream(self, messages: list[dict]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        with httpx.stream("POST", f"{self.url}/api/chat", json=payload, timeout=300) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                event = json.loads(line)
                if token := event.get("message", {}).get("content"):
                    yield token
                if event.get("done"):
                    break


class HashEmbedder:
    """Deterministic bag-of-words embedder. No model needed: useful for tests and offline demos."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        from .bm25 import tokenize

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, zlib.crc32(token.encode()) % self.dim] += 1.0
        return normalize(out)


def normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
