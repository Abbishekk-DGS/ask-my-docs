"""Grounded answering: retrieve, build a cited prompt, stream the answer."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .chunker import chunk_pages
from .llm import ChatModel, Embedder
from .loaders import load_path
from .retriever import Hit, HybridRetriever
from .store import VectorStore

SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages.
Rules:
- Cite every claim with the passage number in square brackets, e.g. [2].
- If the context does not contain the answer, say "I couldn't find that in your documents."
- Be concise. Prefer bullet points for lists."""


def build_messages(question: str, hits: list[Hit], history: list[dict] | None = None) -> list[dict]:
    context = "\n\n".join(
        f"[{n}] ({hit.chunk.citation})\n{hit.chunk.text}" for n, hit in enumerate(hits, 1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *(history or [])[-6:],
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


@dataclass
class Answer:
    hits: list[Hit]
    tokens: Iterator[str]


class DocMind:
    def __init__(self, index_dir: Path, embedder: Embedder, chat: ChatModel):
        self.index_dir = index_dir
        self.embedder, self.chat = embedder, chat
        self.store = VectorStore.load(index_dir)
        self.retriever = HybridRetriever(self.store, embedder)

    def ingest(self, path: Path, max_words: int = 180, overlap_words: int = 40) -> int:
        chunks = chunk_pages(load_path(path), max_words, overlap_words)
        if not chunks:
            return 0
        added = self.store.add(chunks, self.embedder.embed([c.text for c in chunks]))
        self.store.save(self.index_dir)
        return added

    def ask(self, question: str, k: int = 5, history: list[dict] | None = None) -> Answer:
        hits = self.retriever.retrieve(question, k=k)
        if not hits:
            return Answer(
                hits=[], tokens=iter(["Your index is empty. Run `docmind ingest` first."])
            )
        return Answer(hits=hits, tokens=self.chat.stream(build_messages(question, hits, history)))
