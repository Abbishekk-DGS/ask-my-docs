"""A small persistent vector store: numpy matrix + JSON metadata. Fast enough for ~100k chunks."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunker import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk] | None = None, vectors: np.ndarray | None = None):
        self.chunks = chunks or []
        self.vectors = vectors if vectors is not None else np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Add chunks, skipping ones already indexed. Returns how many were new."""
        known = {c.id for c in self.chunks}
        keep = [i for i, c in enumerate(chunks) if c.id not in known]
        if not keep:
            return 0
        new_vectors = vectors[keep]
        self.vectors = (
            new_vectors if not len(self.chunks) else np.vstack([self.vectors, new_vectors])
        )
        self.chunks.extend(chunks[i] for i in keep)
        return len(keep)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.chunks):
            return []
        sims = self.vectors @ query_vector
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(json.dumps([asdict(c) for c in self.chunks]))

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        if not (directory / "chunks.json").exists():
            return cls()
        chunks = [Chunk(**c) for c in json.loads((directory / "chunks.json").read_text())]
        return cls(chunks, np.load(directory / "vectors.npy"))
