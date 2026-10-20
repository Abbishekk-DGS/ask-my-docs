"""Hybrid retrieval: dense vectors + BM25, merged with Reciprocal Rank Fusion."""

from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .chunker import Chunk
from .llm import Embedder
from .store import VectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None
    sparse_rank: int | None


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion (Cormack et al., 2009). Robust to score scales that don't match."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, candidates: int = 30):
        self.store, self.embedder, self.candidates = store, embedder, candidates
        self._bm25: BM25 | None = None

    @property
    def bm25(self) -> BM25:
        if self._bm25 is None or len(self._bm25.doc_len) != len(self.store):
            self._bm25 = BM25([c.text for c in self.store.chunks])
        return self._bm25

    def retrieve(self, query: str, k: int = 5) -> list[Hit]:
        if not len(self.store):
            return []
        qvec = self.embedder.embed([query])[0]
        dense = [i for i, _ in self.store.search(qvec, self.candidates)]
        sparse_scores = self.bm25.scores(query)
        sparse = sorted(
            (i for i, s in enumerate(sparse_scores) if s > 0), key=lambda i: -sparse_scores[i]
        )[: self.candidates]
        fused = rrf([dense, sparse])
        ranked = sorted(fused, key=lambda i: -fused[i])[:k]
        return [
            Hit(
                chunk=self.store.chunks[i],
                score=fused[i],
                dense_rank=dense.index(i) + 1 if i in dense else None,
                sparse_rank=sparse.index(i) + 1 if i in sparse else None,
            )
            for i in ranked
        ]
