"""Okapi BM25 implemented from scratch, so keyword matches (IDs, names, acronyms) never get lost."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were "
    "will with what which who how why when where do does did can i you we they".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(d)) for d in docs]
        self.doc_len = [sum(c.values()) for c in self.doc_tokens]
        self.avg_len = (sum(self.doc_len) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for counts in self.doc_tokens:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for counts, length in zip(self.doc_tokens, self.doc_len, strict=True):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
            for term in terms:
                tf = counts.get(term)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            out.append(score)
        return out
