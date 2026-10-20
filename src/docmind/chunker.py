"""Structure-aware chunking: split on paragraphs first, then sentences, then words."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .loaders import Page

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: str
    source: str
    page: int | None
    text: str

    @property
    def citation(self) -> str:
        name = self.source.rsplit("/", 1)[-1]
        return f"{name} p.{self.page}" if self.page else name


def _split_units(text: str, max_words: int) -> list[str]:
    """Break text into units no longer than max_words, preferring natural boundaries."""
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        if len(para.split()) <= max_words:
            units.append(para)
            continue
        for sentence in _SENTENCE.split(para):
            words = sentence.split()
            for i in range(0, len(words), max_words):
                units.append(" ".join(words[i : i + max_words]))
    return units


def chunk_pages(pages: list[Page], max_words: int = 180, overlap_words: int = 40) -> list[Chunk]:
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")
    chunks: list[Chunk] = []
    # Units leave room for the carried-over overlap, so no chunk exceeds max_words.
    unit_words_max = max_words - overlap_words
    for page in pages:
        buffer: list[str] = []
        for unit in _split_units(page.text, unit_words_max):
            unit_words = unit.split()
            if buffer and len(buffer) + len(unit_words) > max_words:
                chunks.append(_make_chunk(page, buffer))
                buffer = buffer[-overlap_words:] if overlap_words else []
            buffer.extend(unit_words)
        if buffer:
            chunks.append(_make_chunk(page, buffer))
    return chunks


def _make_chunk(page: Page, words: list[str]) -> Chunk:
    text = " ".join(words)
    digest = hashlib.sha1(f"{page.source}|{page.page}|{text}".encode()).hexdigest()[:16]
    return Chunk(id=digest, source=page.source, page=page.page, text=text)
