from pathlib import Path

import numpy as np

from docmind.bm25 import BM25
from docmind.chunker import chunk_pages
from docmind.llm import HashEmbedder
from docmind.loaders import Page, load_path
from docmind.rag import DocMind, build_messages
from docmind.retriever import rrf


class EchoChat:
    def __init__(self):
        self.last = None

    def stream(self, messages):
        self.last = messages
        yield "ok [1]"


def test_chunker_respects_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_pages([Page("doc.txt", None, text)], max_words=100, overlap_words=20)
    assert all(len(c.text.split()) <= 100 for c in chunks)
    first, second = chunks[0].text.split(), chunks[1].text.split()
    assert first[-20:] == second[:20]


def test_bm25_prefers_exact_terms():
    bm = BM25(["the invoice number is INV-2291", "weather is sunny today", "cats sleep a lot"])
    scores = bm.scores("INV-2291 invoice")
    assert int(np.argmax(scores)) == 0


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [1, 3, 2]])
    assert max(fused, key=fused.get) == 1


def test_end_to_end_ingest_and_ask(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "solar.md").write_text(
        "The solar panel warranty lasts 25 years.\n\nInverters: 10 years."
    )
    (docs / "cooking.txt").write_text("Boil pasta for nine minutes in salted water.")
    chat = EchoChat()
    mind = DocMind(tmp_path / "index", HashEmbedder(), chat)
    assert mind.ingest(docs) == 2
    assert mind.ingest(docs) == 0  # idempotent

    answer = mind.ask("How long is the solar panel warranty?", k=1)
    assert "".join(answer.tokens) == "ok [1]"
    assert answer.hits[0].chunk.source.endswith("solar.md")
    assert "25 years" in chat.last[-1]["content"]

    reloaded = DocMind(tmp_path / "index", HashEmbedder(), chat)
    assert len(reloaded.store) == 2


def test_prompt_numbers_passages(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha beta")
    pages = load_path(tmp_path)
    assert pages[0].text == "alpha beta"
    from docmind.retriever import Hit

    hits = [Hit(c, 1.0, 1, 1) for c in chunk_pages(pages)]
    messages = build_messages("q", hits)
    assert "[1] (a.txt)" in messages[-1]["content"]
