"""Turn files on disk into plain-text documents with page-level provenance."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}


@dataclass
class Page:
    source: str
    page: int | None
    text: str
    meta: dict = field(default_factory=dict)


def load_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            Page(source=str(path), page=i + 1, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        ]
    if suffix in TEXT_SUFFIXES:
        return [Page(source=str(path), page=None, text=path.read_text(errors="ignore"))]
    raise ValueError(f"Unsupported file type: {path}")


def iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            if not any(part.startswith(".") for part in path.relative_to(root).parts):
                yield path


def load_path(root: Path) -> list[Page]:
    pages: list[Page] = []
    for path in iter_files(root):
        pages.extend(p for p in load_file(path) if p.text.strip())
    return pages
