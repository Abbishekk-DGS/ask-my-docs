"""Command line: `ask-my-docs ingest <path>` and `ask-my-docs ask "<question>"`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .llm import OllamaChat, OllamaEmbedder
from .rag import AskMyDocs

DEFAULT_INDEX = Path.home() / ".ask-my-docs" / "index"


def build(args: argparse.Namespace) -> AskMyDocs:
    return AskMyDocs(
        Path(args.index),
        OllamaEmbedder(model=args.embed_model),
        OllamaChat(model=args.model),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ask-my-docs", description=__doc__)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__} · by Abbishekk DGS · abbishekkdgs.tech",
    )
    parser.add_argument("--index", default=str(DEFAULT_INDEX), help="index directory")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama chat model")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a file or folder")
    ingest.add_argument("path")

    ask = sub.add_parser("ask", help="ask a question")
    ask.add_argument("question")
    ask.add_argument("-k", type=int, default=5, help="passages to retrieve")
    ask.add_argument("--show-sources", action="store_true")

    sub.add_parser("stats", help="show index size")

    args = parser.parse_args(argv)
    mind = build(args)

    if args.command == "ingest":
        added = mind.ingest(Path(args.path).expanduser())
        print(f"Indexed {added} new chunks ({len(mind.store)} total).")
    elif args.command == "stats":
        sources = {c.source for c in mind.store.chunks}
        print(f"{len(mind.store)} chunks from {len(sources)} files in {args.index}")
    else:
        answer = mind.ask(args.question, k=args.k)
        for token in answer.tokens:
            sys.stdout.write(token)
            sys.stdout.flush()
        print()
        if args.show_sources:
            for n, hit in enumerate(answer.hits, 1):
                print(
                    f"  [{n}] {hit.chunk.citation}  (dense #{hit.dense_rank or '-'}, bm25 #{hit.sparse_rank or '-'})"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
