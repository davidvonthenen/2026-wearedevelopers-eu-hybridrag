"""Simple client for local vector RAG with an embedded llama.cpp model."""
from __future__ import annotations

import argparse
import time
from typing import List, Sequence

from common.common import ask, load_llm
from common.config import load_settings


def _print_hits(hits: List[dict]) -> None:
    """Print compact metadata for the retrieved vector chunks."""
    print("\n")
    print("RAG Hits:")
    if not hits:
        print("  (none)")
        return

    for hit in hits:
        print(
            f"  - {hit.get('title', 'unknown')} | "
            f"{hit.get('category', 'unknown')} | "
            f"score={hit.get('score', 0.0):.4f}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for asking one question or running the demo questions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=str, help="Ask a single question and exit")
    args = parser.parse_args()

    settings = load_settings()
    llm = load_llm(settings)

    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        # Demo questions (safe to replace for your corpus)
        questions = [
            "How much did OpenAI purchase Windsurf for?",
            "How much did Google purchase Windsurf for?",
        ]

    for q in questions:
        t0 = time.time()
        answer, hits = ask(
            llm,
            q,
            settings=settings,
            top_k=settings.rag_top_k,
            num_candidates=settings.rag_num_candidates,
        )
        dt = time.time() - t0

        print("\n" + "=" * 88)
        print(f"QUESTION: {q}")
        print("=" * 88)
        print("")
        print("=" * 88)
        print(f"ANSWER: {answer}")
        _print_hits(hits)
        print("=" * 88)
        print(f"\nQuery time: {dt:.2f}s   (OpenSearch vector k-NN + local generation)")
        print(f"Docs provided to LLM: {len(hits)}\n\n")


if __name__ == "__main__":
    main()
