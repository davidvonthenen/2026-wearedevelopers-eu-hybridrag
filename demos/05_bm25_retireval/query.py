#!/usr/bin/env python3
"""OpenSearch-only BM25 query utility for recipe chunks.

The query text is sent to OpenSearch as BM25 full-text search over indexed
recipe chunks. Explicit CLI options such as ``--exclude-caution`` and
``--require-health-label`` become keyword filters against structured metadata
fields populated during ingestion.

The same query text is also sent to the external NER service. The returned
entity strings are passed through the BM25 search helper's tag-filter hook, but
this project version primarily demonstrates lexical search plus explicit
metadata filters rather than vector retrieval or local LLM inference.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence

from common.ner_client import extract_named_entities
from common.opensearch_client import bm25_search, create_bm25_client, normalize_bm25_hits
from common.recipe_utils import normalize_values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for one recipe BM25 search."""

    parser = argparse.ArgumentParser(description="OpenSearch-only recipe BM25 search")
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Recipe query used for BM25 full-text search.",
    )
    parser.add_argument("--exclude-caution", action="append", default=[], help="Exclude an allergy warning, repeatable.")
    parser.add_argument("--require-caution", action="append", default=[], help="Require an allergy warning, repeatable.")
    parser.add_argument("--require-health-label", action="append", default=[], help="Require a health label, repeatable.")
    parser.add_argument("--k", type=int, default=5, help="BM25 hits to return.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    return parser.parse_args(argv)


def build_filter_summary(args: argparse.Namespace) -> Dict[str, List[str]]:
    """Normalize filter options for the JSON payload printed to the console."""

    return {
        "exclude_cautions": normalize_values(args.exclude_caution),
        "require_cautions": normalize_values(args.require_caution),
        "require_health_labels": normalize_values(args.require_health_label),
    }


def build_output(
    *,
    query_text: str,
    filters: Dict[str, List[str]],
    named_entities: List[str],
    bm25_index: str,
    bm25_hits: List[Dict[str, Any]],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble a compact, inspectable result object for the CLI output."""

    return {
        "query": query_text,
        "filters": filters,
        "named_entities": named_entities,
        "bm25_index": bm25_index,
        "hit_count": len(bm25_hits),
        "bm25_hits": bm25_hits,
        "error": response.get("_error", ""),
    }


def print_output(payload: Dict[str, Any], verbose: bool) -> None:
    """Print BM25 hits, hiding chunk text unless verbose output is requested."""

    print("\n")

    print("-" * 20)
    print("BM25 Hits:")
    print("-" * 20)
    if verbose:
        print(json.dumps(payload["bm25_hits"], ensure_ascii=False, indent=2))
    else:
        # Non-verbose mode keeps the console focused on scores and metadata.
        # Use --verbose when inspecting the full indexed chunk text.
        for match in payload["bm25_hits"]:
            match.pop("text", None)
        print(json.dumps(payload["bm25_hits"], ensure_ascii=False, indent=2))

    print("\n")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for running one BM25 search against OpenSearch."""

    args = parse_args(argv)
    query_text = args.query.strip()
    if not query_text:
        raise SystemExit("--query must contain non-empty text for BM25 search")
    if int(args.k) <= 0:
        raise SystemExit("--k must be > 0")

    bm25_client, bm25_index = create_bm25_client()
    settings = bm25_client.settings

    print("\n")
    print("Let's look for recipes that satisfy the following criteria:")
    print(" - Includes: {}".format(query_text))
    print(" - Exclude Severe Allergens: {}".format(", ".join(args.exclude_caution)))
    print("\n")

    try:
        # Query-time NER keeps the demonstration symmetrical with ingestion: the
        # same service extracts entities from chunks and from user queries.
        named_entities = extract_named_entities(
            query_text,
            url=settings.ner_service_url,
            timeout=settings.ner_http_timeout_secs,
        )
        response = bm25_search(
            "BM25",
            bm25_client,
            bm25_index,
            query_text,
            k=int(args.k),
            named_entities=named_entities,
            exclude_cautions=args.exclude_caution,
            require_cautions=args.require_caution,
            require_health_labels=args.require_health_label,
        )
        bm25_hits = normalize_bm25_hits(response)
    finally:
        bm25_client.close()

    print_output(
        build_output(
            query_text=query_text,
            filters=build_filter_summary(args),
            named_entities=response.get("_named_entities", named_entities),
            bm25_index=bm25_index,
            bm25_hits=bm25_hits,
            response=response,
        ),
        args.verbose,
    )


if __name__ == "__main__":
    main()
