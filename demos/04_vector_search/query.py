#!/usr/bin/env python3
"""OpenSearch-only vector query for recipes.

The query text is embedded and sent directly to OpenSearch as a kNN search.
The metadata-style CLI options are normalized for reporting, but this vector-only
demo does not apply them as hard OpenSearch filters. No second database is
consulted, because the dependency pile had already become a lifestyle choice.
"""

from __future__ import annotations

import argparse
from asyncio import sleep
import json
from typing import Any, Dict, List, Sequence

from common.embeddings import EmbeddingModel, to_list
from common.opensearch_client import create_vector_client, knn_search, normalize_vector_hits
from common.recipe_utils import normalize_values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for a single vector-only recipe search."""

    parser = argparse.ArgumentParser(description="OpenSearch-only recipe vector search")
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Semantic recipe query used for kNN vector search.",
    )
    parser.add_argument(
        "--exclude-caution",
        action="append",
        default=[],
        help="Record an allergy warning exclusion in the output summary; not applied as a vector filter."
    )
    parser.add_argument(
        "--require-caution",
        action="append",
        default=[],
        help="Record a required allergy warning in the output summary; not applied as a vector filter."
    )
    parser.add_argument(
        "--require-health-label",
        action="append",
        default=[],
        help="Record a required health label in the output summary; not applied as a vector filter."
    )
    parser.add_argument("--k", type=int, default=10, help="Vector hits to return.")
    parser.add_argument("--candidate-k", type=int, default=6, help="Candidate count sent to the OpenSearch kNN clause.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    return parser.parse_args(argv)


def build_filter_summary(args: argparse.Namespace) -> Dict[str, List[str]]:
    """Normalize metadata-style flags for transparent output reporting.

    The current query path does not push these values into the OpenSearch body;
    they are shown so users can compare requested constraints with pure vector
    retrieval behavior.
    """

    return {
        "exclude_cautions": normalize_values(args.exclude_caution),
        "require_cautions": normalize_values(args.require_caution),
        "require_health_labels": normalize_values(args.require_health_label),
    }


def build_output(
    *,
    query_text: str,
    filters: Dict[str, List[str]],
    vector_index: str,
    vector_hits: List[Dict[str, Any]],
    response: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the compact JSON payload printed by the CLI."""

    return {
        "query": query_text,
        "filters": filters,
        "vector_index": vector_index,
        "query_vector_dim": response.get("_query_vector_dim"),
        "hit_count": len(vector_hits),
        "vector_hits": vector_hits,
        "error": response.get("_error", ""),
    }


def print_output(payload: Dict[str, Any], verbose: bool) -> None:
    """Print vector hits, hiding chunk text unless verbose mode is enabled."""

    print("\n")

    print("-" * 20)
    print("Vector Hits:")
    print("-" * 20)
    if verbose:
        print(json.dumps(payload["vector_hits"], ensure_ascii=False, indent=2))
    else:
        # Compact mode keeps the metadata and score visible while omitting the
        # full embedded chunk text, which can be very noisy in terminal output.
        for match in payload["vector_hits"]:
            match.pop("text", None)
        print(json.dumps(payload["vector_hits"], ensure_ascii=False, indent=2))

    print("\n")


def main(argv: Sequence[str] | None = None) -> None:
    """Embed the query, run OpenSearch kNN, and print normalized hits."""

    args = parse_args(argv)
    query_text = args.query.strip()
    if not query_text:
        raise SystemExit("--query must contain non-empty text for vector search")
    if int(args.k) <= 0:
        raise SystemExit("--k must be > 0")
    if int(args.candidate_k) <= 0:
        raise SystemExit("--candidate-k must be > 0")

    vector_client, vector_index = create_vector_client()

    print("\n")
    print('Let\'s look for recipes that satisfy the following criteria:')
    print(" - Includes: {}".format(query_text))
    print("\n")

    try:
        embedder = EmbeddingModel()
        query_vector = to_list(embedder.encode([query_text])[0])
        response = knn_search(
            "VECTOR",
            vector_client,
            vector_index,
            query_vector,
            k=int(args.k),
            candidate_k=int(args.candidate_k),
            exclude_cautions=args.exclude_caution,
            require_cautions=args.require_caution,
            require_health_labels=args.require_health_label,
        )
        vector_hits = normalize_vector_hits(response)
    finally:
        vector_client.close()

    print_output(
        build_output(
            query_text=query_text,
            filters=build_filter_summary(args),
            vector_index=vector_index,
            vector_hits=vector_hits,
            response=response,
        ),
        args.verbose,
    )


if __name__ == "__main__":
    main()
