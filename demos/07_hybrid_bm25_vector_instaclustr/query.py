#!/usr/bin/env python3
"""Inspect hybrid recipe retrieval without running LLM inference.

The retrieval flow runs BM25 first for grounding and structured filtering, then
uses the grounded recipe IDs to constrain vector search for semantic detail. This
keeps retrieval behavior visible before prompt construction or generation.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, List, Sequence

from common.embeddings import EmbeddingModel, to_list
from common.ner import NERClient
from common.opensearch_client import bm25_search, create_bm25_client, create_vector_client, knn_search, normalize_bm25_hits, normalize_vector_hits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line options for the retrieval inspection CLI."""

    parser = argparse.ArgumentParser(description="BM25-grounded recipe vector search")
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Recipe query. Used for BM25 grounding and semantic vector search.",
    )
    parser.add_argument(
        "--notquery",
        type=str,
        default="",
        help="Exclude chunks whose recipe name/text contains this text, case-insensitive through BM25.",
    )
    parser.add_argument("--exclude-caution", action="append", default=[], help="Exclude an allergy warning, repeatable.")
    parser.add_argument("--require-caution", action="append", default=[], help="Require an allergy warning, repeatable.")
    parser.add_argument("--require-health-label", action="append", default=[], help="Require a health label, repeatable.")
    parser.add_argument("--require-diet-label", action="append", default=[], help="Require a diet label, repeatable.")
    parser.add_argument("--cuisine-type", action="append", default=[], help="Filter by cuisine type, repeatable.")
    parser.add_argument("--meal-type", action="append", default=[], help="Filter by meal type, repeatable.")
    parser.add_argument("--dish-type", action="append", default=[], help="Filter by dish type, repeatable.")
    parser.add_argument("--vk", type=int, default=10, help="Vector hits to return.")
    parser.add_argument("--gk", type=int, default=5, help="BM25 grounding hits to return.")
    parser.add_argument("--candidate-k", type=int, default=50, help="Vector candidates to retrieve before final top-k.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    return parser.parse_args(argv)


def has_bm25_filters(args: argparse.Namespace) -> bool:
    """Return true when structured BM25 metadata filters are present."""

    fields = [
        args.exclude_caution,
        args.require_caution,
        args.require_health_label,
        args.require_diet_label,
        args.cuisine_type,
        args.meal_type,
        args.dish_type,
    ]
    return any(bool(x) for x in fields)


def has_bm25_constraints(args: argparse.Namespace) -> bool:
    """Return true when BM25 should constrain whether vector search can run."""

    return has_bm25_filters(args) or bool(args.query.strip()) or bool(args.notquery.strip())


def unique_recipe_ids(rows: Iterable[Dict[str, Any]]) -> List[str]:
    """Collect recipe IDs from BM25 hits while preserving first-seen order."""

    seen: set[str] = set()
    out: List[str] = []
    for row in rows:
        recipe_id = str(row.get("recipe_id") or "").strip()
        if recipe_id and recipe_id not in seen:
            seen.add(recipe_id)
            out.append(recipe_id)
    return out


def build_output(
    *,
    query_text: str,
    notquery_text: str,
    question_entities: List[str],
    bm25_records: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the JSON-serializable payload printed by the CLI."""

    return {
        "query": query_text,
        "notquery": notquery_text,
        "question_entities": question_entities,
        "bm25_entity_tag_query": bool(question_entities),
        "bm25_content_exclusion_filter": bool(notquery_text),
        "bm25_match_count": len(bm25_records),
        "bm25_matches": bm25_records,
        "vector_hits": vector_hits,
    }


def _strip_large_fields(rows: List[Dict[str, Any]], *field_names: str) -> List[Dict[str, Any]]:
    """Return copies of result rows with large fields removed for compact display."""

    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for field_name in field_names:
            item.pop(field_name, None)
        cleaned.append(item)
    return cleaned


def print_output(payload: Dict[str, Any], verbose: bool) -> None:
    """Print BM25 and vector result sections, optionally including full chunk text."""

    print("\n")

    print("-" * 20)
    print("BM25 Grounding Matches:")
    print("-" * 20)
    if verbose:
        print(json.dumps(payload["bm25_matches"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_strip_large_fields(payload["bm25_matches"], "text"), ensure_ascii=False, indent=2))

    print("\n\n")

    print("-" * 20)
    print("Vector Hits:")
    print("-" * 20)
    if verbose:
        print(json.dumps(payload["vector_hits"], ensure_ascii=False, indent=2))
    else:
        print(json.dumps(_strip_large_fields(payload["vector_hits"], "text"), ensure_ascii=False, indent=2))

    print("\n")


def main(argv: Sequence[str] | None = None) -> None:
    """Run BM25 grounding and, when grounded recipes exist, filtered vector search."""

    args = parse_args(argv)
    query_text = args.query.strip()
    notquery_text = args.notquery.strip()
    bm25_constraints_active = has_bm25_constraints(args)

    print("\n")
    print("Let's look for recipes that satisfy the following criteria:")
    print(" - Includes: {}".format(query_text))
    print(" - Excludes: {}".format(notquery_text))
    print(" - Exclude Severe Allergens: {}".format(", ".join(args.exclude_caution)))
    print("\n")

    ner_client = NERClient()
    question_entities = ner_client.extract_entities(query_text) if query_text else []

    opensearch_bm25_client, bm25_index = create_bm25_client()
    opensearch_vector_client, vector_index = create_vector_client()

    try:
        # Stage 1: BM25 provides the grounding set and applies structured metadata filters.
        bm25_response = bm25_search(
            "BM25",
            opensearch_bm25_client,
            bm25_index,
            query_text,
            k=args.gk,
            entities=question_entities,
            content_notquery=notquery_text,
            exclude_cautions=args.exclude_caution,
            require_cautions=args.require_caution,
            require_health_labels=args.require_health_label,
            require_diet_labels=args.require_diet_label,
            cuisine_type=args.cuisine_type,
            meal_type=args.meal_type,
            dish_type=args.dish_type,
        )
        bm25_records = normalize_bm25_hits(bm25_response)
        recipe_ids = unique_recipe_ids(bm25_records)

        # When the grounding stage finds nothing, vector search is intentionally skipped.
        # Otherwise vector retrieval could reintroduce semantically similar but ungrounded recipes.
        if bm25_constraints_active and not recipe_ids:
            print_output(
                build_output(
                    query_text=query_text,
                    notquery_text=notquery_text,
                    question_entities=question_entities,
                    bm25_records=[],
                    vector_hits=[],
                ),
                args.verbose,
            )
            return

        vector_hits: List[Dict[str, Any]] = []
        if query_text and recipe_ids:
            # Stage 2: vector search is filtered to the BM25-grounded recipe IDs.
            # This lets embeddings add semantic detail without expanding the answer set.
            embedder = EmbeddingModel(settings=opensearch_vector_client.settings)
            query_vector = to_list(embedder.encode([query_text])[0])
            vector_response = knn_search(
                "VECTOR",
                opensearch_vector_client,
                vector_index,
                query_vector,
                k=args.vk,
                candidate_k=args.candidate_k,
                include_recipe_ids=recipe_ids,
                exclude_cautions=args.exclude_caution,
                require_cautions=args.require_caution,
                require_health_labels=args.require_health_label,
            )
            vector_hits = normalize_vector_hits(vector_response)
    finally:
        opensearch_bm25_client.close()
        opensearch_vector_client.close()

    print_output(
        build_output(
            query_text=query_text,
            notquery_text=notquery_text,
            question_entities=question_entities,
            bm25_records=bm25_records,
            vector_hits=vector_hits,
        ),
        args.verbose,
    )


if __name__ == "__main__":
    main()
