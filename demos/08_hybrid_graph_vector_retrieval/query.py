#!/usr/bin/env python3
"""Minimal graph-constrained vector query for recipes.

This script keeps retrieval visible by printing both sides of the hybrid
pipeline:

* Neo4j graph matches, which apply exact content, allergy/caution, label, meal,
  cuisine, and dish-type filters.
* OpenSearch vector hits, which use the natural-language query for semantic
  chunk retrieval and are optionally constrained to graph-matched recipe IDs.

In other words: let the graph enforce facts, let embeddings handle vibes. A rare
division of labor that does not require a committee.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Sequence

from common.embeddings import EmbeddingModel, to_list
from common.graph import find_recipes_by_profile
from common.neo4j_client import MyNeo4j, create_graph_client
from common.opensearch_client import create_vector_client, knn_search, normalize_vector_hits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Graph-filtered recipe vector search")
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Recipe query. Used as both a graph content substring filter and the semantic vector query.",
    )
    parser.add_argument(
        "--notquery",
        type=str,
        default="",
        help="Exclude recipes whose Recipe.content contains this text, case-insensitive.",
    )
    parser.add_argument("--exclude-caution", action="append", default=[], help="Exclude an allergy warning, repeatable.")
    parser.add_argument("--require-caution", action="append", default=[], help="Require an allergy warning, repeatable.")
    parser.add_argument("--require-health-label", action="append", default=[], help="Require a health label, repeatable.")
    parser.add_argument("--require-diet-label", action="append", default=[], help="Require a diet label, repeatable.")
    parser.add_argument("--cuisine-type", action="append", default=[], help="Filter by cuisine type, repeatable.")
    parser.add_argument("--meal-type", action="append", default=[], help="Filter by meal type, repeatable.")
    parser.add_argument("--dish-type", action="append", default=[], help="Filter by dish type, repeatable.")
    parser.add_argument("--vk", type=int, default=10, help="Vector hits to return.")
    parser.add_argument("--gk", type=int, default=5, help="Graph hits to return.")
    parser.add_argument("--candidate-k", type=int, default=50, help="Vector candidates to retrieve before final top-k.")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output.")
    return parser.parse_args(argv)


def has_graph_filters(args: argparse.Namespace) -> bool:
    """Return true when explicit structured graph filters were supplied."""

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


def has_graph_constraints(args: argparse.Namespace) -> bool:
    """Return true when graph results should constrain vector search.

    ``--query`` participates in graph filtering by requiring the extracted
    recipe text in ``Recipe.content`` to contain the query string. ``--notquery``
    applies the matching negative content filter. Yes, this is stricter than
    semantic search. That is the point.
    """

    return has_graph_filters(args) or bool(args.query.strip()) or bool(args.notquery.strip())


def build_output(
    *,
    query_text: str,
    notquery_text: str,
    graph_records: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the printable retrieval payload without changing result ordering."""

    return {
        "query": query_text,
        "notquery": notquery_text,
        "graph_content_filter": bool(query_text),
        "graph_content_exclusion_filter": bool(notquery_text),
        "graph_match_count": len(graph_records),
        "graph_matches": graph_records,
        "vector_hits": vector_hits,
    }


def print_output(payload: Dict[str, Any], verbose: bool) -> None:
    """Print graph and vector retrieval results in verbose or compact form."""

    print("\n")

    print("-" * 20)
    print("Graph Matches:")
    print("-" * 20)
    if verbose:
        print(json.dumps(payload["graph_matches"], ensure_ascii=False, indent=2))
    else:
        # Compact mode hides the large Recipe.content field so the terminal shows
        # recipe metadata and constraints without dumping full page text. Because
        # apparently even terminals deserve mercy.
        for match in payload["graph_matches"]:
            match.pop("content", None)
        print(json.dumps(payload["graph_matches"], ensure_ascii=False, indent=2))

    print("\n\n")

    print("-" * 20)
    print("Vector Hits:")
    print("-" * 20)
    if verbose:
        print(json.dumps(payload["vector_hits"], ensure_ascii=False, indent=2))
    else:
        # Compact mode hides vector chunk text. Use --verbose when inspecting the
        # exact text that OpenSearch returned for prompt construction.
        for match in payload["vector_hits"]:
            match.pop("text", None)
        print(json.dumps(payload["vector_hits"], ensure_ascii=False, indent=2))

    print("\n")


def main(argv: Sequence[str] | None = None) -> None:
    """Run one graph retrieval pass and, when possible, one vector retrieval pass."""

    args = parse_args(argv)
    query_text = args.query.strip()
    notquery_text = args.notquery.strip()
    graph_constraints_active = has_graph_constraints(args)

    # Neo4j and OpenSearch are queried separately on purpose so the workshop can
    # show how exact graph constraints and semantic vector retrieval interact.
    graph_client = create_graph_client()
    vector_client, vector_index = create_vector_client()

    print("\n")
    print('Let\'s look for recipes that satisfy the following criteria:')
    print(" - Includes: {}".format(query_text))
    print(" - Excludes: {}".format(notquery_text))
    print(" - Exclude Severe Allergens: {}".format(", ".join(args.exclude_caution)))
    print("\n")

    try:
        graph_records = find_recipes_by_profile(
            graph_client,
            content_query=query_text,
            content_notquery=notquery_text,
            required_cautions=args.require_caution,
            excluded_cautions=args.exclude_caution,
            required_health_labels=args.require_health_label,
            required_diet_labels=args.require_diet_label,
            cuisine_type=args.cuisine_type,
            meal_type=args.meal_type,
            dish_type=args.dish_type,
            limit=args.gk,
        )
    finally:
        graph_client.close()

    recipe_ids: List[str] = [str(row.get("recipe_id")) for row in graph_records if row.get("recipe_id")]

    if graph_constraints_active and not recipe_ids:
        # When graph constraints are active, an empty graph result means there is
        # no authoritative recipe set to refine with vector search. Falling back
        # to unconstrained vectors here would reintroduce the fuzzy behavior this
        # demo is intentionally avoiding. Humans keep trying this. The graph says no.
        print_output(
            build_output(
                query_text=query_text,
                notquery_text=notquery_text,
                graph_records=[],
                vector_hits=[],
            ),
            args.verbose,
        )
        return

    if not query_text:
        # Structured graph filters can still be useful without a semantic query;
        # in that case there is nothing meaningful to embed for kNN search.
        print_output(
            build_output(
                query_text=query_text,
                notquery_text=notquery_text,
                graph_records=graph_records,
                vector_hits=[],
            ),
            args.verbose,
        )
        return
    try:
        embedder = EmbeddingModel()
        query_vector = to_list(embedder.encode([query_text])[0])
        # Vector search is semantic, but it can still be narrowed to recipe IDs
        # that survived the graph pass. This keeps semantic detail attached to
        # graph-authoritative candidates instead of wandering off into embedding
        # fairyland with a suspiciously confident smile.
        response = knn_search(
            "VECTOR",
            vector_client,
            vector_index,
            query_vector,
            k=args.vk,
            candidate_k=args.candidate_k,
            include_recipe_ids=recipe_ids if graph_constraints_active else None,
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
            notquery_text=notquery_text,
            graph_records=graph_records,
            vector_hits=vector_hits,
        ),
        args.verbose,
    )


if __name__ == "__main__":
    main()
