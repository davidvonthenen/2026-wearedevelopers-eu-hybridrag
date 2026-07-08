"""Neo4j recipe graph helpers.

The graph stores structured recipe metadata explicitly. This matters for allergy
and label handling because safety-sensitive constraints should be represented as
deterministic graph relationships rather than delegated only to semantic search.

Schema overview
---------------
(:Recipe {recipe_id, ...})
(:RecipeChunk {recipe_id, chunk_index, text, ...})
(:AllergyWarning {name, display_name})
(:DietLabel {name, display_name})
(:HealthLabel {name, display_name})
(:CuisineType {name, display_name})
(:MealType {name, display_name})
(:DishType {name, display_name})
(:RecipeSource {name, display_name})

Relationships
-------------
(:Recipe)-[:HAS_CHUNK]->(:RecipeChunk)
(:Recipe)-[:HAS_CAUTION]->(:AllergyWarning)
(:Recipe)-[:HAS_DIET_LABEL]->(:DietLabel)
(:Recipe)-[:HAS_HEALTH_LABEL]->(:HealthLabel)
(:Recipe)-[:HAS_CUISINE_TYPE]->(:CuisineType)
(:Recipe)-[:HAS_MEAL_TYPE]->(:MealType)
(:Recipe)-[:HAS_DISH_TYPE]->(:DishType)
(:Recipe)-[:FROM_SOURCE]->(:RecipeSource)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from neo4j.exceptions import Neo4jError

from .logging import get_logger
from .neo4j_client import MyNeo4j


LOGGER = get_logger(__name__)

_LABEL_SPACE_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------------------


def normalize_key(value: Any) -> str:
    """Normalize a human label for deterministic graph keys and keyword filters."""

    text = str(value or "").strip().lower()
    text = _LABEL_SPACE_RE.sub(" ", text)
    return text


def to_label_maps(values: Iterable[Any]) -> List[Dict[str, str]]:
    """Convert raw labels into Neo4j node payload maps."""

    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    for raw in values or []:
        display = str(raw or "").strip()
        key = normalize_key(display)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append({"name": key, "display_name": display})
    return out


# --------------------------------------------------------------------------------------
# Schema helpers
# --------------------------------------------------------------------------------------


def ensure_graph_schema(
    client: MyNeo4j,
    *,
    create_fulltext: bool = True,
    observability: bool = False,
) -> None:
    """Ensure recipe graph constraints and indexes exist.

    Full-text index creation is best-effort because Neo4j versions and privileges
    can vary across local and managed environments. The uniqueness constraints
    and standard indexes are sufficient for correctness if full-text creation is
    unavailable.
    """

    statements = [
        "CREATE CONSTRAINT recipe_unique IF NOT EXISTS FOR (r:Recipe) REQUIRE (r.recipe_id) IS UNIQUE",
        "CREATE CONSTRAINT recipe_chunk_unique IF NOT EXISTS FOR (c:RecipeChunk) REQUIRE (c.recipe_id, c.chunk_index) IS UNIQUE",
        "CREATE CONSTRAINT allergy_warning_name_unique IF NOT EXISTS FOR (a:AllergyWarning) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT diet_label_name_unique IF NOT EXISTS FOR (d:DietLabel) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT health_label_name_unique IF NOT EXISTS FOR (h:HealthLabel) REQUIRE h.name IS UNIQUE",
        "CREATE CONSTRAINT cuisine_type_name_unique IF NOT EXISTS FOR (c:CuisineType) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT meal_type_name_unique IF NOT EXISTS FOR (m:MealType) REQUIRE m.name IS UNIQUE",
        "CREATE CONSTRAINT dish_type_name_unique IF NOT EXISTS FOR (d:DishType) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT recipe_source_name_unique IF NOT EXISTS FOR (s:RecipeSource) REQUIRE s.name IS UNIQUE",
        "CREATE INDEX recipe_url IF NOT EXISTS FOR (r:Recipe) ON (r.url)",
        "CREATE INDEX recipe_name IF NOT EXISTS FOR (r:Recipe) ON (r.name)",
        "CREATE INDEX recipe_chunk_lookup IF NOT EXISTS FOR (c:RecipeChunk) ON (c.recipe_id)",
    ]

    for cypher in statements:
        _run_schema(client, cypher, observability=observability)

    if create_fulltext:
        _run_schema(
            client,
            "CREATE FULLTEXT INDEX recipe_fulltext IF NOT EXISTS "
            "FOR (n:Recipe|RecipeChunk) ON EACH [n.name, n.source, n.content, n.text]",
            observability=observability,
        )


def _run_schema(
    client: MyNeo4j,
    cypher: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    observability: bool = False,
) -> None:
    if observability:
        print(
            "\n[GRAPH_SCHEMA_CYPHER]\n"
            + json.dumps({"cypher": cypher, "params": params or {}}, ensure_ascii=False, indent=2, sort_keys=True)
        )

    try:
        client.run(cypher, params or {}, readonly=False)
    except Neo4jError as exc:
        LOGGER.warning(
            "Neo4j schema statement failed (db=%s): %s",
            client.database,
            getattr(exc, "message", str(exc)),
        )


# --------------------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------------------


_CYPHER_INGEST_RECIPE = """
MERGE (r:Recipe {recipe_id: $recipe_id})
WITH r
CALL (r) {
  OPTIONAL MATCH (r)-[:HAS_CHUNK]->(old_chunk:RecipeChunk)
  WITH collect(old_chunk) AS old_chunks
  FOREACH (old_chunk IN old_chunks | DETACH DELETE old_chunk)
  RETURN size(old_chunks) AS deleted_chunks
}
WITH r
CALL (r) {
  OPTIONAL MATCH (r)-[old_rel:FROM_SOURCE|HAS_CAUTION|HAS_DIET_LABEL|HAS_HEALTH_LABEL|HAS_CUISINE_TYPE|HAS_MEAL_TYPE|HAS_DISH_TYPE]->()
  WITH collect(old_rel) AS old_rels
  FOREACH (old_rel IN old_rels | DELETE old_rel)
  RETURN size(old_rels) AS deleted_rels
}
WITH r
SET r.name = $recipe_name,
    r.source = $source,
    r.url = $url,
    r.path = $url,
    r.servings = $servings,
    r.calories = $calories,
    r.image_url = $image_url,
    r.diet_labels = $diet_label_names,
    r.health_labels = $health_label_names,
    r.cautions = $caution_names,
    r.cuisine_type = $cuisine_type_names,
    r.meal_type = $meal_type_names,
    r.dish_type = $dish_type_names,
    r.content = $content,
    r.content_sha1 = $content_sha1,
    r.fetch_status = $fetch_status,
    r.fetch_error = $fetch_error,
    r.ingested_at_ms = $now_ms,
    r.doc_version = $now_ms
FOREACH (src IN $source_nodes |
  MERGE (s:RecipeSource {name: src.name})
  SET s.display_name = src.display_name
  MERGE (r)-[:FROM_SOURCE]->(s)
)
FOREACH (item IN $cautions |
  MERGE (a:AllergyWarning {name: item.name})
  SET a.display_name = item.display_name
  MERGE (r)-[:HAS_CAUTION]->(a)
)
FOREACH (item IN $diet_labels |
  MERGE (d:DietLabel {name: item.name})
  SET d.display_name = item.display_name
  MERGE (r)-[:HAS_DIET_LABEL]->(d)
)
FOREACH (item IN $health_labels |
  MERGE (h:HealthLabel {name: item.name})
  SET h.display_name = item.display_name
  MERGE (r)-[:HAS_HEALTH_LABEL]->(h)
)
FOREACH (item IN $cuisine_type |
  MERGE (c:CuisineType {name: item.name})
  SET c.display_name = item.display_name
  MERGE (r)-[:HAS_CUISINE_TYPE]->(c)
)
FOREACH (item IN $meal_type |
  MERGE (m:MealType {name: item.name})
  SET m.display_name = item.display_name
  MERGE (r)-[:HAS_MEAL_TYPE]->(m)
)
FOREACH (item IN $dish_type |
  MERGE (d:DishType {name: item.name})
  SET d.display_name = item.display_name
  MERGE (r)-[:HAS_DISH_TYPE]->(d)
)
WITH r
UNWIND $chunks AS ch
MERGE (c:RecipeChunk {recipe_id: $recipe_id, chunk_index: ch.chunk_index})
SET c.chunk_id = ch.chunk_id,
    c.chunk_count = ch.chunk_count,
    c.text = ch.text,
    c.recipe_name = $recipe_name,
    c.url = $url,
    c.cautions = $caution_names,
    c.health_labels = $health_label_names,
    c.ingested_at_ms = $now_ms,
    c.doc_version = $now_ms
MERGE (r)-[:HAS_CHUNK]->(c)
""".strip()


def ingest_recipe(
    client: MyNeo4j,
    *,
    recipe: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    now_ms: int,
) -> int:
    """Write one recipe and its metadata/chunks to Neo4j.

    Args:
        client: Explicit Neo4j client.
        recipe: Recipe payload with scalar fields, normalized label lists, and content.
        chunks: Chunk payloads sharing IDs with the vector index documents.
        now_ms: Ingestion timestamp.

    Returns:
        Number of graph chunks written.
    """

    source = str(recipe.get("source") or "").strip()
    source_nodes = to_label_maps([source]) if source else []

    params: Dict[str, Any] = {
        "recipe_id": recipe["recipe_id"],
        "recipe_name": recipe.get("recipe_name") or "",
        "source": source,
        "url": recipe.get("url") or "",
        "servings": recipe.get("servings"),
        "calories": recipe.get("calories"),
        "image_url": recipe.get("image_url") or "",
        "content": recipe.get("content") or "",
        "content_sha1": recipe.get("content_sha1") or "",
        "fetch_status": recipe.get("fetch_status") or "",
        "fetch_error": recipe.get("fetch_error") or "",
        "diet_labels": to_label_maps(recipe.get("diet_labels") or []),
        "health_labels": to_label_maps(recipe.get("health_labels") or []),
        "cautions": to_label_maps(recipe.get("cautions") or []),
        "cuisine_type": to_label_maps(recipe.get("cuisine_type") or []),
        "meal_type": to_label_maps(recipe.get("meal_type") or []),
        "dish_type": to_label_maps(recipe.get("dish_type") or []),
        "source_nodes": source_nodes,
        "chunks": chunks,
        "now_ms": int(now_ms),
    }

    params["diet_label_names"] = [x["name"] for x in params["diet_labels"]]
    params["health_label_names"] = [x["name"] for x in params["health_labels"]]
    params["caution_names"] = [x["name"] for x in params["cautions"]]
    params["cuisine_type_names"] = [x["name"] for x in params["cuisine_type"]]
    params["meal_type_names"] = [x["name"] for x in params["meal_type"]]
    params["dish_type_names"] = [x["name"] for x in params["dish_type"]]

    client.run(_CYPHER_INGEST_RECIPE, params, readonly=False)
    return len(chunks)


_CYPHER_DELETE_RECIPE = """
MATCH (r:Recipe {recipe_id: $recipe_id})
CALL (r) {
  OPTIONAL MATCH (r)-[:HAS_CHUNK]->(c:RecipeChunk)
  WITH [chunk IN collect(c) WHERE chunk IS NOT NULL] AS chunks
  FOREACH (chunk IN chunks | DETACH DELETE chunk)
  RETURN size(chunks) AS deleted_chunks
}
DETACH DELETE r
RETURN deleted_chunks
""".strip()


def delete_recipe(client: MyNeo4j, *, recipe_id: str) -> int:
    """Remove one recipe and its private chunks from Neo4j.

    Shared metadata nodes such as ``AllergyWarning`` and ``HealthLabel`` are left in
    place because other recipes may still reference them.
    """

    rows = client.run(
        _CYPHER_DELETE_RECIPE,
        {"recipe_id": recipe_id},
        readonly=False,
    )
    if not rows:
        return 0
    deleted_chunks = int(rows[0].get("deleted_chunks") or 0)
    return deleted_chunks + 1


# --------------------------------------------------------------------------------------
# Query helpers for graph-first allergy filtering
# --------------------------------------------------------------------------------------


def _recipe_return_clause() -> str:
    return (
        "RETURN r.recipe_id AS recipe_id, r.name AS recipe_name, r.source AS source, "
        "r.url AS url, r.content AS content, "
        "r.servings AS servings, r.calories AS calories, "
        "r.cautions AS cautions, r.health_labels AS health_labels, "
        "r.diet_labels AS diet_labels, r.cuisine_type AS cuisine_type, "
        "r.meal_type AS meal_type, r.dish_type AS dish_type "
    )


def find_recipes_with_cautions(
    client: MyNeo4j,
    cautions: Iterable[str],
    *,
    match_all: bool = False,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Return recipes that have any or all requested allergy warning nodes."""

    caution_keys = [normalize_key(x) for x in cautions if normalize_key(x)]
    if not caution_keys:
        return []

    cypher = (
        "MATCH (r:Recipe)-[:HAS_CAUTION]->(a:AllergyWarning) "
        "WHERE a.name IN $cautions "
        "WITH r, collect(DISTINCT a.name) AS matched_cautions "
        "WHERE $match_all = false OR size(matched_cautions) = size($cautions) "
        + _recipe_return_clause()
        + ", matched_cautions "
        "ORDER BY r.name ASC "
        "LIMIT $limit"
    )
    params = {
        "cautions": caution_keys,
        "match_all": bool(match_all),
        "limit": int(limit),
    }
    return client.run(cypher, params, readonly=True)


def find_recipes_without_cautions(
    client: MyNeo4j,
    excluded_cautions: Iterable[str],
    *,
    required_health_labels: Optional[Iterable[str]] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Return recipes that do not contain any of the excluded warning nodes.

    ``required_health_labels`` can be used for positive evidence such as
    ``Gluten-Free`` in addition to the absence of ``Gluten`` warnings.
    """

    excluded = [normalize_key(x) for x in excluded_cautions if normalize_key(x)]
    required_health = [normalize_key(x) for x in (required_health_labels or []) if normalize_key(x)]

    cypher = (
        "MATCH (r:Recipe) "
        "WHERE NOT EXISTS { "
        "  MATCH (r)-[:HAS_CAUTION]->(a:AllergyWarning) "
        "  WHERE a.name IN $excluded_cautions "
        "} "
        "AND all(label IN $required_health_labels WHERE EXISTS { "
        "  MATCH (r)-[:HAS_HEALTH_LABEL]->(h:HealthLabel {name: label}) "
        "}) "
        + _recipe_return_clause()
        + "ORDER BY r.name ASC LIMIT $limit"
    )
    params = {
        "excluded_cautions": excluded,
        "required_health_labels": required_health,
        "limit": int(limit),
    }
    return client.run(cypher, params, readonly=True)


def find_recipes_by_profile(
    client: MyNeo4j,
    *,
    content_query: Optional[str] = None,
    content_notquery: Optional[str] = None,
    required_cautions: Optional[Iterable[str]] = None,
    excluded_cautions: Optional[Iterable[str]] = None,
    required_health_labels: Optional[Iterable[str]] = None,
    required_diet_labels: Optional[Iterable[str]] = None,
    cuisine_type: Optional[Iterable[str]] = None,
    meal_type: Optional[Iterable[str]] = None,
    dish_type: Optional[Iterable[str]] = None,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Filter recipes with structured graph metadata and optional content text.

    ``content_query`` and ``content_notquery`` are retained for compatibility
    with earlier callers, but the current Cypher applies them to ``Recipe.name``.
    Structured cautions, labels, cuisine type, meal type, and dish type are
    enforced through explicit graph relationships.
    """

    params = {
        "content_query": str(content_query or "").strip().lower(),
        "content_notquery": str(content_notquery or "").strip().lower(),
        "required_cautions": [normalize_key(x) for x in (required_cautions or []) if normalize_key(x)],
        "excluded_cautions": [normalize_key(x) for x in (excluded_cautions or []) if normalize_key(x)],
        "required_health_labels": [normalize_key(x) for x in (required_health_labels or []) if normalize_key(x)],
        "required_diet_labels": [normalize_key(x) for x in (required_diet_labels or []) if normalize_key(x)],
        "cuisine_type": [normalize_key(x) for x in (cuisine_type or []) if normalize_key(x)],
        "meal_type": [normalize_key(x) for x in (meal_type or []) if normalize_key(x)],
        "dish_type": [normalize_key(x) for x in (dish_type or []) if normalize_key(x)],
        "limit": int(limit),
    }

    cypher = (
        "MATCH (r:Recipe) "
        "WHERE ($content_query = '' OR toLower(coalesce(r.name, '')) CONTAINS $content_query) "
        "AND ($content_notquery = '' OR NOT (toLower(coalesce(r.name, '')) CONTAINS $content_notquery)) "
        "AND NOT EXISTS { "
        "  MATCH (r)-[:HAS_CAUTION]->(a:AllergyWarning) "
        "  WHERE a.name IN $excluded_cautions "
        "} "
        "AND all(name IN $required_cautions WHERE EXISTS { "
        "  MATCH (r)-[:HAS_CAUTION]->(:AllergyWarning {name: name}) "
        "}) "
        "AND all(name IN $required_health_labels WHERE EXISTS { "
        "  MATCH (r)-[:HAS_HEALTH_LABEL]->(:HealthLabel {name: name}) "
        "}) "
        "AND all(name IN $required_diet_labels WHERE EXISTS { "
        "  MATCH (r)-[:HAS_DIET_LABEL]->(:DietLabel {name: name}) "
        "}) "
        "AND (size($cuisine_type) = 0 OR EXISTS { "
        "  MATCH (r)-[:HAS_CUISINE_TYPE]->(c:CuisineType) WHERE c.name IN $cuisine_type "
        "}) "
        "AND (size($meal_type) = 0 OR EXISTS { "
        "  MATCH (r)-[:HAS_MEAL_TYPE]->(m:MealType) WHERE m.name IN $meal_type "
        "}) "
        "AND (size($dish_type) = 0 OR EXISTS { "
        "  MATCH (r)-[:HAS_DISH_TYPE]->(d:DishType) WHERE d.name IN $dish_type "
        "}) "
        + _recipe_return_clause()
        + "ORDER BY r.name ASC LIMIT $limit"
    )
    return client.run(cypher, params, readonly=True)


__all__ = [
    "ensure_graph_schema",
    "ingest_recipe",
    "delete_recipe",
    "normalize_key",
    "to_label_maps",
    "find_recipes_with_cautions",
    "find_recipes_without_cautions",
    "find_recipes_by_profile",
]
