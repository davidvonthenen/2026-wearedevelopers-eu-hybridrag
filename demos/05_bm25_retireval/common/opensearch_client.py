"""OpenSearch BM25 client utilities for recipe full-text search."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError

from .config import Settings, load_settings
from .labels import normalize_values
from .logging import get_logger


LOGGER = get_logger(__name__)

SOURCE_FIELDS = [
    "recipe_id",
    "recipe_name",
    "source",
    "url",
    "filename",
    "chunk_id",
    "chunk_index",
    "chunk_number",
    "chunk_count",
    "text",
    "tags",
    "named_entities",
    "cautions",
    "health_labels",
    "diet_labels",
    "cuisine_type",
    "meal_type",
    "dish_type",
]


class MyOpenSearch(OpenSearch):
    """OpenSearch client that carries loaded project settings with the connection."""

    settings: Settings

    def __init__(self, *args: Any, settings: Settings, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.settings = settings


def _build_client(
    host: str,
    port: int,
    *,
    user: Optional[str] = None,
    password: Optional[str] = None,
    ssl: bool = False,
    settings: Settings,
) -> MyOpenSearch:
    """Construct an OpenSearch client from explicit connection settings."""

    http_auth = (user, password) if user and password else None
    scheme = "https" if ssl else "http"

    return MyOpenSearch(
        hosts=[{"host": host, "port": port, "scheme": scheme}],
        http_compress=True,
        http_auth=http_auth,
        use_ssl=ssl,
        verify_certs=ssl,
        ssl_assert_hostname=False if not ssl else None,
        ssl_show_warn=ssl,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
        settings=settings,
    )


def create_bm25_client(settings: Optional[Settings] = None) -> Tuple[MyOpenSearch, str]:
    """Create an explicit OpenSearch client for the recipe BM25 chunk index."""

    settings = settings or load_settings()
    LOGGER.info(
        "Connecting to OpenSearch BM25 store at %s:%s (index=%s)",
        settings.opensearch_host,
        settings.opensearch_port,
        settings.opensearch_bm25_index,
    )
    client = _build_client(
        host=settings.opensearch_host,
        port=settings.opensearch_port,
        user=settings.opensearch_user or None,
        password=settings.opensearch_password or None,
        ssl=bool(settings.opensearch_ssl),
        settings=settings,
    )
    return client, settings.opensearch_bm25_index


def build_tag_filter(key: str, values: Iterable[str]) -> Optional[Dict[str, Any]]:
    """Build a nested tag filter matching any value for one tag key.

    This helper builds the query shape for a nested ``tags`` field. The current
    ingestion mapping does not populate nested tags, so callers should treat this
    as compatibility scaffolding for tag-based variants of the demo.
    """

    normalized = normalize_values(values)
    if not normalized:
        return None

    return {
        "nested": {
            "path": "tags",
            "query": {
                "bool": {
                    "must": [
                        {"term": {"tags.key": str(key)}},
                        {"terms": {"tags.value": normalized}},
                    ]
                }
            },
        }
    }


def build_recipe_bm25_query(
    query_text: str,
    *,
    k: int,
    named_entities: Optional[Iterable[str]] = None,
    exclude_cautions: Optional[Iterable[str]] = None,
    require_cautions: Optional[Iterable[str]] = None,
    require_health_labels: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a BM25 full-text query over recipe chunks plus keyword/tag filters."""

    filters: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []

    # Explicit user-provided filters become exact keyword constraints. This is
    # where structured metadata avoids BM25 treating exclusions as positive words.
    for caution in normalize_values(exclude_cautions or []):
        must_not.append({"term": {"cautions": caution}})

    for caution in normalize_values(require_cautions or []):
        filters.append({"term": {"cautions": caution}})

    for label in normalize_values(require_health_labels or []):
        filters.append({"term": {"health_labels": label}})

    # Query-time entities are routed through the legacy nested-tag filter hook.
    # The default ingest path stores entities in ``named_entities`` instead.
    entity_filter = build_tag_filter("entity", named_entities or [])
    if entity_filter is not None:
        filters.append(entity_filter)

    query_text = str(query_text or "").strip()
    must: List[Dict[str, Any]] = []
    if query_text:
        # The multi_match query is the BM25 lexical core of this retrieval demo.
        must.append(
            {
                "multi_match": {
                    "query": query_text,
                    "fields": [
                        "recipe_name^3",
                        "text^2",
                        "source",
                        "cuisine_type",
                        "meal_type",
                        "dish_type",
                    ],
                    "type": "best_fields",
                }
            }
        )
    else:
        must.append({"match_all": {}})

    return {
        "size": int(k),
        "_source": SOURCE_FIELDS,
        "query": {
            "bool": {
                "must": must,
                "filter": filters,
                "must_not": must_not,
            }
        },
    }


def bm25_search(
    label: str,
    client: MyOpenSearch,
    index_name: str,
    query_text: str,
    *,
    k: int,
    named_entities: Optional[Iterable[str]] = None,
    exclude_cautions: Optional[Iterable[str]] = None,
    require_cautions: Optional[Iterable[str]] = None,
    require_health_labels: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Run one recipe BM25 search and return the raw OpenSearch response."""

    entities = normalize_values(named_entities or [])
    body = build_recipe_bm25_query(
        query_text,
        k=k,
        named_entities=entities,
        exclude_cautions=exclude_cautions,
        require_cautions=require_cautions,
        require_health_labels=require_health_labels,
    )

    try:
        # Keep the raw OpenSearch response so the CLI can expose scores and the
        # exact query body for debugging when needed.
        kwargs: Dict[str, Any] = {"index": index_name, "body": body, "request_timeout": 30}
        if client.settings.search_preference:
            kwargs["preference"] = client.settings.search_preference
        res = client.search(**kwargs)
    except TransportError as exc:
        LOGGER.warning(
            "OpenSearch BM25 query failed on %s/%s: %s",
            label,
            index_name,
            getattr(exc, "error", str(exc)),
        )
        return {
            "_store_label": label,
            "_index_used": index_name,
            "_query": body,
            "_named_entities": entities,
            "_error": f"{exc.__class__.__name__}: {getattr(exc, 'error', str(exc))}",
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        }

    res["_store_label"] = label
    res["_index_used"] = index_name
    res["_query"] = body
    res["_named_entities"] = entities
    return res


def normalize_bm25_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize OpenSearch hits into compact recipe chunk dictionaries."""

    hits = response.get("hits", {}).get("hits", []) or []
    out: List[Dict[str, Any]] = []
    for hit in hits:
        src = hit.get("_source") or {}
        out.append(
            {
                "score": float(hit.get("_score") or 0.0),
                "id": hit.get("_id") or "",
                "index": response.get("_index_used") or hit.get("_index") or "",
                "recipe_id": src.get("recipe_id") or "",
                "recipe_name": src.get("recipe_name") or "",
                "source": src.get("source") or "",
                "url": src.get("url") or "",
                "filename": src.get("filename") or "",
                "chunk_id": src.get("chunk_id") or "",
                "chunk_index": src.get("chunk_index"),
                "chunk_number": src.get("chunk_number"),
                "chunk_count": src.get("chunk_count"),
                "text": (src.get("text") or "").strip(),
                "tags": src.get("tags") or [],
                "named_entities": src.get("named_entities") or [],
                "cautions": src.get("cautions") or [],
                "health_labels": src.get("health_labels") or [],
                "diet_labels": src.get("diet_labels") or [],
            }
        )
    return out


__all__ = [
    "MyOpenSearch",
    "create_bm25_client",
    "build_recipe_bm25_query",
    "bm25_search",
    "normalize_bm25_hits",
]
