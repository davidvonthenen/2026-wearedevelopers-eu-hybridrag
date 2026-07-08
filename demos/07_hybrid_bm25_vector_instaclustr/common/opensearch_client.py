"""OpenSearch client utilities for recipe BM25 and vector search."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError

from .config import Settings, load_settings, _get_bool, _get_str
from .labels import normalize_values
from .logging import get_logger


LOGGER = get_logger(__name__)

VECTOR_FIELD = _get_str("VECTOR_FIELD", "embedding")
VECTOR_USE_RESCORE = _get_bool("VECTOR_USE_RESCORE", False)


class MyOpenSearch(OpenSearch):
    """Explicit OpenSearch client for recipe storage/search."""

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


def _create_client(settings: Optional[Settings] = None) -> MyOpenSearch:
    settings = settings or load_settings()
    return _build_client(
        host=settings.opensearch_host,
        port=settings.opensearch_port,
        user=settings.opensearch_user or None,
        password=settings.opensearch_password or None,
        ssl=bool(settings.opensearch_ssl),
        settings=settings,
    )


def create_vector_client(settings: Optional[Settings] = None) -> Tuple[MyOpenSearch, str]:
    """Create an OpenSearch client for the recipe vector index."""

    settings = settings or load_settings()
    LOGGER.info(
        "Connecting to OpenSearch vector store at %s:%s (index=%s)",
        settings.opensearch_host,
        settings.opensearch_port,
        settings.opensearch_vector_index,
    )
    return _create_client(settings), settings.opensearch_vector_index


def create_bm25_client(settings: Optional[Settings] = None) -> Tuple[MyOpenSearch, str]:
    """Create an OpenSearch client for the recipe BM25 grounding index."""

    settings = settings or load_settings()
    LOGGER.info(
        "Connecting to OpenSearch BM25 store at %s:%s (index=%s)",
        settings.opensearch_host,
        settings.opensearch_port,
        settings.opensearch_bm25_index,
    )
    return _create_client(settings), settings.opensearch_bm25_index


def _recipe_source_fields() -> List[str]:
    return [
        "recipe_id",
        "recipe_name",
        "source",
        "url",
        "image_url",
        "servings",
        "calories",
        "chunk_id",
        "chunk_index",
        "chunk_count",
        "chunk_number",
        "document_filename",
        "text",
        "entities",
        "tags",
        "cautions",
        "cautions_display",
        "health_labels",
        "health_labels_display",
        "diet_labels",
        "diet_labels_display",
        "cuisine_type",
        "cuisine_type_display",
        "meal_type",
        "meal_type_display",
        "dish_type",
        "dish_type_display",
    ]


def _entity_tag_filter(entities: Optional[Iterable[str]]) -> Optional[Dict[str, Any]]:
    normalized = normalize_values(entities or [])
    if not normalized:
        return None
    return {
        "nested": {
            "path": "tags",
            "query": {
                "bool": {
                    "must": [
                        {"term": {"tags.key": "entity"}},
                        {"terms": {"tags.value": normalized}},
                    ]
                }
            },
            "score_mode": "avg",
        }
    }


def build_recipe_bm25_query(
    query_text: str,
    *,
    k: int,
    entities: Optional[Iterable[str]] = None,
    include_recipe_ids: Optional[Iterable[str]] = None,
    content_notquery: Optional[str] = None,
    exclude_cautions: Optional[Iterable[str]] = None,
    require_cautions: Optional[Iterable[str]] = None,
    require_health_labels: Optional[Iterable[str]] = None,
    require_diet_labels: Optional[Iterable[str]] = None,
    cuisine_type: Optional[Iterable[str]] = None,
    meal_type: Optional[Iterable[str]] = None,
    dish_type: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a BM25 query over recipe chunks with optional entity-tag boosting."""

    text = str(query_text or "").strip()
    filters: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []
    should: List[Dict[str, Any]] = []

    recipe_ids = [str(x).strip() for x in (include_recipe_ids or []) if str(x).strip()]
    if recipe_ids:
        filters.append({"terms": {"recipe_id": recipe_ids}})

    excluded = normalize_values(exclude_cautions or [])
    if excluded:
        must_not.append({"terms": {"cautions": excluded}})

    required_cautions = normalize_values(require_cautions or [])
    for caution in required_cautions:
        filters.append({"term": {"cautions": caution}})

    required_health = normalize_values(require_health_labels or [])
    for label in required_health:
        filters.append({"term": {"health_labels": label}})

    required_diet = normalize_values(require_diet_labels or [])
    for label in required_diet:
        filters.append({"term": {"diet_labels": label}})

    cuisine = normalize_values(cuisine_type or [])
    if cuisine:
        filters.append({"terms": {"cuisine_type": cuisine}})

    meals = normalize_values(meal_type or [])
    if meals:
        filters.append({"terms": {"meal_type": meals}})

    dishes = normalize_values(dish_type or [])
    if dishes:
        filters.append({"terms": {"dish_type": dishes}})

    not_text = str(content_notquery or "").strip()
    if not_text:
        must_not.append(
            {
                "multi_match": {
                    "query": not_text,
                    "fields": ["recipe_name^2", "text"],
                    "operator": "and",
                }
            }
        )

    entity_filter = _entity_tag_filter(entities)
    if entity_filter:
        should.append({"constant_score": {"filter": entity_filter, "boost": 6.0}})

    must: List[Dict[str, Any]]
    if text:
        must = [
            {
                "multi_match": {
                    "query": text,
                    "fields": [
                        "recipe_name^4",
                        "entities^3",
                        "text^2",
                        "source",
                    ],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        ]
    else:
        must = [{"match_all": {}}]

    bool_query: Dict[str, Any] = {
        "must": must,
        "filter": filters,
        "must_not": must_not,
    }
    if should:
        bool_query["should"] = should
        if not text:
            bool_query["minimum_should_match"] = 1

    return {
        "size": int(k),
        "_source": _recipe_source_fields(),
        "query": {"bool": bool_query},
    }


def bm25_search(
    label: str,
    client: MyOpenSearch,
    index_name: str,
    query_text: str,
    *,
    k: int,
    entities: Optional[Iterable[str]] = None,
    include_recipe_ids: Optional[Iterable[str]] = None,
    content_notquery: Optional[str] = None,
    exclude_cautions: Optional[Iterable[str]] = None,
    require_cautions: Optional[Iterable[str]] = None,
    require_health_labels: Optional[Iterable[str]] = None,
    require_diet_labels: Optional[Iterable[str]] = None,
    cuisine_type: Optional[Iterable[str]] = None,
    meal_type: Optional[Iterable[str]] = None,
    dish_type: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Run one BM25 grounding search and return the raw OpenSearch response."""

    body = build_recipe_bm25_query(
        query_text,
        k=k,
        entities=entities,
        include_recipe_ids=include_recipe_ids,
        content_notquery=content_notquery,
        exclude_cautions=exclude_cautions,
        require_cautions=require_cautions,
        require_health_labels=require_health_labels,
        require_diet_labels=require_diet_labels,
        cuisine_type=cuisine_type,
        meal_type=meal_type,
        dish_type=dish_type,
    )

    try:
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
            "_error": f"{exc.__class__.__name__}: {getattr(exc, 'error', str(exc))}",
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        }

    res["_store_label"] = label
    res["_index_used"] = index_name
    res["_query"] = body
    return res


def build_recipe_vector_query(
    query_vector: List[float],
    *,
    k: int,
    candidate_k: Optional[int] = None,
    field: str = VECTOR_FIELD,
    include_recipe_ids: Optional[Iterable[str]] = None,
    exclude_cautions: Optional[Iterable[str]] = None,
    require_cautions: Optional[Iterable[str]] = None,
    require_health_labels: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build a filtered kNN query over recipe vector chunks."""

    candidate_k = int(candidate_k or max(k, 50))
    knn_body: Dict[str, Any] = {"vector": query_vector, "k": candidate_k}

    if VECTOR_USE_RESCORE and candidate_k > k:
        knn_body["rescore"] = {"oversample_factor": float(candidate_k) / float(max(k, 1))}

    filters: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []

    recipe_ids = [str(x) for x in (include_recipe_ids or []) if str(x).strip()]
    if recipe_ids:
        filters.append({"terms": {"recipe_id": recipe_ids}})

    excluded = normalize_values(exclude_cautions or [])
    if excluded:
        must_not.append({"terms": {"cautions": excluded}})

    required_cautions = normalize_values(require_cautions or [])
    if required_cautions:
        filters.append({"terms": {"cautions": required_cautions}})

    required_health = normalize_values(require_health_labels or [])
    for label in required_health:
        filters.append({"term": {"health_labels": label}})

    return {
        "size": int(k),
        "_source": _recipe_source_fields(),
        "query": {
            "bool": {
                "must": [{"knn": {field: knn_body}}],
                "filter": filters,
                "must_not": must_not,
            }
        },
    }


def knn_search(
    label: str,
    client: MyOpenSearch,
    index_name: str,
    query_vector: List[float],
    *,
    k: int,
    field: str = VECTOR_FIELD,
    candidate_k: Optional[int] = None,
    include_recipe_ids: Optional[Iterable[str]] = None,
    exclude_cautions: Optional[Iterable[str]] = None,
    require_cautions: Optional[Iterable[str]] = None,
    require_health_labels: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Run one recipe vector search and return the raw OpenSearch response."""

    body = build_recipe_vector_query(
        query_vector,
        k=k,
        candidate_k=candidate_k,
        field=field,
        include_recipe_ids=include_recipe_ids,
        exclude_cautions=exclude_cautions,
        require_cautions=require_cautions,
        require_health_labels=require_health_labels,
    )

    try:
        kwargs: Dict[str, Any] = {"index": index_name, "body": body, "request_timeout": 30}
        if client.settings.search_preference:
            kwargs["preference"] = client.settings.search_preference
        res = client.search(**kwargs)
    except TransportError as exc:
        LOGGER.warning(
            "OpenSearch vector query failed on %s/%s: %s",
            label,
            index_name,
            getattr(exc, "error", str(exc)),
        )
        return {
            "_store_label": label,
            "_index_used": index_name,
            "_query": _redact_vector(body, field),
            "_error": f"{exc.__class__.__name__}: {getattr(exc, 'error', str(exc))}",
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
        }

    res["_store_label"] = label
    res["_index_used"] = index_name
    res["_query"] = _redact_vector(body, field)
    res["_query_vector_dim"] = len(query_vector)
    return res


def _redact_vector(body: Dict[str, Any], field: str) -> Dict[str, Any]:
    """Return a JSON-ish query copy without dumping the embedding vector."""

    import copy

    redacted = copy.deepcopy(body)
    try:
        redacted["query"]["bool"]["must"][0]["knn"][field]["vector"] = "<omitted>"
    except Exception:
        pass
    return redacted


def _normalize_hit(response: Dict[str, Any], hit: Dict[str, Any]) -> Dict[str, Any]:
    src = hit.get("_source") or {}
    return {
        "score": float(hit.get("_score") or 0.0),
        "id": hit.get("_id") or "",
        "index": response.get("_index_used") or hit.get("_index") or "",
        "recipe_id": src.get("recipe_id") or "",
        "recipe_name": src.get("recipe_name") or "",
        "source": src.get("source") or "",
        "url": src.get("url") or "",
        "image_url": src.get("image_url") or "",
        "servings": src.get("servings"),
        "calories": src.get("calories"),
        "chunk_id": src.get("chunk_id") or "",
        "chunk_index": src.get("chunk_index"),
        "chunk_count": src.get("chunk_count"),
        "chunk_number": src.get("chunk_number"),
        "document_filename": src.get("document_filename") or "",
        "text": (src.get("text") or "").strip(),
        "entities": src.get("entities") or [],
        "tags": src.get("tags") or [],
        "cautions": src.get("cautions") or [],
        "cautions_display": src.get("cautions_display") or [],
        "health_labels": src.get("health_labels") or [],
        "health_labels_display": src.get("health_labels_display") or [],
        "diet_labels": src.get("diet_labels") or [],
        "diet_labels_display": src.get("diet_labels_display") or [],
        "cuisine_type": src.get("cuisine_type") or [],
        "meal_type": src.get("meal_type") or [],
        "dish_type": src.get("dish_type") or [],
    }


def normalize_bm25_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize OpenSearch BM25 hits into compact recipe chunk dictionaries."""

    hits = response.get("hits", {}).get("hits", []) or []
    return [_normalize_hit(response, hit) for hit in hits]


def normalize_vector_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize OpenSearch vector hits into compact recipe chunk dictionaries."""

    hits = response.get("hits", {}).get("hits", []) or []
    return [_normalize_hit(response, hit) for hit in hits]


__all__ = [
    "MyOpenSearch",
    "create_vector_client",
    "create_bm25_client",
    "build_recipe_bm25_query",
    "bm25_search",
    "normalize_bm25_hits",
    "build_recipe_vector_query",
    "knn_search",
    "normalize_vector_hits",
    "VECTOR_FIELD",
]
