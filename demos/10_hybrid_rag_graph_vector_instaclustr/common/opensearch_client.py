"""OpenSearch vector client utilities for recipe semantic search."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError

from .config import Settings, load_settings, _get_bool, _get_str
from .logging import get_logger


LOGGER = get_logger(__name__)

VECTOR_FIELD = _get_str("VECTOR_FIELD", "embedding")
VECTOR_USE_RESCORE = _get_bool("VECTOR_USE_RESCORE", False)


class MyOpenSearch(OpenSearch):
    """Explicit OpenSearch client for vector storage/search."""

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


def create_vector_client(settings: Optional[Settings] = None) -> Tuple[MyOpenSearch, str]:
    """Create an explicit OpenSearch client for the recipe vector index."""

    settings = settings or load_settings()
    LOGGER.info(
        "Connecting to OpenSearch vector store at %s:%s (index=%s)",
        settings.opensearch_host,
        settings.opensearch_port,
        settings.opensearch_vector_index,
    )
    client = _build_client(
        host=settings.opensearch_host,
        port=settings.opensearch_port,
        user=settings.opensearch_user or None,
        password=settings.opensearch_password or None,
        ssl=bool(settings.opensearch_ssl),
        settings=settings,
    )
    return client, settings.opensearch_vector_index


def _metadata_filter_query(filters: List[Dict[str, Any]], must_not: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Build a reusable bool filter for constrained kNN queries."""

    clauses: Dict[str, Any] = {}
    if filters:
        clauses["filter"] = filters
    if must_not:
        clauses["must_not"] = must_not
    if not clauses:
        return None
    return {"bool": clauses}


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
    filter_in_knn: bool = True,
) -> Dict[str, Any]:
    """
    Build a filtered kNN query over recipe vector chunks.

    The graph stage can narrow retrieval to a tiny set of recipe IDs. Those
    filters need to live inside the kNN clause so OpenSearch applies them before
    nearest-neighbor selection. A bool wrapper around kNN can starve filtered
    recipes because the global nearest-neighbor set is chosen first.
    """

    k = int(k)
    if candidate_k is None or int(candidate_k) <= 0:
        candidate_k = max(k, 50)
    candidate_k = max(k, int(candidate_k))
    knn_body: Dict[str, Any] = {"vector": query_vector, "k": candidate_k}

    if VECTOR_USE_RESCORE and candidate_k > k:
        knn_body["rescore"] = {"oversample_factor": float(candidate_k) / float(max(k, 1))}

    filters: List[Dict[str, Any]] = []
    must_not: List[Dict[str, Any]] = []

    recipe_ids = [str(x).strip() for x in (include_recipe_ids or []) if str(x).strip()]
    if recipe_ids:
        filters.append({"terms": {"recipe_id": recipe_ids}})

    excluded = [str(x).strip().lower() for x in (exclude_cautions or []) if str(x).strip()]
    if excluded:
        must_not.append({"terms": {"cautions": excluded}})

    required_cautions = [str(x).strip().lower() for x in (require_cautions or []) if str(x).strip()]
    if required_cautions:
        filters.append({"terms": {"cautions": required_cautions}})

    required_health = [str(x).strip().lower() for x in (require_health_labels or []) if str(x).strip()]
    for label in required_health:
        filters.append({"term": {"health_labels": label}})

    filter_query = _metadata_filter_query(filters, must_not)
    if filter_query and filter_in_knn:
        knn_body["filter"] = filter_query
        query: Dict[str, Any] = {"knn": {field: knn_body}}
    elif filter_query:
        query = {
            "bool": {
                "must": [{"knn": {field: knn_body}}],
                "filter": filters,
                "must_not": must_not,
            }
        }
    else:
        query = {"knn": {field: knn_body}}

    return {
        "size": int(k),
        "_source": [
            "recipe_id",
            "recipe_name",
            "source",
            "url",
            "chunk_id",
            "chunk_index",
            "chunk_count",
            "text",
            "cautions",
            "health_labels",
            "diet_labels",
            "cuisine_type",
            "meal_type",
            "dish_type",
        ],
        "query": query,
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
        filter_in_knn=True,
    )

    def _search(search_body: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"index": index_name, "body": search_body, "request_timeout": 30}
        if client.settings.search_preference:
            kwargs["preference"] = client.settings.search_preference
        return client.search(**kwargs)

    try:
        res = _search(body)
    except TransportError as exc:
        if _uses_knn_filter(body, field):
            legacy_body = build_recipe_vector_query(
                query_vector,
                k=k,
                candidate_k=candidate_k,
                field=field,
                include_recipe_ids=include_recipe_ids,
                exclude_cautions=exclude_cautions,
                require_cautions=require_cautions,
                require_health_labels=require_health_labels,
                filter_in_knn=False,
            )
            try:
                res = _search(legacy_body)
                body = legacy_body
            except TransportError as legacy_exc:
                LOGGER.warning(
                    "OpenSearch vector query failed on %s/%s: %s",
                    label,
                    index_name,
                    getattr(legacy_exc, "error", str(legacy_exc)),
                )
                return {
                    "_store_label": label,
                    "_index_used": index_name,
                    "_query": _redact_vector(legacy_body, field),
                    "_error": f"{legacy_exc.__class__.__name__}: {getattr(legacy_exc, 'error', str(legacy_exc))}",
                    "_fallback_error": f"{exc.__class__.__name__}: {getattr(exc, 'error', str(exc))}",
                    "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []},
                }
        else:
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


def _uses_knn_filter(body: Dict[str, Any], field: str) -> bool:
    try:
        return "filter" in body["query"]["knn"][field]
    except Exception:
        return False


def _redact_vector(body: Dict[str, Any], field: str) -> Dict[str, Any]:
    """Return a JSON-ish query copy without dumping the embedding vector."""

    import copy

    redacted = copy.deepcopy(body)
    try:
        redacted["query"]["knn"][field]["vector"] = "<omitted>"
    except Exception:
        pass
    try:
        redacted["query"]["bool"]["must"][0]["knn"][field]["vector"] = "<omitted>"
    except Exception:
        pass
    return redacted


def normalize_vector_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
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
                "chunk_id": src.get("chunk_id") or "",
                "chunk_index": src.get("chunk_index"),
                "chunk_count": src.get("chunk_count"),
                "text": (src.get("text") or "").strip(),
                "cautions": src.get("cautions") or [],
                "health_labels": src.get("health_labels") or [],
                "diet_labels": src.get("diet_labels") or [],
            }
        )
    return out


__all__ = [
    "MyOpenSearch",
    "create_vector_client",
    "build_recipe_vector_query",
    "knn_search",
    "normalize_vector_hits",
    "VECTOR_FIELD",
]
