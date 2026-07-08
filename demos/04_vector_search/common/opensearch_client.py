"""OpenSearch vector client utilities for recipe semantic search.

This module builds the OpenSearch client, constructs the pure kNN query used by
``query.py``, executes the search, and normalizes raw hits for display.
"""
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
    """OpenSearch client subclass that carries the resolved runtime settings."""

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
    # OpenSearch accepts unauthenticated HTTP for the local lab container and
    # basic auth/HTTPS for secured deployments.
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
        timeout=10,
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


def build_recipe_vector_query(
    query_vector: List[float],
    *,
    k: int,
    candidate_k: Optional[int] = None,
    field: str = VECTOR_FIELD,
    exclude_cautions: Optional[Iterable[str]] = None,
    require_cautions: Optional[Iterable[str]] = None,
    require_health_labels: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Build the pure kNN query body used for recipe vector chunks.

    The metadata constraint parameters are accepted for interface compatibility
    with later retrieval examples, but this vector-only query does not apply
    them as OpenSearch filters. That is intentional for this lab: it lets users
    see how dense retrieval behaves before Boolean constraints rescue it from
    its own geometric optimism.
    """

    # ``candidate_k`` is sent as the kNN clause's candidate count, while the
    # top-level ``size`` controls how many hits OpenSearch returns.
    candidate_k = int(candidate_k or max(k, 50))
    knn_body: Dict[str, Any] = {"vector": query_vector, "k": candidate_k}

    if VECTOR_USE_RESCORE and candidate_k > k:
        # Optional oversampling asks OpenSearch to score a larger candidate set
        # before returning the requested number of hits.
        knn_body["rescore"] = {"oversample_factor": float(candidate_k) / float(max(k, 1))}

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
        "query": {"knn": {field: knn_body}},
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
        exclude_cautions=exclude_cautions,
        require_cautions=require_cautions,
        require_health_labels=require_health_labels,
    )

    try:
        # ``preference`` is optional but useful when someone wants repeatable
        # shard routing during demos or experiments. Because apparently even
        # search results enjoy freelancing.
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
    """Return a diagnostic query copy.

    The guarded mutation below only redacts older bool-wrapped query shapes. The
    direct ``query.knn`` body used by this demo is returned unchanged, which is
    acceptable here because the diagnostic query is not printed by the CLI.
    """

    import copy

    redacted = copy.deepcopy(body)
    try:
        redacted["query"]["bool"]["must"][0]["knn"][field]["vector"] = "<omitted>"
    except Exception:
        pass
    return redacted


def normalize_vector_hits(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize raw OpenSearch hits into compact recipe chunk dictionaries."""

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
