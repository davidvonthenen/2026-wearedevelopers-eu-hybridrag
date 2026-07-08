"""OpenSearch client utilities."""
from __future__ import annotations

from typing import List, Optional

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError

from .config import Settings
from .logging import get_logger


LOGGER = get_logger(__name__)


def create_client(settings: Settings) -> OpenSearch:
    """Create an unsecured HTTP OpenSearch client from runtime settings."""
    LOGGER.info(
        "Connecting to OpenSearch at %s:%s", settings.opensearch_host, settings.opensearch_port
    )
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port, "scheme": "http"}],
        http_compress=True,
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
    )


def ensure_index(settings: Settings, dim: int) -> None:
    """Create the vector index when it does not already exist.

    Existing indexes are left untouched; this function does not validate or
    migrate mappings. Delete the index manually when changing embedding models
    with a different vector dimension.
    """
    client = create_client(settings)
    index_name = settings.opensearch_index
    try:
        if client.indices.exists(index=index_name):
            LOGGER.info("OpenSearch index '%s' already exists", index_name)
            return
    except NotFoundError:
        LOGGER.info("OpenSearch index '%s' does not exist, creating", index_name)

    body = {
        "settings": {
            "index": {
                # Enable OpenSearch k-NN support for the vector field mapping below.
                "knn": True,
                # Retain a generic k-NN search tuning value. Exact behavior is
                # engine/version-specific, so the query still controls returned k.
                "knn.algo_param.ef_search": 256,
            }
        },
        "mappings": {
            "properties": {
                "path": {"type": "keyword"},
                "title": {"type": "keyword"},
                "category": {"type": "keyword"},
                "text": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                        # Add HNSW construction parameters here if you need to tune
                        # indexing behavior for a larger corpus.
                    },
                },
            }
        },
    }
    LOGGER.info("Creating OpenSearch index '%s'", index_name)
    client.indices.create(index=index_name, body=body)


def knn_search(
    client: OpenSearch,
    index: str,
    query_vec: List[float],
    k: int,
    num_candidates: Optional[int] = None,
):
    """
    Execute a vector k-NN search against the 'embedding' field.

    OpenSearch 3.x k-NN DSL uses the vector field name as the key and
    expects the query vector under 'vector' (not 'query_vector'), and no 'field' key.
    """
    # OpenSearch 3.x vector k-NN query body. This project retrieves only the
    # fields needed for prompting and compact CLI display.
    body = {
        "size": k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vec,
                    "k": k,
                }
            }
        },
        "_source": ["path", "title", "category", "text"],
    }

    # This is not the old "num_candidates" query parameter from other vector
    # stores. For OpenSearch here, a larger value is represented as rescore
    # oversampling when the caller asks for more candidates than final results.
    if num_candidates and num_candidates > k:
        body["query"]["knn"]["embedding"]["rescore"] = {
            "oversample_factor": float(num_candidates) / float(max(k, 1))
        }

    response = client.search(index=index, body=body)
    LOGGER.info("OpenSearch knn_search took %.2f ms", response.get("took", 0.0))
    return response


__all__ = ["create_client", "ensure_index", "knn_search"]
