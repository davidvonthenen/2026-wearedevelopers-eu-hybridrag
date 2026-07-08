"""Runtime configuration loaded from environment variables.

The defaults target the local OpenSearch container used in the workshop, while
environment variables allow the same code to point at another deployment.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _get_bool(name: str, default: bool = False) -> bool:
    """Parse common truthy environment variable strings."""

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass
class Settings:
    """Connection and model settings for recipe vector ingestion/search."""

    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_ssl: bool = False
    opensearch_vector_index: str = "recipes-vector"
    search_preference: str = ""

    # Embeddings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # alt: thenlper/gte-small
    embedding_dimension: Optional[int] = None

    # Recipe page fetching
    recipe_http_timeout_secs: float = 15.0
    recipe_http_user_agent: str = "recipe-hybrid-rag/1.0 (+https://example.local; contact=dev-null)"


def load_settings() -> Settings:
    """Load settings from environment variables with local-dev defaults."""

    return Settings(
        opensearch_host=_get_str("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_user=_get_str("OPENSEARCH_USER", Settings.opensearch_user),
        opensearch_password=_get_str("OPENSEARCH_PASS", Settings.opensearch_password),
        opensearch_ssl=_get_bool("OPENSEARCH_SSL", Settings.opensearch_ssl),
        opensearch_vector_index=_get_str("OPENSEARCH_VECTOR_INDEX", Settings.opensearch_vector_index),
        search_preference=_get_str("OPENSEARCH_SEARCH_PREFERENCE", Settings.search_preference),

        # Embeddings
        embedding_model=_get_str("EMBEDDING_MODEL", _get_str("EMBEDDING_MODEL", Settings.embedding_model)),
        embedding_dimension=_get_optional_int("EMBEDDING_DIMENSION", Settings.embedding_dimension),

        # Fetch Recipes
        recipe_http_timeout_secs=_get_float("RECIPE_HTTP_TIMEOUT_SECS", Settings.recipe_http_timeout_secs),
        recipe_http_user_agent=_get_str("RECIPE_HTTP_USER_AGENT", Settings.recipe_http_user_agent),
    )


__all__ = [
    "Settings",
    "load_settings",
    "_get_str",
    "_get_int",
    "_get_float",
    "_get_bool",
]
