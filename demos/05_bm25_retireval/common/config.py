"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os


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
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    """Connection and service settings for recipe BM25 ingestion/search."""

    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_ssl: bool = False
    opensearch_bm25_index: str = "recipes-bm25"
    search_preference: str = ""

    # NER service
    ner_service_url: str = "http://127.0.0.1:8000/ner"
    ner_http_timeout_secs: float = 10.0

    # Recipe page fetching
    recipe_http_timeout_secs: float = 15.0
    recipe_http_user_agent: str = "recipe-bm25-rag/1.0 (+https://example.local; contact=dev-null)"


def load_settings() -> Settings:
    """Load settings from environment variables with local-dev defaults."""

    return Settings(
        opensearch_host=_get_str("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_user=_get_str("OPENSEARCH_USER", Settings.opensearch_user),
        opensearch_password=_get_str("OPENSEARCH_PASS", Settings.opensearch_password),
        opensearch_ssl=_get_bool("OPENSEARCH_SSL", Settings.opensearch_ssl),
        opensearch_bm25_index=_get_str("OPENSEARCH_BM25_INDEX", Settings.opensearch_bm25_index),
        search_preference=_get_str("OPENSEARCH_SEARCH_PREFERENCE", Settings.search_preference),

        # NER service
        ner_service_url=_get_str("NER_SERVICE_URL", Settings.ner_service_url),
        ner_http_timeout_secs=_get_float("NER_HTTP_TIMEOUT_SECS", Settings.ner_http_timeout_secs),

        # Recipe page fetching
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
