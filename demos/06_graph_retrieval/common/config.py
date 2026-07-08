"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


def _get_str(name: str, default: str = "") -> str:
    """Read a string environment variable with whitespace trimming."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    """Read an integer environment variable with a default fallback."""

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _get_float(name: str, default: float) -> float:
    """Read a floating-point environment variable with a default fallback."""

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _get_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable from common truthy string values."""

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_optional_int(name: str) -> Optional[int]:
    """Read an optional integer environment variable, returning ``None`` if unset."""

    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


@dataclass
class Settings:
    """Connection and HTTP settings for recipe graph ingestion/search."""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4jneo4j"
    neo4j_database: str = "neo4j"

    # HTTP fetch settings for recipe page enrichment.
    recipe_http_timeout_secs: float = 15.0
    recipe_http_user_agent: str = "recipe-graph-search/1.0 (+https://example.local; contact=dev-null)"

def load_settings() -> Settings:
    """Load settings from environment variables with local-dev defaults."""

    return Settings(
        neo4j_uri=_get_str("NEO4J_URI", Settings.neo4j_uri),
        neo4j_user=_get_str("NEO4J_LONG_USER", Settings.neo4j_user),
        neo4j_password=_get_str("NEO4J_PASSWORD", Settings.neo4j_password),
        neo4j_database=_get_str("NEO4J_DATABASE", Settings.neo4j_database),

        # HTTP fetch settings for recipe page enrichment.
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
