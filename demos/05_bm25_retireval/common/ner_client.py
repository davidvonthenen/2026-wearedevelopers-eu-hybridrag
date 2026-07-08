"""Small client for the external NER service used by BM25 enrichment."""
from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional

import requests

from .labels import normalize_values


def _ner_endpoint(url: str) -> str:
    """Return a usable /ner endpoint from either a base URL or full endpoint URL."""

    cleaned = str(url or "").strip().rstrip("/")
    if not cleaned:
        return "http://127.0.0.1:8000/ner"
    if cleaned.endswith("/ner"):
        return cleaned
    return f"{cleaned}/ner"


def extract_named_entities(
    text: str,
    *,
    url: str,
    timeout: float,
    labels: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Call the external NER service and return normalized entity strings.

    The service normally returns lowercased strings, but this client normalizes
    again so ingestion and query code receive deterministic keyword values even
    if the service implementation changes.
    """

    text = str(text or "")
    if not text.strip():
        return []

    payload: dict[str, Any] = {"text": text}
    label_list = [str(label).strip() for label in labels or [] if str(label).strip()]
    if label_list:
        payload["labels"] = label_list

    endpoint = _ner_endpoint(url)
    try:
        response = requests.post(endpoint, json=payload, timeout=float(timeout))
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        if logger is not None:
            logger.warning("NER request failed for %s: %s", endpoint, exc)
        return []

    entities = body.get("entities") if isinstance(body, dict) else None
    if not isinstance(entities, list):
        if logger is not None:
            logger.warning("NER response from %s did not contain an entities list", endpoint)
        return []

    return normalize_values(entities)


__all__ = ["extract_named_entities"]
