"""Shared helpers for the Graph-based RAG demo scripts."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests
from neo4j import Driver, GraphDatabase, Session

NER_SERVICE_URL = os.getenv("NER_SERVICE_URL", "http://127.0.0.1:8000/ner")
DEFAULT_TIMEOUT = float(os.getenv("NER_SERVICE_TIMEOUT", "8.0"))

EntityPair = Tuple[str, str]


class NerServiceError(RuntimeError):
    """Raised when the NER service returns an error response."""


def _build_payload(
    text: str,
    labels: Optional[Sequence[str]] = None,
    promote: Optional[bool] = None,
    ttl_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the JSON body sent to the NER service.

    `promote` and `ttl_ms` are retained for call-shape compatibility with other
    lab variants. The local `ner_service.py` in this project ignores them.
    """
    payload: Dict[str, Any] = {"text": text}
    if labels is not None:
        payload["labels"] = list(labels)
    if promote is not None:
        payload["promote"] = bool(promote)
    if ttl_ms is not None:
        payload["ttl_ms"] = int(ttl_ms)
    return payload


def call_ner_service(
    text: str,
    *,
    labels: Optional[Sequence[str]] = None,
    promote: Optional[bool] = None,
    ttl_ms: Optional[int] = None,
    timeout: Optional[float] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the NER REST service and return the decoded JSON payload.

    Args:
        text: Free-form input string to analyse.
        labels: Optional whitelist of entity labels (PERSON, ORG, …).
        promote: Compatibility flag forwarded in the request body; ignored by
            the local NER service in this project.
        ttl_ms: Compatibility TTL value forwarded in the request body; ignored
            by the local NER service in this project.
        timeout: Requests timeout in seconds (defaults to ``NER_SERVICE_TIMEOUT``).
        url: Optional override for the service endpoint.

    Raises:
        NerServiceError: if the service returns a non-2xx status or invalid JSON.

    Returns:
        Parsed JSON body from the NER service.
    """

    endpoint = url or NER_SERVICE_URL
    payload = _build_payload(text, labels=labels, promote=promote, ttl_ms=ttl_ms)
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout or DEFAULT_TIMEOUT,
        )
    except requests.RequestException as exc:  # pragma: no cover - passthrough
        raise NerServiceError(f"Failed to reach NER service: {exc}") from exc

    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        raise NerServiceError(
            f"NER service returned HTTP {response.status_code}: {detail}"
        )

    try:
        return response.json()
    except Exception as exc:  # pragma: no cover - unexpected JSON shape
        raise NerServiceError(
            f"NER service produced invalid JSON: {response.text[:2000]}"
        ) from exc


def parse_entity_pairs(data: Mapping[str, Any]) -> List[EntityPair]:
    """Extract normalized ``(name, label)`` tuples from a NER response.

    Prefer the structured `entity_pairs` field. Fall back to the older `entities`
    list when necessary so callers can tolerate slightly different service
    payloads without requiring each caller to duplicate parsing logic.
    """
    pairs: List[EntityPair] = []

    details = data.get("entity_pairs")
    if isinstance(details, list):
        for item in details:
            if isinstance(item, Mapping):
                name = str(item.get("name", "")).strip().lower()
                label = str(item.get("label", "")).strip().upper()
                if name and label:
                    pairs.append((name, label))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                name = str(item[0]).strip().lower()
                label = str(item[1]).strip().upper()
                if name and label:
                    pairs.append((name, label))

    if not pairs:
        entities = data.get("entities")
        if isinstance(entities, list):
            for name in entities:
                name_str = str(name).strip().lower()
                if name_str:
                    pairs.append((name_str, ""))

    # Deduplicate while preserving order.
    seen = set()
    unique_pairs: List[EntityPair] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs


def detect_entities(
    text: str,
    *,
    labels: Optional[Sequence[str]] = None,
    promote: bool = True,
    ttl_ms: Optional[int] = None,
    timeout: Optional[float] = None,
    url: Optional[str] = None,
) -> List[EntityPair]:
    """Call the NER service and return parsed entity pairs directly."""
    payload = call_ner_service(
        text,
        labels=labels,
        promote=promote,
        ttl_ms=ttl_ms,
        timeout=timeout,
        url=url,
    )
    return parse_entity_pairs(payload)


def create_indexes(session: Session) -> None:
    """Ensure the Neo4j indexes used by ingest and query paths are online.

    The entity-name/label index supports lookup by extracted question entities.
    UUID indexes support deterministic node matching when creating relationships.
    """

    session.run(
        """
        CREATE RANGE INDEX ent_name_label IF NOT EXISTS
        FOR (e:Entity) ON (e.name, e.label)
        """
    )
    session.run(
        """
        CREATE RANGE INDEX ent_uuid_idx IF NOT EXISTS
        FOR (e:Entity) ON (e.ent_uuid)
        """
    )
    session.run(
        """
        CREATE RANGE INDEX para_uuid_idx IF NOT EXISTS
        FOR (p:Paragraph) ON (p.para_uuid)
        """
    )
    session.run(
        """
        CREATE RANGE INDEX doc_uuid_idx IF NOT EXISTS
        FOR (d:Document) ON (d.doc_uuid)
        """
    )
    session.run("CALL db.awaitIndexes()")


def build_driver(
    uri_env: str,
    user_env: str,
    password_env: str,
    default_uri: str,
    *,
    default_user: str = "neo4j",
    default_password: str = "neo4j",
) -> Driver:
    """Construct a Neo4j driver from named environment variables.

    This helper is not required by every script, but it keeps configuration
    patterns consistent when a demo needs more than one graph connection.
    """

    return GraphDatabase.driver(
        os.getenv(uri_env, default_uri),
        auth=(
            os.getenv(user_env, default_user),
            os.getenv(password_env, default_password),
        ),
    )
