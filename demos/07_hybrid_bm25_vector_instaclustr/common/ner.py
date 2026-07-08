"""Small client for the external recipe NER service."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional

import requests

from .config import Settings, load_settings
from .labels import normalize_values
from .logging import get_logger


LOGGER = get_logger(__name__)


def _service_endpoint(url: str) -> str:
    value = str(url or "").strip().rstrip("/")
    if not value:
        value = Settings.ner_service_url.rstrip("/")
    if value.endswith("/ner"):
        return value
    return f"{value}/ner"


class NERClient:
    """Call the external NER service and normalize the returned entity list."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        service_url: Optional[str] = None,
        timeout: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.settings = settings or load_settings()
        self.service_url = _service_endpoint(service_url or self.settings.ner_service_url)
        self.timeout = float(timeout if timeout is not None else self.settings.ner_timeout_secs)
        self.session = session or requests.Session()

    def extract_entities(self, text: str, *, labels: Optional[Iterable[str]] = None) -> List[str]:
        """Return normalized entity strings for ``text``.

        The service returns a plain ``entities`` list. Failures are treated as an
        empty list so ingestion can continue, because a single flaky HTTP call
        should not turn the pipeline into performance art.
        """

        payload: dict[str, Any] = {"text": str(text or "")}
        label_list = [str(label).strip() for label in (labels or []) if str(label).strip()]
        if label_list:
            payload["labels"] = label_list

        try:
            response = self.session.post(self.service_url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            LOGGER.warning("NER request failed for %d chars: %s", len(payload["text"]), exc)
            return []

        entities = data.get("entities") if isinstance(data, dict) else []
        if not isinstance(entities, list):
            return []
        return normalize_values(entities)


__all__ = ["NERClient"]
