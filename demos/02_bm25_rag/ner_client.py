#!/usr/bin/env python3
"""Small command-line client for the local NER REST service."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Optional, Sequence

import requests

DEFAULT_URL = "http://127.0.0.1:8000/ner"


class NERServiceError(RuntimeError):
    """Raised when the NER service returns an invalid or unsuccessful response."""


class NERServiceClient:
    """Thin HTTP client used by demos and shell tests to call the NER service."""

    def __init__(
        self,
        *,
        url: str = DEFAULT_URL,
        timeout: float = 8.0,
        session: Optional[requests.sessions.Session] = None,
    ) -> None:
        cleaned_url = str(url).strip()
        if not cleaned_url:
            raise ValueError("NER endpoint URL is required")
        self.url = cleaned_url
        self.timeout = float(timeout)
        self.session = session or requests.Session()

    def post_ner(self, text: str, labels: Optional[Sequence[str]] = None) -> dict[str, Any]:
        """POST text to /ner and return the validated JSON object response."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        payload: dict[str, Any] = {"text": text}
        # The service treats labels as an optional allow-list of spaCy entity
        # types, such as ORG, PERSON, GPE, PRODUCT, or EVENT.
        normalized_labels = [str(label).strip() for label in labels or [] if str(label).strip()]
        if normalized_labels:
            payload["labels"] = normalized_labels

        response = self.session.post(self.url, json=payload, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                details: Any = response.json()
            except Exception:
                details = response.text
            raise NERServiceError(f"HTTP {response.status_code} from server: {details}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise NERServiceError(f"Server returned non-JSON response: {response.text[:2000]}") from exc

        if not isinstance(data, dict):
            raise NERServiceError("Server returned a non-object JSON payload")
        return data

    def extract_entities(self, text: str, labels: Optional[Sequence[str]] = None) -> list[str]:
        """Return a de-duplicated list of entity strings from the service payload."""
        data = self.post_ner(text, labels=labels)
        raw_entities = data.get("entities")
        if raw_entities is None:
            return []
        if not isinstance(raw_entities, list):
            raise NERServiceError("Server returned an invalid 'entities' payload")

        seen: set[str] = set()
        entities: list[str] = []
        for item in raw_entities:
            if not isinstance(item, str):
                continue
            entity = item.strip()
            if not entity or entity in seen:
                continue
            seen.add(entity)
            entities.append(entity)
        return entities


def read_stdin() -> str:
    """Read piped text when --text is not provided."""
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No input on stdin. Pass --text or pipe text to stdin.")
    return data


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without performing any network work."""
    ap = argparse.ArgumentParser(description="Client for the local NER REST API")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"NER endpoint URL (default: {DEFAULT_URL})")
    ap.add_argument("--text", help="Text to analyze. If omitted, reads from stdin.")
    ap.add_argument("--labels", nargs="*", help="Optional entity labels to request from the service.")
    ap.add_argument("--timeout", type=float, default=8.0, help="Request timeout in seconds (default: 8.0)")
    ap.add_argument("--json", action="store_true", help="Print the full JSON response instead of a summary.")
    return ap


def main() -> None:
    """Run one NER request and print either the raw JSON or a compact summary."""
    args = build_arg_parser().parse_args()
    text = args.text if args.text is not None else read_stdin()

    client = NERServiceClient(url=args.url, timeout=args.timeout)
    response = client.post_ner(text, labels=args.labels)

    if args.json:
        print(json.dumps(response, indent=2, sort_keys=True))
        return

    model = response.get("model")
    request_id = response.get("request_id")
    entities = response.get("entities", [])

    print(f"Model:        {model}")
    print(f"Request ID:   {request_id}")
    print(f"Entity Count: {len(entities)}")
    print("Entities:")
    for i, ent in enumerate(entities, 1):
        print(f"  {i:2d}. {ent}")


if __name__ == "__main__":
    main()
