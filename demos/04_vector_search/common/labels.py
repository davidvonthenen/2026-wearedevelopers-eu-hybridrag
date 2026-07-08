"""Shared label normalization helpers for OpenSearch keyword metadata."""
from __future__ import annotations

import re
from typing import Any, Iterable, List


_LABEL_SPACE_RE = re.compile(r"\s+")


def normalize_key(value: Any) -> str:
    """Normalize a human label for deterministic keyword filtering."""

    text = str(value or "").strip().lower()
    text = _LABEL_SPACE_RE.sub(" ", text)
    return text


def normalize_values(values: Iterable[Any]) -> List[str]:
    """Normalize and de-duplicate a sequence of labels while preserving order."""

    out: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        key = normalize_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


__all__ = ["normalize_key", "normalize_values"]
