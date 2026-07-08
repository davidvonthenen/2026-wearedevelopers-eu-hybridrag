"""Shared lightweight data models for recipe Hybrid RAG."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RetrievalHit:
    """Normalized evidence item passed to prompts and audit output.

    ``channel`` separates graph truth grounding from vector semantic support.
    ``handle`` is the citation token without brackets, for example ``G1`` or
    ``V2``. Recipe-specific metadata is optional so the same object can hold a
    full graph recipe record, a vector chunk, or a minimal fallback item.
    """

    channel: str = ""  # graph_recipe | vector
    handle: str = ""   # G1..Gn or V1..Vn
    index: str = ""
    os_id: str = ""
    score: float = 0.0
    path: str = ""
    category: str = ""
    chunk_index: Optional[int] = None
    chunk_count: Optional[int] = None

    text: str = ""

    recipe_id: str = ""
    recipe_name: str = ""
    source: str = ""
    url: str = ""
    image_url: str = ""
    servings: Optional[int] = None
    calories: Optional[float] = None
    chunk_id: str = ""

    explicit_terms: Optional[List[str]] = None
    entity_overlap: Optional[int] = None
    meta: Optional[Dict[str, Any]] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["RetrievalHit"]
