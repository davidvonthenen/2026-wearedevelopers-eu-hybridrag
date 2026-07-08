#!/usr/bin/env python3
"""Recipe vector ingestion: OpenSearch vector chunks plus stored metadata.

This ingests recipe CSV rows with webpage text enrichment:

* OpenSearch stores dense vector chunks generated from human-readable recipe text.
* Each chunk carries structured recipe metadata as keyword fields for later filtering
  or inspection.
* URL text extraction is attempted for each recipe with bounded exponential backoff.
* Recipes with unavailable or unreadable pages are skipped so the vector store does not
  contain partial metadata-only records.

Embeddings handle semantic intent. Metadata is stored alongside each chunk so future
or downstream retrieval stages can apply hard constraints instead of asking vectors
to remember that "not peanuts" means "not peanuts," which is how software quietly
becomes a snack-themed liability hearing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import requests
from opensearchpy import OpenSearch
from tqdm import tqdm

try:  # BeautifulSoup is optional; the fallback parser keeps the code dependency-light.
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - depends on runtime environment
    BeautifulSoup = None  # type: ignore

from common.embeddings import EmbeddingModel, to_list
from common.labels import normalize_key, normalize_values
from common.logging import get_logger
from common.opensearch_client import MyOpenSearch, create_vector_client


LOGGER = get_logger(__name__)

SPACE_RE = re.compile(r"[ \t\x0b\x0c\xa0]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")
SCRIPT_TAGS = {"script", "style", "noscript", "template", "svg", "canvas"}
NOISY_TAGS = {"nav", "header", "footer", "form", "aside", "iframe"}
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 410}


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments that control CSV ingest, fetching, and chunking."""

    parser = argparse.ArgumentParser(
        description="Recipe vector ingestion: OpenSearch vector chunks plus filterable metadata"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="recipes-with-nutrition-sample.csv",
        help="Path to the recipe CSV file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding/indexing batch size for vector chunks.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Target maximum characters per vector chunk, before metadata header overhead.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=256,
        help="Characters of overlap between consecutive vector chunks.",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=float,
        default=None,
        help="Per-attempt HTTP timeout in seconds. Defaults to RECIPE_HTTP_TIMEOUT_SECS.",
    )
    parser.add_argument(
        "--fetch-max-attempts",
        type=int,
        default=3,
        help="Maximum URL fetch attempts before skipping a recipe.",
    )
    parser.add_argument(
        "--fetch-max-backoff-seconds",
        type=float,
        default=15.0,
        help="Maximum total retry backoff sleep per URL.",
    )
    parser.add_argument(
        "--max-page-chars",
        type=int,
        default=60000,
        help="Maximum extracted page characters to keep per recipe before chunking.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Recipe model and CSV loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeRecord:
    """Normalized recipe metadata loaded from one CSV row.

    The raw display labels are preserved for human-readable chunk text, while
    normalized variants are exposed through properties for stable keyword fields.
    """

    recipe_id: str
    row_number: int
    recipe_name: str
    source: str
    url: str
    servings: Optional[int]
    calories: Optional[float]
    image_url: str
    diet_labels: List[str]
    health_labels: List[str]
    cautions: List[str]
    cuisine_type: List[str]
    meal_type: List[str]
    dish_type: List[str]

    @property
    def normalized_cautions(self) -> List[str]:
        return normalize_values(self.cautions)

    @property
    def normalized_health_labels(self) -> List[str]:
        return normalize_values(self.health_labels)

    @property
    def normalized_diet_labels(self) -> List[str]:
        return normalize_values(self.diet_labels)

    @property
    def normalized_cuisine_type(self) -> List[str]:
        return normalize_values(self.cuisine_type)

    @property
    def normalized_meal_type(self) -> List[str]:
        return normalize_values(self.meal_type)

    @property
    def normalized_dish_type(self) -> List[str]:
        return normalize_values(self.dish_type)


def parse_json_list(value: Any) -> List[str]:
    """Parse CSV list columns stored as JSON arrays."""

    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "none"}:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Defensive fallback for dirty CSVs. Because CSV and JSON had a child and
        # nobody taught it manners.
        parsed = [x.strip() for x in text.split(",")]

    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def parse_int(value: Any) -> Optional[int]:
    """Parse numeric CSV values that may arrive as empty strings or floats."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "none"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(value: Any) -> Optional[float]:
    """Parse optional floating-point CSV values without raising on blanks."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def stable_recipe_id(row: Mapping[str, Any], row_number: int) -> str:
    """Create a stable ID from URL when available, else source/name/row."""

    key = str(row.get("url") or "").strip()
    if not key:
        key = f"{row.get('source') or ''}|{row.get('recipe_name') or ''}|row:{row_number}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def iter_recipe_records(csv_path: Path) -> Iterator[RecipeRecord]:
    """Yield normalized recipe records from the CSV header/row structure."""

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            yield RecipeRecord(
                recipe_id=stable_recipe_id(row, row_number),
                row_number=row_number,
                recipe_name=str(row.get("recipe_name") or "").strip(),
                source=str(row.get("source") or "").strip(),
                url=str(row.get("url") or "").strip(),
                servings=parse_int(row.get("servings")),
                calories=parse_float(row.get("calories")),
                image_url=str(row.get("image_url") or "").strip(),
                diet_labels=parse_json_list(row.get("diet_labels")),
                health_labels=parse_json_list(row.get("health_labels")),
                cautions=parse_json_list(row.get("cautions")),
                cuisine_type=parse_json_list(row.get("cuisine_type")),
                meal_type=parse_json_list(row.get("meal_type")),
                dish_type=parse_json_list(row.get("dish_type")),
            )


# --------------------------------------------------------------------------------------
# Human-readable recipe text extraction
# --------------------------------------------------------------------------------------


class VisibleTextParser(HTMLParser):
    """Small fallback visible-text extractor when BeautifulSoup is unavailable."""

    def __init__(self) -> None:
        super().__init__()
        # Tracks nested script/style-like tags so their text is not indexed.
        self._skip_depth = 0
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag in SCRIPT_TAGS:
            self._skip_depth += 1
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in SCRIPT_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data)

    def text(self) -> str:
        return clean_text(" ".join(self.parts))


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving useful paragraph breaks."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]

    out: List[str] = []
    last_blank = False
    for line in lines:
        if not line:
            if out and not last_blank:
                out.append("")
            last_blank = True
            continue
        out.append(line)
        last_blank = False

    cleaned = "\n".join(out).strip()
    cleaned = BLANK_LINE_RE.sub("\n\n", cleaned)
    return cleaned


def dedupe_lines(text: str) -> str:
    """Remove repeated lines while preserving order."""

    seen: set[str] = set()
    out: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        key = normalize_key(line)
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return clean_text("\n".join(out))


def _json_loads_maybe(text: str) -> Any:
    """Best-effort JSON parse for JSON-LD script bodies."""

    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some sites wrap JSON-LD in comments or leave trailing junk. Strip the
        # most common garbage without pretending this is a JavaScript parser.
        text = text.strip("\ufeff \t\n\r")
        text = re.sub(r"^<!--", "", text).strip()
        text = re.sub(r"-->$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


def _type_contains_recipe(value: Any) -> bool:
    """Return True when a JSON-LD @type value identifies a Recipe."""

    if isinstance(value, str):
        return value.lower() == "recipe"
    if isinstance(value, list):
        return any(_type_contains_recipe(v) for v in value)
    return False


def _iter_recipe_jsonld(value: Any) -> Iterator[Mapping[str, Any]]:
    """Walk JSON-LD structures and yield nodes typed as recipes."""

    if isinstance(value, Mapping):
        if _type_contains_recipe(value.get("@type") or value.get("type")):
            yield value
        for child_key in ("@graph", "graph", "itemListElement", "mainEntity"):
            child = value.get(child_key)
            if child is not None:
                yield from _iter_recipe_jsonld(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_recipe_jsonld(item)


def _instruction_text(value: Any) -> List[str]:
    """Flatten schema.org instruction shapes into readable instruction text."""

    if value is None:
        return []
    if isinstance(value, str):
        text = clean_text(value)
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_instruction_text(item))
        return out
    if isinstance(value, Mapping):
        if value.get("text"):
            return _instruction_text(value.get("text"))
        if value.get("itemListElement"):
            return _instruction_text(value.get("itemListElement"))
        if value.get("name"):
            return _instruction_text(value.get("name"))
    return []


def _string_list(value: Any) -> List[str]:
    """Normalize schema.org scalar/list/name objects into strings."""

    if value is None:
        return []
    if isinstance(value, str):
        text = clean_text(value)
        return [text] if text else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            out.extend(_string_list(item))
        return out
    if isinstance(value, Mapping) and value.get("name"):
        return _string_list(value.get("name"))
    return [clean_text(str(value))] if str(value).strip() else []


def _render_jsonld_recipe(recipe: Mapping[str, Any]) -> str:
    """Render selected JSON-LD recipe fields into plain text for embedding."""

    sections: List[str] = []

    def add_scalar(label: str, key: str) -> None:
        value = recipe.get(key)
        if value:
            rendered = clean_text(str(value))
            if rendered:
                sections.append(f"{label}: {rendered}")

    add_scalar("Recipe", "name")
    add_scalar("Description", "description")
    add_scalar("Recipe category", "recipeCategory")
    add_scalar("Recipe cuisine", "recipeCuisine")
    add_scalar("Yield", "recipeYield")
    add_scalar("Prep time", "prepTime")
    add_scalar("Cook time", "cookTime")
    add_scalar("Total time", "totalTime")

    ingredients = _string_list(recipe.get("recipeIngredient"))
    if ingredients:
        sections.append("Ingredients:\n" + "\n".join(f"- {x}" for x in ingredients))

    instructions = _instruction_text(recipe.get("recipeInstructions"))
    if instructions:
        sections.append("Instructions:\n" + "\n".join(f"{idx}. {text}" for idx, text in enumerate(instructions, start=1)))

    return clean_text("\n\n".join(sections))


def extract_jsonld_recipe_text(html: str) -> str:
    """Extract recipe-specific JSON-LD text when BeautifulSoup is available."""

    if BeautifulSoup is None:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    rendered: List[str] = []
    for script in soup.find_all("script"):
        script_type = str(script.get("type") or "").lower()
        if "ld+json" not in script_type:
            continue
        payload = _json_loads_maybe(script.string or script.get_text(" ", strip=False))
        for recipe in _iter_recipe_jsonld(payload):
            text = _render_jsonld_recipe(recipe)
            if text:
                rendered.append(text)
    return dedupe_lines("\n\n".join(rendered))


def extract_visible_text(html: str) -> str:
    """Extract visible page text, dropping script and layout-heavy tags."""

    if BeautifulSoup is None:
        parser = VisibleTextParser()
        parser.feed(html)
        return parser.text()

    soup = BeautifulSoup(html, "html.parser")
    for tag_name in SCRIPT_TAGS | NOISY_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Prefer content-like tags over raw body text. Recipe sites love turning pages
    # into SEO confetti, so this keeps the noise at least slightly less tragic.
    lines: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = clean_text(tag.get_text(" ", strip=True))
        if text:
            lines.append(text)

    if len("\n".join(lines)) < 500:
        body = soup.body or soup
        fallback = body.get_text("\n", strip=True)
        lines.append(fallback)

    return dedupe_lines("\n".join(lines))


def extract_human_readable_text(html: str) -> str:
    """Combine structured JSON-LD recipe text with visible page text."""

    jsonld_text = extract_jsonld_recipe_text(html)
    visible_text = extract_visible_text(html)

    if jsonld_text and visible_text:
        return dedupe_lines(f"{jsonld_text}\n\nVisible page text:\n{visible_text}")
    return jsonld_text or visible_text


def _retry_after_seconds(response: requests.Response, *, remaining_backoff: float) -> Optional[float]:
    """Parse Retry-After while respecting the per-URL backoff budget."""

    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        # HTTP-date Retry-After is valid, but deliberately ignored here to keep
        # the ingest loop bounded and readable. Calendars: the final boss of HTTP.
        return None
    if delay < 0:
        return None
    return min(delay, max(0.0, remaining_backoff))


def _next_backoff_seconds(
    attempt_index: int,
    *,
    response: Optional[requests.Response],
    total_slept: float,
    max_total_backoff_seconds: float,
) -> float:
    """Compute the next retry delay within the per-URL backoff budget."""

    remaining = max(0.0, float(max_total_backoff_seconds) - total_slept)
    if remaining <= 0:
        return 0.0

    if response is not None:
        retry_after = _retry_after_seconds(response, remaining_backoff=remaining)
        if retry_after is not None:
            return retry_after

    # 1s, 2s, 4s... capped by the remaining per-URL backoff budget.
    return min(float(2 ** max(0, attempt_index - 1)), remaining)


def _response_error(response: requests.Response) -> str:
    """Format an HTTP response failure without raising it to the caller."""

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        return f"HTTPError: {exc}"
    return f"HTTP {response.status_code}"


def fetch_url_human_text(
    url: str,
    *,
    timeout: float,
    user_agent: str,
    max_chars: int,
    max_attempts: int = 3,
    max_total_backoff_seconds: float = 15.0,
) -> Tuple[str, str, str]:
    """Fetch a URL and extract human-readable recipe text.

    Returns ``(text, fetch_status, fetch_error)``. The function retries transient
    failures with bounded exponential backoff. Callers should skip recipes when
    ``text`` is empty.
    """

    if not url:
        return "", "missing_url", ""
    if not url.startswith(("http://", "https://")):
        return "", "invalid_url", f"URL does not start with http/https: {url}"

    attempts = max(1, int(max_attempts))
    max_backoff = max(0.0, float(max_total_backoff_seconds))
    headers = {"User-Agent": user_agent}
    total_slept = 0.0
    last_status = "fetch_error"
    last_error = ""

    for attempt in range(1, attempts + 1):
        response: Optional[requests.Response] = None
        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            status = f"http_{response.status_code}"
            last_status = status

            if response.status_code in NON_RETRYABLE_HTTP_STATUS_CODES:
                return "", status, _response_error(response)

            if response.status_code >= 400:
                last_error = _response_error(response)
                if response.status_code not in RETRYABLE_HTTP_STATUS_CODES or attempt >= attempts:
                    return "", status, last_error

                delay = _next_backoff_seconds(
                    attempt,
                    response=response,
                    total_slept=total_slept,
                    max_total_backoff_seconds=max_backoff,
                )
                if delay > 0:
                    LOGGER.debug(
                        "URL fetch attempt %d/%d failed with %s for %s; retrying in %.1fs",
                        attempt,
                        attempts,
                        status,
                        url,
                        delay,
                    )
                    time.sleep(delay)
                    total_slept += delay
                continue

            html = response.text
            text = extract_human_readable_text(html)
            if max_chars > 0 and len(text) > max_chars:
                text = text[:max_chars].rsplit("\n", 1)[0].strip() or text[:max_chars]
            if not text.strip():
                return "", status, "No readable text extracted"
            return text, status, ""
        except requests.RequestException as exc:
            last_status = "fetch_error"
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt >= attempts:
                break

            delay = _next_backoff_seconds(
                attempt,
                response=response,
                total_slept=total_slept,
                max_total_backoff_seconds=max_backoff,
            )
            if delay > 0:
                LOGGER.debug(
                    "URL fetch attempt %d/%d failed for %s: %s; retrying in %.1fs",
                    attempt,
                    attempts,
                    url,
                    last_error,
                    delay,
                )
                time.sleep(delay)
                total_slept += delay

    return "", last_status, last_error


# --------------------------------------------------------------------------------------
# Chunking and text assembly
# --------------------------------------------------------------------------------------


def format_list(values: Iterable[str], *, none_text: str = "none listed") -> str:
    """Render optional list metadata into a short display string."""

    items = [str(v).strip() for v in values or [] if str(v).strip()]
    return ", ".join(items) if items else none_text


def build_metadata_header(recipe: RecipeRecord) -> str:
    """Create compact human-readable metadata prepended to each vector chunk."""

    parts = [
        f"Recipe: {recipe.recipe_name or '(unnamed recipe)'}",
        f"Source: {recipe.source or 'unknown'}",
        f"URL: {recipe.url or 'unknown'}",
        f"Servings: {recipe.servings if recipe.servings is not None else 'unknown'}",
        f"Calories: {recipe.calories:.2f}" if recipe.calories is not None else "Calories: unknown",
        f"Allergy cautions: {format_list(recipe.cautions)}",
        f"Diet labels: {format_list(recipe.diet_labels)}",
        f"Health labels: {format_list(recipe.health_labels)}",
        f"Cuisine type: {format_list(recipe.cuisine_type)}",
        f"Meal type: {format_list(recipe.meal_type)}",
        f"Dish type: {format_list(recipe.dish_type)}",
    ]
    return clean_text("\n".join(parts))


def build_full_recipe_text(recipe: RecipeRecord, page_text: str) -> str:
    """Build the full text fingerprinted for content change tracking."""

    header = build_metadata_header(recipe)
    page_text = clean_text(page_text)
    if not page_text:
        return header
    return clean_text(f"{header}\n\nRecipe page text:\n{page_text}")


def sliding_window_chunks(text: str, *, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence boundaries."""

    text = clean_text(text)
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        hard_end = min(start + chunk_size, text_len)
        end = hard_end

        if hard_end < text_len:
            boundary_start = start + max(1, int(chunk_size * 0.55))
            boundary_candidates = [
                text.rfind("\n\n", boundary_start, hard_end),
                text.rfind("\n", boundary_start, hard_end),
                text.rfind(". ", boundary_start, hard_end),
                text.rfind("; ", boundary_start, hard_end),
            ]
            boundary = max(boundary_candidates)
            if boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        next_start = max(end - chunk_overlap, start + 1)
        start = next_start

    return chunks


def build_recipe_vector_chunks(recipe: RecipeRecord, page_text: str, *, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Create vector chunks with recipe/allergy metadata repeated in each chunk.

    Repeating the header makes each chunk self-describing when retrieved on its
    own, at the cost of spending some embedding budget on metadata. Because even
    chunking text has trade-offs, naturally.
    """

    header = build_metadata_header(recipe)
    page_text = clean_text(page_text)

    if not page_text:
        return [header]

    body_prefix = "Recipe page text:"
    body_budget = chunk_size - len(header) - len(body_prefix) - 4
    # Keep body chunks meaningful even when metadata is long. The final chunk can
    # exceed chunk_size a bit because the allergy header is deliberately repeated.
    body_chunk_size = max(body_budget, min(chunk_size, 900))
    body_overlap = min(chunk_overlap, max(0, body_chunk_size - 1))

    body_chunks = sliding_window_chunks(page_text, chunk_size=body_chunk_size, chunk_overlap=body_overlap)
    if not body_chunks:
        return [header]

    return [clean_text(f"{header}\n\n{body_prefix}\n{body}") for body in body_chunks]


def build_chunk_payloads(recipe: RecipeRecord, chunk_texts: List[str]) -> List[Dict[str, Any]]:
    """Attach stable chunk IDs and ordinal metadata to chunk text."""

    payloads: List[Dict[str, Any]] = []
    chunk_count = len(chunk_texts)
    for idx, text in enumerate(chunk_texts):
        chunk_id = f"{recipe.recipe_id}::chunk-{idx:03d}"
        payloads.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "chunk_count": chunk_count,
                "text": text,
            }
        )
    return payloads


def doc_sha1(text: str) -> str:
    """Return a deterministic SHA-1 hex digest for document and chunk IDs."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# OpenSearch vector index management
# --------------------------------------------------------------------------------------


def ensure_vector_index(client: OpenSearch, index_name: str, dim: int) -> None:
    """Ensure the recipe vector index exists with metadata fields for filtering."""

    if client.indices.exists(index=index_name):
        LOGGER.info("OpenSearch index '%s' already exists", index_name)
        return

    # ``knn_vector`` stores the dense embedding. Keyword fields preserve recipe
    # metadata for inspection or downstream hard filters.
    body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 256,
            }
        },
        "mappings": {
            "properties": {
                "recipe_id": {"type": "keyword"},
                "recipe_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "source": {"type": "keyword"},
                "url": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "image_url": {"type": "keyword"},
                "servings": {"type": "integer"},
                "calories": {"type": "float"},
                "chunk_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_count": {"type": "integer"},
                "text": {"type": "text"},
                "cautions": {"type": "keyword"},
                "cautions_display": {"type": "keyword"},
                "diet_labels": {"type": "keyword"},
                "diet_labels_display": {"type": "keyword"},
                "health_labels": {"type": "keyword"},
                "health_labels_display": {"type": "keyword"},
                "cuisine_type": {"type": "keyword"},
                "cuisine_type_display": {"type": "keyword"},
                "meal_type": {"type": "keyword"},
                "meal_type_display": {"type": "keyword"},
                "dish_type": {"type": "keyword"},
                "dish_type_display": {"type": "keyword"},
                "content_sha1": {"type": "keyword"},
                "fetch_status": {"type": "keyword"},
                "ingested_at_ms": {"type": "long"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
            }
        },
    }
    LOGGER.info("Creating OpenSearch recipe vector index '%s'", index_name)
    client.indices.create(index=index_name, body=body)


def delete_existing_vector_docs(client: MyOpenSearch, index_name: str, recipe_id: str) -> None:
    """Remove old vector chunks for a recipe before re-indexing it."""

    body = {"query": {"term": {"recipe_id": recipe_id}}}
    try:
        client.delete_by_query(index=index_name, body=body, conflicts="proceed", refresh=False)
    except Exception as exc:
        LOGGER.warning("Could not delete existing vector chunks for recipe_id=%s: %s", recipe_id, exc)


def build_vector_document(
    recipe: RecipeRecord,
    *,
    chunk: Dict[str, Any],
    filename: str,
    embedding: List[float],
    content_sha1: str,
    fetch_status: str,
    now_ms: int,
) -> Dict[str, Any]:
    """Create the OpenSearch document stored for one embedded recipe chunk."""

    return {
        "recipe_id": recipe.recipe_id,
        "recipe_name": recipe.recipe_name,
        "source": recipe.source,
        "url": recipe.url,
        "filename": filename,
        "image_url": recipe.image_url,
        "servings": recipe.servings,
        "calories": recipe.calories,
        "chunk_id": chunk["chunk_id"],
        "chunk_index": chunk["chunk_index"],
        "chunk_count": chunk["chunk_count"],
        "text": chunk["text"],
        "cautions": recipe.normalized_cautions,
        "cautions_display": recipe.cautions,
        "diet_labels": recipe.normalized_diet_labels,
        "diet_labels_display": recipe.diet_labels,
        "health_labels": recipe.normalized_health_labels,
        "health_labels_display": recipe.health_labels,
        "cuisine_type": recipe.normalized_cuisine_type,
        "cuisine_type_display": recipe.cuisine_type,
        "meal_type": recipe.normalized_meal_type,
        "meal_type_display": recipe.meal_type,
        "dish_type": recipe.normalized_dish_type,
        "dish_type_display": recipe.dish_type,
        "content_sha1": content_sha1,
        "fetch_status": fetch_status,
        "ingested_at_ms": int(now_ms),
        "embedding": embedding,
    }


# --------------------------------------------------------------------------------------
# Ingestion orchestration
# --------------------------------------------------------------------------------------


@dataclass
class IngestStats:
    """Counters reported after the ingest run completes."""

    recipes: int = 0
    vector_chunks: int = 0
    fetched_ok: int = 0
    fetch_failed: int = 0
    recipes_skipped: int = 0


def ingest_vector(
    csv_path: Path,
    *,
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    fetch_timeout: Optional[float],
    fetch_max_attempts: int,
    fetch_max_backoff_seconds: float,
    max_page_chars: int,
) -> None:
    """Fetch recipe pages, build vector chunks, embed them, and index them."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if fetch_max_attempts <= 0:
        raise ValueError("fetch_max_attempts must be > 0")
    if fetch_max_backoff_seconds < 0:
        raise ValueError("fetch_max_backoff_seconds must be >= 0")

    vector_client, vector_index = create_vector_client()
    embedder = EmbeddingModel()
    ensure_vector_index(vector_client, vector_index, embedder.dimension)

    settings = vector_client.settings
    timeout = float(fetch_timeout if fetch_timeout is not None else settings.recipe_http_timeout_secs)
    user_agent = settings.recipe_http_user_agent
    now_ms = int(time.time() * 1000)
    stats = IngestStats()
    filename = csv_path.name

    recipes = list(iter_recipe_records(csv_path))
    progress = tqdm(recipes, desc="Recipe vector ingest", unit="recipe")

    try:
        for recipe in progress:
            page_text, fetch_status, fetch_error = fetch_url_human_text(
                recipe.url,
                timeout=timeout,
                user_agent=user_agent,
                max_chars=max_page_chars,
                max_attempts=fetch_max_attempts,
                max_total_backoff_seconds=fetch_max_backoff_seconds,
            )
            if page_text:
                stats.fetched_ok += 1
            else:
                stats.fetch_failed += 1
                stats.recipes_skipped += 1
                LOGGER.warning(
                    "Skipping recipe after URL text unavailable for row=%s recipe_id=%s url=%s status=%s error=%s",
                    recipe.row_number,
                    recipe.recipe_id,
                    recipe.url,
                    fetch_status,
                    fetch_error,
                )
                delete_existing_vector_docs(vector_client, vector_index, recipe.recipe_id)
                continue

            # The full content hash represents the recipe metadata plus extracted
            # page text; chunks are indexed separately but share this fingerprint.
            content = build_full_recipe_text(recipe, page_text)
            content_sha1 = doc_sha1(content)
            chunk_texts = build_recipe_vector_chunks(
                recipe,
                page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            chunks = build_chunk_payloads(recipe, chunk_texts)
            delete_existing_vector_docs(vector_client, vector_index, recipe.recipe_id)
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset : offset + batch_size]
                texts = [str(chunk["text"]) for chunk in batch]
                # Embeddings are generated in batches to avoid loading/scoring one
                # chunk at a time, a classic way to make demos feel haunted.
                embeddings = embedder.encode(texts)

                for chunk, embedding in zip(batch, embeddings):
                    vector_doc = build_vector_document(
                        recipe,
                        chunk=chunk,
                        filename=filename,
                        embedding=to_list(embedding),
                        content_sha1=content_sha1,
                        fetch_status=fetch_status,
                        now_ms=now_ms,
                    )
                    vector_client.index(
                        index=vector_index,
                        id=doc_sha1(str(chunk["chunk_id"])),
                        body=vector_doc,
                        refresh=False,
                    )
                    stats.vector_chunks += 1

            stats.recipes += 1
            progress.set_postfix_str(f"chunks={stats.vector_chunks}")
    finally:
        progress.close()

    vector_client.indices.refresh(index=vector_index)
    vector_client.close()

    LOGGER.info(
        "Recipe vector ingest complete: %d recipes, %d vector chunks, "
        "fetched_ok=%d, fetch_failed=%d, recipes_skipped=%d, vector_index='%s'",
        stats.recipes,
        stats.vector_chunks,
        stats.fetched_ok,
        stats.fetch_failed,
        stats.recipes_skipped,
        vector_index,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for recipe vector ingestion."""

    args = parse_args(argv)
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    ingest_vector(
        csv_path=csv_path,
        batch_size=int(args.batch_size),
        chunk_size=int(args.chunk_size),
        chunk_overlap=int(args.chunk_overlap),
        fetch_timeout=args.fetch_timeout,
        fetch_max_attempts=int(args.fetch_max_attempts),
        fetch_max_backoff_seconds=float(args.fetch_max_backoff_seconds),
        max_page_chars=int(args.max_page_chars),
    )


if __name__ == "__main__":
    main()
