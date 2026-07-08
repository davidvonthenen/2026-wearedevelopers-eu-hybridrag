#!/usr/bin/env python3
"""Export recipe CSV rows to human-readable flat text files.

This is the file-system version of the recipe ingest flow. It does not touch
OpenSearch, embeddings, or vector indexes. It reads the structured CSV,
fetches each recipe URL with bounded retry/backoff, extracts readable recipe text,
and writes one `.txt` file per recipe under `recipes/` by default.

Default behavior mirrors the stricter ingest path: if URL text cannot be fetched
or extracted, the recipe is skipped. That keeps the flat-file corpus from becoming
metadata-only confetti, because apparently even recipes need data hygiene now.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html.parser import HTMLParser
import json
import logging
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import requests

try:  # BeautifulSoup improves JSON-LD and visible text extraction, but is optional.
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover - depends on runtime environment
    BeautifulSoup = None  # type: ignore


LOGGER = logging.getLogger("recipe_file_export")

SPACE_RE = re.compile(r"[ \t\x0b\x0c\xa0]+")
BLANK_LINE_RE = re.compile(r"\n{3,}")
FILENAME_UNSAFE_RE = re.compile(r"[^a-zA-Z0-9._-]+")
SCRIPT_TAGS = {"script", "style", "noscript", "template", "svg", "canvas"}
NOISY_TAGS = {"nav", "header", "footer", "form", "aside", "iframe"}
RETRYABLE_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 410}
CSV_COLUMNS = [
    "recipe_name",
    "source",
    "url",
    "servings",
    "calories",
    "image_url",
    "diet_labels",
    "health_labels",
    "cautions",
    "cuisine_type",
    "meal_type",
    "dish_type",
]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export recipe CSV rows to human-readable text files under recipes/."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("recipes-with-nutrition-sample.csv"),
        help="Path to the recipe CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("recipes"),
        help="Folder where recipe .txt files will be written.",
    )
    parser.add_argument(
        "--clean-output-dir",
        action="store_true",
        default=False,
        help="Delete the output directory before writing files.",
    )
    parser.add_argument(
        "--allow-metadata-only",
        action="store_true",
        default=False,
        help="Write metadata-only files when URL text is unavailable instead of skipping the recipe.",
    )
    parser.add_argument(
        "--fetch-timeout",
        type=float,
        default=15.0,
        help="Per-attempt HTTP timeout in seconds.",
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
        help="Maximum extracted page characters to include per recipe.",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="recipe-flat-file-export/1.0 (+https://example.local; contact=dev-null)",
        help="User-Agent header used when fetching recipe URLs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of CSV rows to process. Zero means no limit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Recipe model and CSV loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeRecord:
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
    raw_row: Mapping[str, str]


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
        parsed = [x.strip() for x in text.split(",")]

    if isinstance(parsed, list):
        return [str(x).strip() for x in parsed if str(x).strip()]
    if isinstance(parsed, str) and parsed.strip():
        return [parsed.strip()]
    return []


def parse_int(value: Any) -> Optional[int]:
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


def iter_recipe_records(csv_path: Path, *, limit: int = 0) -> Iterator[RecipeRecord]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in CSV_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"CSV is missing expected columns: {', '.join(missing)}")

        for idx, row in enumerate(reader, start=1):
            if limit > 0 and idx > limit:
                break
            row_number = idx + 1  # account for CSV header row
            raw_row = {key: str(value or "") for key, value in row.items()}
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
                raw_row=raw_row,
            )


# --------------------------------------------------------------------------------------
# Text cleanup and webpage extraction
# --------------------------------------------------------------------------------------


class VisibleTextParser(HTMLParser):
    """Small fallback visible-text extractor when BeautifulSoup is unavailable."""

    def __init__(self) -> None:
        super().__init__()
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


def normalize_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


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
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        text = text.strip("\ufeff \t\n\r")
        text = re.sub(r"^<!--", "", text).strip()
        text = re.sub(r"-->$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


def _type_contains_recipe(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() == "recipe"
    if isinstance(value, list):
        return any(_type_contains_recipe(v) for v in value)
    return False


def _iter_recipe_jsonld(value: Any) -> Iterator[Mapping[str, Any]]:
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
    if BeautifulSoup is None:
        parser = VisibleTextParser()
        parser.feed(html)
        return parser.text()

    soup = BeautifulSoup(html, "html.parser")
    for tag_name in SCRIPT_TAGS | NOISY_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

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
    jsonld_text = extract_jsonld_recipe_text(html)
    visible_text = extract_visible_text(html)

    if jsonld_text and visible_text:
        return dedupe_lines(f"{jsonld_text}\n\nVisible page text:\n{visible_text}")
    return jsonld_text or visible_text


# --------------------------------------------------------------------------------------
# HTTP fetch with bounded retry/backoff
# --------------------------------------------------------------------------------------


def _retry_after_seconds(response: requests.Response, *, remaining_backoff: float) -> Optional[float]:
    value = response.headers.get("Retry-After", "").strip()
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
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
    remaining = max(0.0, float(max_total_backoff_seconds) - total_slept)
    if remaining <= 0:
        return 0.0

    if response is not None:
        retry_after = _retry_after_seconds(response, remaining_backoff=remaining)
        if retry_after is not None:
            return retry_after

    return min(float(2 ** max(0, attempt_index - 1)), remaining)


def _response_error(response: requests.Response) -> str:
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

    Returns ``(text, fetch_status, fetch_error)``. Callers should skip recipes
    when text is empty, unless metadata-only export was explicitly requested.
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
# File rendering
# --------------------------------------------------------------------------------------


def format_list(values: Iterable[str], *, none_text: str = "none listed") -> str:
    items = [str(v).strip() for v in values or [] if str(v).strip()]
    return ", ".join(items) if items else none_text


def slugify_filename(value: str, *, max_len: int = 90) -> str:
    text = value.strip().lower()
    text = FILENAME_UNSAFE_RE.sub("-", text)
    text = re.sub(r"-+", "-", text).strip("-._")
    if not text:
        text = "unnamed-recipe"
    return text[:max_len].strip("-._") or "unnamed-recipe"


def recipe_output_path(recipe: RecipeRecord, output_dir: Path) -> Path:
    slug = slugify_filename(recipe.recipe_name)
    filename = f"row-{recipe.row_number:04d}_{slug}_{recipe.recipe_id}.txt"
    return output_dir / filename


def render_recipe_file(
    recipe: RecipeRecord,
    *,
    page_text: str,
    fetch_status: str,
    fetch_error: str,
) -> str:
    title = recipe.recipe_name or "(unnamed recipe)"
    calories = f"{recipe.calories:.2f}" if recipe.calories is not None else "unknown"
    servings = str(recipe.servings) if recipe.servings is not None else "unknown"

    sections: List[str] = [
        title,
        "=" * len(title),
        "",
        "Recipe Metadata",
        "---------------",
        f"Recipe ID: {recipe.recipe_id}",
        f"CSV row number: {recipe.row_number}",
        f"Recipe name: {recipe.recipe_name or 'unknown'}",
        f"Source: {recipe.source or 'unknown'}",
        f"URL: {recipe.url or 'unknown'}",
        f"Image URL: {recipe.image_url or 'unknown'}",
        f"Servings: {servings}",
        f"Calories: {calories}",
        f"Diet labels: {format_list(recipe.diet_labels)}",
        f"Health labels: {format_list(recipe.health_labels)}",
        f"Allergy cautions: {format_list(recipe.cautions)}",
        f"Cuisine type: {format_list(recipe.cuisine_type)}",
        f"Meal type: {format_list(recipe.meal_type)}",
        f"Dish type: {format_list(recipe.dish_type)}",
        "",
        "Original CSV Fields",
        "-------------------",
    ]

    for column in CSV_COLUMNS:
        sections.append(f"{column}: {recipe.raw_row.get(column, '')}")

    sections.extend(
        [
            "",
            "URL Fetch",
            "---------",
            f"Fetch status: {fetch_status}",
            f"Fetch error: {fetch_error or 'none'}",
            "",
            "Recipe Text From URL",
            "--------------------",
            clean_text(page_text) if page_text.strip() else "URL text unavailable.",
            "",
        ]
    )
    return "\n".join(sections)


# --------------------------------------------------------------------------------------
# Export orchestration
# --------------------------------------------------------------------------------------


@dataclass
class ExportStats:
    rows_seen: int = 0
    files_written: int = 0
    fetched_ok: int = 0
    fetch_failed: int = 0
    skipped: int = 0


def prepare_output_dir(output_dir: Path, *, clean: bool) -> None:
    if clean and output_dir.exists():
        if output_dir.resolve() == Path.cwd().resolve():
            raise ValueError("Refusing to clean the current working directory.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def export_recipes_to_files(
    csv_path: Path,
    *,
    output_dir: Path,
    clean_output_dir: bool,
    allow_metadata_only: bool,
    fetch_timeout: float,
    fetch_max_attempts: int,
    fetch_max_backoff_seconds: float,
    max_page_chars: int,
    user_agent: str,
    limit: int = 0,
) -> ExportStats:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if fetch_timeout <= 0:
        raise ValueError("fetch_timeout must be > 0")
    if fetch_max_attempts <= 0:
        raise ValueError("fetch_max_attempts must be > 0")
    if fetch_max_backoff_seconds < 0:
        raise ValueError("fetch_max_backoff_seconds must be >= 0")

    prepare_output_dir(output_dir, clean=clean_output_dir)
    stats = ExportStats()

    for recipe in iter_recipe_records(csv_path, limit=limit):
        stats.rows_seen += 1
        page_text, fetch_status, fetch_error = fetch_url_human_text(
            recipe.url,
            timeout=fetch_timeout,
            user_agent=user_agent,
            max_chars=max_page_chars,
            max_attempts=fetch_max_attempts,
            max_total_backoff_seconds=fetch_max_backoff_seconds,
        )

        if page_text:
            stats.fetched_ok += 1
        else:
            stats.fetch_failed += 1
            if not allow_metadata_only:
                stats.skipped += 1
                LOGGER.warning(
                    "Skipping recipe row=%s recipe_id=%s url=%s status=%s error=%s",
                    recipe.row_number,
                    recipe.recipe_id,
                    recipe.url,
                    fetch_status,
                    fetch_error,
                )
                continue

        output_path = recipe_output_path(recipe, output_dir)
        output_path.write_text(
            render_recipe_file(
                recipe,
                page_text=page_text,
                fetch_status=fetch_status,
                fetch_error=fetch_error,
            ),
            encoding="utf-8",
        )
        stats.files_written += 1

        if stats.rows_seen % 25 == 0:
            LOGGER.info(
                "Processed %d rows: %d files written, %d skipped",
                stats.rows_seen,
                stats.files_written,
                stats.skipped,
            )

    return stats


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(bool(args.verbose))

    stats = export_recipes_to_files(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        clean_output_dir=bool(args.clean_output_dir),
        allow_metadata_only=bool(args.allow_metadata_only),
        fetch_timeout=float(args.fetch_timeout),
        fetch_max_attempts=int(args.fetch_max_attempts),
        fetch_max_backoff_seconds=float(args.fetch_max_backoff_seconds),
        max_page_chars=int(args.max_page_chars),
        user_agent=str(args.user_agent),
        limit=int(args.limit),
    )

    LOGGER.info(
        "Recipe file export complete: rows_seen=%d files_written=%d fetched_ok=%d fetch_failed=%d skipped=%d output_dir=%s",
        stats.rows_seen,
        stats.files_written,
        stats.fetched_ok,
        stats.fetch_failed,
        stats.skipped,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
