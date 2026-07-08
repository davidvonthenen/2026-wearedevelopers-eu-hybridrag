#!/usr/bin/env python3
"""Recipe Hybrid RAG query runner and REST agent.

This module supports two chatbot entry points: an interactive CLI and an
OpenAI-compatible REST service. Retrieval keeps graph and vector evidence
separate so deterministic recipe constraints stay anchored in Neo4j while
OpenSearch contributes semantic recipe detail.

* Neo4j graph retrieval is the truth channel for recipe identity, labels,
  cautions, nutrition, source, URL, and other structured metadata.
* OpenSearch vector retrieval is the semantic support channel for phrasing,
  ingredient intent, instructions, and fuzzy recipe language.
* Generation uses graph grounding first, then optional vector refinement, while
  preserving citation handles [G#] and [V#] for auditability.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
import re
import time
import uuid
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


from common.config import load_settings
from common.llm import (
    allowed_citation_handles,
    build_citation_repair_prompt,
    build_conversation_seed_prompt,
    build_grounding_prompt,
    build_refine_prompt,
    build_vector_only_prompt,
    call_llm_chat,
    contains_closing_citations,
    extract_citations,
    invalid_citations,
    load_llm,
    messages_contain_seed_evidence,
    strip_invalid_citations,
)
from common.models import RetrievalHit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_CITATION_PREFIX_ORDER = {"G": 0, "V": 1}
_HISTORY_CITATION_RE = re.compile(r"\[/?[GV]\d+\]")

DEFAULT_RECIPE_QUESTION = "Find a recipe that matches the request."


def _squash_ws(value: Any) -> str:
    """Collapse whitespace."""

    return " ".join(str(value or "").strip().split())


def _normalize_key(value: Any) -> str:
    """Tiny local normalizer used before graph modules are imported."""

    return _squash_ws(value).lower()


@dataclass
class PromptIntent:
    """Normalized query intent used by graph and vector retrieval."""

    prompt: str = ""
    graph_content_query: str = ""
    graph_content_notquery: str = ""
    vector_query: str = ""
    exclude_cautions: List[str] = None  # type: ignore[assignment]
    require_cautions: List[str] = None  # type: ignore[assignment]
    require_health_labels: List[str] = None  # type: ignore[assignment]
    require_diet_labels: List[str] = None  # type: ignore[assignment]
    cuisine_type: List[str] = None  # type: ignore[assignment]
    meal_type: List[str] = None  # type: ignore[assignment]
    dish_type: List[str] = None  # type: ignore[assignment]
    include_terms: List[str] = None  # type: ignore[assignment]
    exclude_terms: List[str] = None  # type: ignore[assignment]
    vector_query_omitted_sentences: List[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        for field_name in (
            "exclude_cautions",
            "require_cautions",
            "require_health_labels",
            "require_diet_labels",
            "cuisine_type",
            "meal_type",
            "dish_type",
            "include_terms",
            "exclude_terms",
            "vector_query_omitted_sentences",
        ):
            if getattr(self, field_name) is None:
                setattr(self, field_name, [])

    def has_graph_constraints(self) -> bool:
        return any(
            [
                bool(self.graph_content_query.strip()),
                bool(self.graph_content_notquery.strip()),
                bool(self.exclude_cautions),
                bool(self.require_cautions),
                bool(self.require_health_labels),
                bool(self.require_diet_labels),
                bool(self.cuisine_type),
                bool(self.meal_type),
                bool(self.dish_type),
            ]
        )

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI options shared by retrieval-only, CLI chat, and REST modes."""

    parser = argparse.ArgumentParser(
        description="Recipe Hybrid RAG query (Neo4j graph grounding + OpenSearch vector support)."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="User prompt to answer. This is deconstructed into graph filters and used as vector query context.",
    )
    parser.add_argument("--question", type=str, default=None, help="Alias for --prompt for BBC-style scripts.")
    parser.add_argument(
        "--service",
        action="store_true",
        default=False,
        help="Start the OpenAI-compatible Recipe RAG REST API instead of running a CLI query.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        default=False,
        help="Print graph/vector retrieval results without calling the LLM.",
    )

    # Legacy explicit filters are retained for scripted retrieval and workshop parity.
    parser.add_argument("--query", type=str, default="", help="Positive recipe-name filter and vector query override.")
    parser.add_argument("--notquery", type=str, default="", help="Negative recipe-name filter.")
    parser.add_argument("--exclude-caution", action="append", default=[], help="Exclude an allergy warning, repeatable.")
    parser.add_argument("--require-caution", action="append", default=[], help="Require an allergy warning, repeatable.")
    parser.add_argument("--require-health-label", action="append", default=[], help="Require a health label, repeatable.")
    parser.add_argument("--require-diet-label", action="append", default=[], help="Require a diet label, repeatable.")
    parser.add_argument("--cuisine-type", action="append", default=[], help="Filter by cuisine type, repeatable.")
    parser.add_argument("--meal-type", action="append", default=[], help="Filter by meal type, repeatable.")
    parser.add_argument("--dish-type", action="append", default=[], help="Filter by dish type, repeatable.")

    # Retrieval and generation knobs.
    parser.add_argument("--top-k", type=int, default=9, help="Total evidence budget.")
    parser.add_argument("--graph-k", type=int, default=4, help="Graph evidence budget. Defaults to about 60%% of top-k.")
    parser.add_argument("--vec-k", type=int, default=5, help="Vector evidence budget. Defaults to remaining top-k.")
    parser.add_argument("--candidate-k", type=int, default=5, help="Vector candidates before final top-k.")
    parser.add_argument("--max-graph-text-chars", type=int, default=2048, help="Max recipe content chars per [G] block.")
    parser.add_argument("--max-vector-text-chars", type=int, default=2048, help="Max vector chunk chars per [V] block.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-tokens", type=int, default=65536)
    
    parser.add_argument("--disable-citation-repair", action="store_true", default=False)
    parser.add_argument("--observability", action="store_true", default=False, help="Print retrieval queries, evidence, and citation audit data.")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose retrieval-only output.")

    parser.add_argument("--save-results", type=str, default=None, help="Append JSONL audit records to this path.")
    parser.add_argument("--service-host", type=str, default=None, help="Host to bind the Recipe RAG REST service.")
    parser.add_argument("--service-port", type=int, default=None, help="Port to bind the Recipe RAG REST service.")

    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# Prompt intent extraction
# --------------------------------------------------------------------------------------


def _dedupe_clean(values: Iterable[Any]) -> List[str]:
    """Normalize, strip punctuation, and preserve first-seen term order."""

    seen: set[str] = set()
    out: List[str] = []
    for raw in values or []:
        value = _squash_ws(raw)
        value = value.strip(" ,.;:()[]{}\"'")
        if not value:
            continue
        key = _normalize_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _join_terms(values: Iterable[str]) -> str:
    """Join cleaned terms into the space-delimited query format used downstream."""

    return " ".join(_dedupe_clean(values)).strip()


def _filter_content_terms(terms: Iterable[str], *, labels: Iterable[str]) -> List[str]:
    """Remove label-only phrases from content substring filters."""

    label_words: set[str] = set()
    for label in labels or []:
        label_key = _normalize_key(str(label).replace("-", " "))
        label_words.update(w for w in label_key.split() if w)

    filtered: List[str] = []
    for term in _dedupe_clean(terms):
        words = [w for w in _normalize_key(term).replace("-", " ").split() if w]
        if words and label_words and all(w in label_words for w in words):
            continue
        filtered.append(term)
    return filtered


def _split_prompt_parts(prompt: str) -> List[str]:
    """Split prompt into coarse sentence-like spans."""

    text = str(prompt or "").replace("\n", ".")
    for sep in (";", "?", "!"):
        text = text.replace(sep, ".")
    parts = [_squash_ws(part) for part in text.split(".")]
    parts = [part + "." for part in parts if part]
    return parts or ([_squash_ws(prompt) + "."] if _squash_ws(prompt) else [])


def _doc_sentence_texts(doc: Any, *, fallback: str) -> List[str]:
    """Return sentence texts from a spaCy doc with a safe fallback."""

    sentences: List[str] = []
    try:
        sentences = [_squash_ws(sent.text) for sent in doc.sents if _squash_ws(sent.text)]
    except Exception:
        sentences = []
    if sentences:
        return sentences
    fallback_text = _squash_ws(fallback)
    return [fallback_text] if fallback_text else []


def _sentence_key(text: str) -> str:
    """Normalize sentence text for matching omitted vector-query sentences."""

    return _normalize_key(text).rstrip(".")

def _is_allergy_context(text: str) -> bool:
    """Return true when a sentence is discussing allergy/caution constraints."""

    lower = str(text or "").lower()
    return any(marker in lower for marker in ("allergen", "allergens", "allergy", "allergies", "caution", "cautions"))


def _extract_intent_with_spacy(prompt: str, *, observability: bool = False) -> PromptIntent:
    """Extract recipe retrieval intent with spaCy entities and negspacy negation only."""

    import spacy
    from negspacy.negation import Negex  # noqa: F401 - registers the spaCy factory
    from spacy.language import Language

    # Load the general English pipeline for sentence boundaries and POS tags.
    nlp = spacy.load("en_core_web_sm")

    # Load the custom food NER model from the local `model/` directory.
    nlp_food = spacy.load("model")

    # Add the custom NER component after the base NER and give it a unique name
    # so both models can contribute entities to the same document.
    nlp.add_pipe("ner", name="food_ner", source=nlp_food, after="ner")

    if "food_false_positive_filter" not in Language.factories:
        @Language.component("food_false_positive_filter")
        def food_false_positive_filter(doc):  # type: ignore[no-redef]
            filtered_ents = []
            for ent in doc.ents:
                if ent.label_ == "FOOD":
                    valid_pos = all(token.pos_ in {"NOUN"} for token in ent)
                    if valid_pos:
                        filtered_ents.append(ent)
                else:
                    filtered_ents.append(ent)
            try:
                from spacy.util import filter_spans

                doc.ents = filter_spans(filtered_ents)
            except Exception:
                pass
            return doc

    # Add the false-positive filter before negation so negspacy only evaluates
    # food entities that survived the local noun-only filter.
    nlp.add_pipe("food_false_positive_filter", last=True)

    # Initialize negspacy after filtering. FOOD captures ingredient-like entities;
    # GPE is retained for location/cuisine phrasing discovered by the base model.
    nlp.add_pipe(
        "negex", 
        after="food_false_positive_filter", 
        config={"ent_types": ["FOOD", "GPE"]}
    )

    intent = PromptIntent(prompt=prompt)
    parts = _split_prompt_parts(prompt)
    kept_vector_sentences: List[str] = []
    omitted_vector_sentences: List[str] = []

    for part in parts:
        doc = nlp(part)
        negated_sentence_keys: set[str] = set()

        if observability:
            print(f"\n[NEGSPACY_INTENT_PART] {part!r}")
            for ent in doc.ents:
                print(f"  entity={ent.text!r} label={ent.label_!r} canonical={getattr(ent, 'ent_id_', '')!r} negated={getattr(ent._, 'negex', False)}")

        for ent in doc.ents:
            name = str(ent.text or "")
            label = str(ent.label_ or "")
            if label not in {"FOOD", "GPE"}:
                continue

            negated = bool(getattr(ent._, "negex", False))
            sentence_text = _squash_ws(ent.sent.text)
            allergy_context = _is_allergy_context(sentence_text)

            if negated and sentence_text:
                negated_sentence_keys.add(_sentence_key(sentence_text))
                omitted_vector_sentences.append(sentence_text)

            if allergy_context:
                if negated:
                    intent.exclude_cautions.append(name)
                else:
                    intent.require_cautions.append(name)
            elif negated:
                intent.exclude_terms.append(name)
            else:
                intent.include_terms.append(name)

        for sentence_text in _doc_sentence_texts(doc, fallback=part):
            if _sentence_key(sentence_text) in negated_sentence_keys:
                continue
            kept_vector_sentences.append(sentence_text)

    intent.vector_query_omitted_sentences = _dedupe_clean(omitted_vector_sentences)
    if intent.vector_query_omitted_sentences:
        # The vector kNN embedding must not be built from sentences containing
        # negated entities. OpenSearch treats the resulting vector as semantic
        # similarity, not as a Boolean exclusion filter.
        intent.vector_query = _squash_ws(". ".join(_dedupe_clean(kept_vector_sentences)))
    else:
        intent.vector_query = prompt

    return intent


def _extract_prompt_intent(prompt: str, *, observability: bool = False) -> PromptIntent:
    """Extract recipe retrieval intent from a natural-language prompt using negspacy only."""

    prompt = str(prompt or "").strip()
    intent = PromptIntent(prompt=prompt)
    if not prompt:
        return intent

    try:
        intent = _extract_intent_with_spacy(prompt, observability=observability)
    except Exception as exc:
        if observability:
            print(f"\n[NEGSPACY_INTENT_UNAVAILABLE] {exc.__class__.__name__}: {exc}")
        intent = PromptIntent(prompt=prompt)

    intent.exclude_cautions = _dedupe_clean(intent.exclude_cautions)
    intent.require_cautions = _dedupe_clean(intent.require_cautions)
    intent.include_terms = _dedupe_clean(intent.include_terms)
    intent.exclude_terms = _dedupe_clean(intent.exclude_terms)
    intent.require_health_labels = _dedupe_clean(intent.require_health_labels)
    intent.require_diet_labels = _dedupe_clean(intent.require_diet_labels)
    intent.cuisine_type = _dedupe_clean(intent.cuisine_type)
    intent.meal_type = _dedupe_clean(intent.meal_type)
    intent.dish_type = _dedupe_clean(intent.dish_type)
    intent.graph_content_query = _join_terms(intent.include_terms)
    intent.graph_content_notquery = _join_terms(intent.exclude_terms)
    if not intent.vector_query.strip() and not intent.vector_query_omitted_sentences:
        intent.vector_query = prompt
    return intent

def higher_level_query(prompt: str) -> tuple[list[str], list[str], str, str]:
    """Backward-compatible prompt deconstructor from the starter project.

    Returns ``(excluded_cautions, required_cautions, excluded_text, included_text)``.
    New code should use ``_build_effective_intent`` because it preserves labels,
    multiple terms, and the original vector query.
    """

    intent = _extract_prompt_intent(prompt, observability=True)
    return (
        intent.exclude_cautions,
        intent.require_cautions,
        intent.graph_content_notquery,
        intent.graph_content_query,
    )


def _build_effective_intent(args: argparse.Namespace, question: str) -> PromptIntent:
    """Merge NER-derived intent with explicit CLI or REST filter overrides."""

    extracted = _extract_prompt_intent(question, observability=bool(args.observability))
    explicit_query = str(args.query or "").strip()
    explicit_notquery = str(args.notquery or "").strip()

    include_terms = _dedupe_clean([explicit_query] if explicit_query else extracted.include_terms)
    exclude_terms = _dedupe_clean([explicit_notquery] if explicit_notquery else extracted.exclude_terms)
    require_health_labels = _dedupe_clean(list(args.require_health_label or []) + extracted.require_health_labels)
    require_diet_labels = _dedupe_clean(list(args.require_diet_label or []) + extracted.require_diet_labels)
    cuisine_type = _dedupe_clean(list(args.cuisine_type or []) + extracted.cuisine_type)
    meal_type = _dedupe_clean(list(args.meal_type or []) + extracted.meal_type)
    dish_type = _dedupe_clean(list(args.dish_type or []) + extracted.dish_type)
    label_terms = require_health_labels + require_diet_labels + cuisine_type + meal_type + dish_type
    content_terms = include_terms if str(args.query or "").strip() else _filter_content_terms(include_terms, labels=label_terms)
    graph_content_query = _join_terms(content_terms)
    graph_content_notquery = _join_terms(exclude_terms)
    
    vector_query = _squash_ws(explicit_query) if explicit_query else _squash_ws(extracted.vector_query)
    if not vector_query:
        # When negspacy found a negated entity, do not fall back to the raw
        # question because that would reinsert the excluded sentence into the
        # embedding query. Prefer positive graph terms, then a neutral default.
        vector_query = graph_content_query or _join_terms(include_terms)
    if not vector_query and not extracted.vector_query_omitted_sentences:
        vector_query = question

    intent = PromptIntent(
        prompt=vector_query,
        graph_content_query=graph_content_query,
        graph_content_notquery=graph_content_notquery,
        vector_query=vector_query,
        exclude_cautions=_dedupe_clean(list(args.exclude_caution or []) + extracted.exclude_cautions),
        require_cautions=_dedupe_clean(list(args.require_caution or []) + extracted.require_cautions),
        require_health_labels=require_health_labels,
        require_diet_labels=require_diet_labels,
        cuisine_type=cuisine_type,
        meal_type=meal_type,
        dish_type=dish_type,
        include_terms=include_terms,
        exclude_terms=exclude_terms,
        vector_query_omitted_sentences=extracted.vector_query_omitted_sentences,
    )
    if not intent.vector_query.strip():
        intent.vector_query = intent.graph_content_query or DEFAULT_RECIPE_QUESTION
    return intent


# --------------------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------------------


def _resolve_budgets(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Resolve the total evidence budget into graph and vector hit counts."""

    top_k = max(1, int(args.top_k))
    graph_k_arg = args.graph_k if args.graph_k is not None else args.gk
    vec_k_arg = args.vec_k if args.vec_k is not None else args.vk
    graph_k = int(graph_k_arg) if graph_k_arg is not None else max(1, int(round(top_k * 0.6)))
    vec_k = int(vec_k_arg) if vec_k_arg is not None else max(0, top_k - graph_k)
    if graph_k + vec_k != top_k and vec_k_arg is None:
        vec_k = max(0, top_k - graph_k)
    return top_k, max(0, graph_k), max(0, vec_k)


def make_graph_client() -> MyNeo4j:
    from common.neo4j_client import create_graph_client

    return create_graph_client()


def _close_graph_client(graph_client: MyNeo4j) -> None:
    try:
        graph_client.close()
    except Exception:
        pass


def _normalize_conversation_messages(
    messages: Optional[List[Dict[str, str]]],
    *,
    max_messages: int = 12,
    max_chars: int = 12000,
) -> List[Dict[str, str]]:
    """Return recent user/assistant messages suitable for LLM context."""

    if not messages:
        return []

    normalized: List[Dict[str, str]] = []
    total_chars = 0
    for message in reversed(messages):
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _HISTORY_CITATION_RE.sub("", _squash_ws(message.get("content") or ""))
        if not content:
            continue
        remaining = max_chars - total_chars
        if remaining <= 0:
            break
        if len(content) > remaining:
            content = content[-remaining:]
        normalized.append({"role": role, "content": content})
        total_chars += len(content)
        if len(normalized) >= max_messages:
            break

    return list(reversed(normalized))


def _with_conversation_messages(
    messages: List[Dict[str, str]],
    conversation_messages: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Insert prior user/assistant turns into an OpenAI messages payload."""

    history = _normalize_conversation_messages(conversation_messages)
    if not history:
        return messages
    if not messages:
        return history

    guard = {
        "role": "system",
        "content": (
            "Prior user/assistant messages may clarify the current request. "
            "They are not evidence for recipe facts. Use only the current evidence blocks "
            "and currently allowed citation tags for factual recipe claims."
        ),
    }
    return [messages[0], guard, *history, *messages[1:]]


def _build_contextual_retrieval_question(
    question: str,
    conversation_messages: Optional[List[Dict[str, str]]],
) -> str:
    """Blend recent conversation context into retrieval text for follow-up turns."""

    history = _normalize_conversation_messages(conversation_messages, max_messages=8, max_chars=5000)
    if not history:
        return question

    lines = ["Prior conversation context:"]
    for message in history:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{role}: {message['content']}")
    lines.append(f"Current user request: {_squash_ws(question)}")
    return "\n".join(lines).strip()


def _record_recipe_id(row: Dict[str, Any]) -> str:
    return str(row.get("recipe_id") or "").strip()



def _dedupe_records(records: Iterable[Dict[str, Any]], *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in records:
        rid = _record_recipe_id(row)
        if not rid:
            continue
        key = rid
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def _rank_graph_records(records: List[Dict[str, Any]], vector_recipe_ids: Iterable[str]) -> List[Dict[str, Any]]:
    """Rank graph-approved recipes by vector result order when available."""

    rank = {rid: i for i, rid in enumerate(_dedupe_clean(vector_recipe_ids))}

    def _key(row: Dict[str, Any]) -> Tuple[int, str]:
        rid = _record_recipe_id(row)
        return (rank.get(rid, 10**9), str(row.get("recipe_name") or ""))

    return sorted(records, key=_key)


def _fetch_graph_profile_records(
    graph_client: MyNeo4j,
    intent: PromptIntent,
    *,
    limit: int,
    observability: bool,
) -> List[Dict[str, Any]]:
    from common.graph import find_recipes_by_profile

    rows = find_recipes_by_profile(
        graph_client,
        content_query=intent.graph_content_query,
        content_notquery=intent.graph_content_notquery,
        required_cautions=intent.require_cautions,
        excluded_cautions=intent.exclude_cautions,
        required_health_labels=intent.require_health_labels,
        required_diet_labels=intent.require_diet_labels,
        cuisine_type=intent.cuisine_type,
        meal_type=intent.meal_type,
        dish_type=intent.dish_type,
        limit=max(limit, 1),
    )
    if observability:
        print(f"\n[GRAPH_PROFILE] rows={len(rows)}")
        print(json.dumps(rows[:limit], ensure_ascii=False, indent=2, default=str))
    return _dedupe_records(rows, limit=limit)


def _fetch_graph_records_by_ids(
    graph_client: MyNeo4j,
    recipe_ids: Iterable[str],
    *,
    limit: int,
    observability: bool,
) -> List[Dict[str, Any]]:
    from common.graph import find_recipes_by_ids

    ids = _dedupe_clean(recipe_ids)
    if not ids:
        return []
    rows = find_recipes_by_ids(graph_client, ids, limit=max(limit, 1))
    if observability:
        print(f"\n[GRAPH_BY_ID] ids={ids[:limit]} rows={len(rows)}")
        print(json.dumps(rows[:limit], ensure_ascii=False, indent=2, default=str))
    return _dedupe_records(rows, limit=limit)


def _truncate(text: Any, max_chars: int) -> str:
    value = str(text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _format_list(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(x) for x in value if str(x).strip()) or "none listed"
    text = str(value or "").strip()
    return text or "none listed"


def _format_graph_evidence(row: Dict[str, Any], *, max_chars: int) -> str:
    content = _truncate(row.get("content") or "", max_chars=max_chars)
    lines = [
        f"Recipe name: {row.get('recipe_name') or ''}",
        f"Source: {row.get('source') or ''}",
        f"URL: {row.get('url') or ''}",
        f"Servings: {row.get('servings') if row.get('servings') is not None else 'not listed'}",
        f"Calories: {row.get('calories') if row.get('calories') is not None else 'not listed'}",
        f"Allergy cautions: {_format_list(row.get('cautions'))}",
        f"Health labels: {_format_list(row.get('health_labels'))}",
        f"Diet labels: {_format_list(row.get('diet_labels'))}",
        f"Cuisine type: {_format_list(row.get('cuisine_type'))}",
        f"Meal type: {_format_list(row.get('meal_type'))}",
        f"Dish type: {_format_list(row.get('dish_type'))}",
    ]
    if content:
        lines.append(f"Recipe text excerpt: {content}")
    return "\n".join(lines).strip()


def _records_to_graph_hits(records: List[Dict[str, Any]], *, max_chars: int, k: int) -> List[RetrievalHit]:
    hits: List[RetrievalHit] = []
    for i, row in enumerate(records[:k], start=1):
        recipe_id = _record_recipe_id(row)
        hits.append(
            RetrievalHit(
                channel="graph_recipe",
                handle=f"G{i}",
                index="neo4j",
                os_id=recipe_id,
                score=max(0.0, 1.0 - ((i - 1) * 0.001)),
                path=str(row.get("url") or ""),
                category=str(row.get("source") or ""),
                text=_format_graph_evidence(row, max_chars=max_chars),
                recipe_id=recipe_id,
                recipe_name=str(row.get("recipe_name") or ""),
                source=str(row.get("source") or ""),
                url=str(row.get("url") or ""),
                servings=row.get("servings"),
                calories=row.get("calories"),
                meta={
                    "cautions": row.get("cautions") or [],
                    "health_labels": row.get("health_labels") or [],
                    "diet_labels": row.get("diet_labels") or [],
                    "cuisine_type": row.get("cuisine_type") or [],
                    "meal_type": row.get("meal_type") or [],
                    "dish_type": row.get("dish_type") or [],
                    "raw": {key: value for key, value in row.items() if key != "content"},
                },
            )
        )
    return hits


def _vector_dicts_to_hits(vector_hits: List[Dict[str, Any]], *, max_chars: int, k: int) -> List[RetrievalHit]:
    hits: List[RetrievalHit] = []
    for i, row in enumerate(vector_hits[:k], start=1):
        text = _truncate(row.get("text") or "", max_chars=max_chars)
        hits.append(
            RetrievalHit(
                channel="vector",
                handle=f"V{i}",
                index=str(row.get("index") or ""),
                os_id=str(row.get("id") or ""),
                score=float(row.get("score") or 0.0),
                path=str(row.get("url") or ""),
                category=str(row.get("source") or ""),
                chunk_index=row.get("chunk_index"),
                chunk_count=row.get("chunk_count"),
                text=text,
                recipe_id=str(row.get("recipe_id") or ""),
                recipe_name=str(row.get("recipe_name") or ""),
                source=str(row.get("source") or ""),
                url=str(row.get("url") or ""),
                chunk_id=str(row.get("chunk_id") or ""),
                meta={
                    "cautions": row.get("cautions") or [],
                    "health_labels": row.get("health_labels") or [],
                    "diet_labels": row.get("diet_labels") or [],
                    "cuisine_type": row.get("cuisine_type") or [],
                    "meal_type": row.get("meal_type") or [],
                    "dish_type": row.get("dish_type") or [],
                },
            )
        )
    return hits


def _run_vector_search(
    vec_client: MyOpenSearch,
    vector_index: str,
    intent: PromptIntent,
    *,
    include_recipe_ids: Optional[List[str]],
    k: int,
    candidate_k: int,
    observability: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Embed the vector query and run constrained OpenSearch kNN retrieval."""

    from common.embeddings import EmbeddingModel, to_list
    from common.opensearch_client import knn_search, normalize_vector_hits

    if k <= 0 or not intent.vector_query.strip():
        return [], {}, {}

    if observability and intent.vector_query_omitted_sentences:
        print("\n[VECTOR_QUERY_NEGATED_SENTENCES_OMITTED]")
        print(json.dumps(intent.vector_query_omitted_sentences, ensure_ascii=False, indent=2))
        print("\n[VECTOR_QUERY_TEXT_FOR_EMBEDDING]")
        print(intent.vector_query)

    embedder = EmbeddingModel()
    query_vector = to_list(embedder.encode([intent.vector_query])[0])
    response = knn_search(
        "VECTOR",
        vec_client,
        vector_index,
        query_vector,
        k=k,
        candidate_k=candidate_k,
        include_recipe_ids=include_recipe_ids,
        exclude_cautions=intent.exclude_cautions,
        require_cautions=intent.require_cautions,
        require_health_labels=intent.require_health_labels,
    )
    query = response.get("_query") or {}
    if observability:
        print("\n[VECTOR_QUERY]")
        print(json.dumps(query, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        if response.get("_error"):
            print("\n[VECTOR_ERROR]", response.get("_error"))
    return normalize_vector_hits(response), query, response


def retrieve_evidence(
    question: str,
    *,
    graph_client: MyNeo4j,
    vec_client: MyOpenSearch,
    args: argparse.Namespace,
) -> Tuple[PromptIntent, List[RetrievalHit], List[RetrievalHit], Dict[str, Any]]:
    """Retrieve graph and vector evidence for one user request.

    If graph constraints are active, Neo4j first selects eligible recipes and
    OpenSearch is constrained to those recipe IDs. If no graph constraints are
    active, vector retrieval finds candidate chunks first and Neo4j supplies the
    authoritative recipe records for those candidates.
    """

    settings = load_settings()
    vector_index = settings.opensearch_vector_index
    _top_k, graph_k, vec_k = _resolve_budgets(args)
    intent = _build_effective_intent(args, question)

    if args.observability:
        print("\n[RECIPE_INTENT]")
        print(json.dumps(intent.to_jsonable(), ensure_ascii=False, indent=2, sort_keys=True))
        print("\n[BUDGETS]")
        print(json.dumps({"graph_k": graph_k, "vec_k": vec_k, "top_k": _top_k}, indent=2))

    graph_constraints_active = intent.has_graph_constraints()
    graph_records: List[Dict[str, Any]] = []

    if graph_constraints_active:
        graph_records = _fetch_graph_profile_records(
            graph_client,
            intent,
            limit=max(graph_k, vec_k, 1) * 4,
            observability=bool(args.observability),
        )

    candidate_recipe_ids = [_record_recipe_id(row) for row in graph_records if _record_recipe_id(row)]
    if graph_constraints_active and not candidate_recipe_ids:
        audit = {
            "intent": intent.to_jsonable(),
            "vector_index": vector_index,
            "graph_constraints_active": True,
            "graph_candidate_recipe_ids": [],
            "vector_query": {},
            "vector_raw": {},
            "vector_dicts": [],
            "graph_records": [],
            "note": "Graph constraints produced no candidate recipes; vector search was intentionally skipped.",
        }
        return intent, [], [], audit

    vector_dicts, vector_query, vector_raw = _run_vector_search(
        vec_client,
        vector_index,
        intent,
        include_recipe_ids=candidate_recipe_ids if graph_constraints_active else None,
        k=max(vec_k, 0),
        candidate_k=args.candidate_k,
        observability=bool(args.observability),
    )

    vector_recipe_ids = [str(row.get("recipe_id") or "") for row in vector_dicts if str(row.get("recipe_id") or "").strip()]

    if not graph_constraints_active:
        graph_records = _fetch_graph_records_by_ids(
            graph_client,
            vector_recipe_ids,
            limit=max(graph_k, 1),
            observability=bool(args.observability),
        )
    else:
        # Use vector ordering to rank graph-approved recipes when semantic chunks are available.
        graph_records = _rank_graph_records(graph_records, vector_recipe_ids)

    graph_hits = _records_to_graph_hits(graph_records, max_chars=int(args.max_graph_text_chars), k=graph_k)
    vec_hits = _vector_dicts_to_hits(vector_dicts, max_chars=int(args.max_vector_text_chars), k=vec_k)

    if args.observability:
        print(f"\n[GRAPH_HITS] {len(graph_hits)}")
        for hit in graph_hits:
            print(f"  {hit.handle} recipe_id={hit.recipe_id} name={hit.recipe_name}")
        print(f"\n[VECTOR_HITS] {len(vec_hits)}")
        for hit in vec_hits:
            print(f"  {hit.handle} recipe_id={hit.recipe_id} score={hit.score:.4f} chunk={hit.chunk_index}")

    audit = {
        "intent": intent.to_jsonable(),
        "vector_index": vector_index,
        "graph_constraints_active": graph_constraints_active,
        "graph_candidate_recipe_ids": candidate_recipe_ids,
        "vector_query": vector_query,
        "vector_raw": vector_raw,
        "vector_dicts": vector_dicts,
        "graph_records": [{k: v for k, v in row.items() if k != "content"} for row in graph_records],
        "graph_hits": [hit.to_jsonable() for hit in graph_hits],
        "vector_hits": [hit.to_jsonable() for hit in vec_hits],
    }
    return intent, graph_hits, vec_hits, audit


# --------------------------------------------------------------------------------------
# Retrieval-only output, retained from starter behavior
# --------------------------------------------------------------------------------------


def build_output(
    *,
    query_text: str,
    notquery_text: str,
    graph_records: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "query": query_text,
        "notquery": notquery_text,
        "graph_content_filter": bool(query_text),
        "graph_content_exclusion_filter": bool(notquery_text),
        "graph_match_count": len(graph_records),
        "graph_matches": graph_records,
        "vector_hits": vector_hits,
    }


def print_output(payload: Dict[str, Any], verbose: bool) -> None:
    safe_payload = copy.deepcopy(payload)
    print("\n")
    print("-" * 20)
    print("Graph Matches:")
    print("-" * 20)
    if verbose:
        print(json.dumps(safe_payload["graph_matches"], ensure_ascii=False, indent=2, default=str))
    else:
        for match in safe_payload["graph_matches"]:
            match.pop("content", None)
        print(json.dumps(safe_payload["graph_matches"], ensure_ascii=False, indent=2, default=str))

    print("\n\n")
    print("-" * 20)
    print("Vector Hits:")
    print("-" * 20)
    if verbose:
        print(json.dumps(safe_payload["vector_hits"], ensure_ascii=False, indent=2, default=str))
    else:
        for match in safe_payload["vector_hits"]:
            match.pop("text", None)
        print(json.dumps(safe_payload["vector_hits"], ensure_ascii=False, indent=2, default=str))
    print("\n")


def lower_level_query(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args(None)
    question = args.prompt or args.question or args.query or DEFAULT_RECIPE_QUESTION
    from common.opensearch_client import create_vector_client

    graph_client = make_graph_client()
    vec_client, _vector_index = create_vector_client()
    try:
        intent, _graph_hits, _vec_hits, audit = retrieve_evidence(
            question,
            graph_client=graph_client,
            vec_client=vec_client,
            args=args,
        )
        print_output(
            build_output(
                query_text=intent.graph_content_query,
                notquery_text=intent.graph_content_notquery,
                graph_records=audit.get("graph_records") or [],
                vector_hits=audit.get("vector_dicts") or [],
            ),
            bool(args.verbose),
        )
    finally:
        _close_graph_client(graph_client)
        vec_client.close()


# --------------------------------------------------------------------------------------
# Generation and audit
# --------------------------------------------------------------------------------------


def _sort_citation_tokens(tokens: Iterable[str]) -> List[str]:
    def key(token: str) -> Tuple[int, int, str]:
        token = str(token or "")
        prefix = token[:1]
        suffix = token[1:]
        if prefix not in _CITATION_PREFIX_ORDER or not suffix.isdigit():
            return (2, 10**9, token)
        return (_CITATION_PREFIX_ORDER[prefix], int(suffix), token)

    return sorted(set(tokens), key=key)


def _maybe_repair_citations(
    *,
    question: str,
    answer: str,
    graph_hits: List[RetrievalHit],
    vec_hits: List[RetrievalHit],
    llm: Any,
    args: argparse.Namespace,
    model: str,
) -> Tuple[str, Dict[str, Any]]:
    """Repair or strip unsupported citation handles emitted by the LLM."""

    allowed = allowed_citation_handles(graph_hits, vec_hits)
    invalid_before = invalid_citations(answer, allowed)
    closing_before = contains_closing_citations(answer)
    repair_attempted = False
    repaired_answer = answer

    if (invalid_before or closing_before) and not bool(args.disable_citation_repair):
        repair_attempted = True
        repair_messages = build_citation_repair_prompt(
            question,
            answer,
            graph_hits,
            vec_hits,
            observability=bool(args.observability),
        )
        repaired_answer = call_llm_chat(
            llm,
            messages=repair_messages,
            model=model,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_tokens),
        )

    invalid_after = invalid_citations(repaired_answer, allowed)
    closing_after = contains_closing_citations(repaired_answer)
    stripped = False
    if invalid_after or closing_after:
        repaired_answer = strip_invalid_citations(repaired_answer, allowed)
        stripped = True

    audit = {
        "allowed_citations": _sort_citation_tokens(allowed),
        "citations_before_repair": extract_citations(answer),
        "invalid_before_repair": invalid_before,
        "closing_tags_before_repair": closing_before,
        "repair_attempted": repair_attempted,
        "citations_after_repair": extract_citations(repaired_answer),
        "invalid_after_repair": invalid_citations(repaired_answer, allowed),
        "closing_tags_after_repair": contains_closing_citations(repaired_answer),
        "stripped_invalid_citations": stripped,
    }
    return repaired_answer, audit


def run_one(
    question: str,
    *,
    graph_client: MyNeo4j,
    vec_client: MyOpenSearch,
    llm: Any,
    args: argparse.Namespace,
    conversation_messages: Optional[List[Dict[str, str]]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run retrieval, generation, optional refinement, and citation validation."""

    settings = load_settings()
    retrieval_question = _build_contextual_retrieval_question(question, conversation_messages)
    if bool(args.observability) and retrieval_question != question:
        print("\n[CONTEXTUAL_RETRIEVAL_QUESTION]")
        print(retrieval_question)

    start_retrieval = time.time()
    intent, graph_hits, vec_hits, retrieval_audit = retrieve_evidence(
        retrieval_question,
        graph_client=graph_client,
        vec_client=vec_client,
        args=args,
    )
    retrieval_s = time.time() - start_retrieval

    model = settings.llm_server_model

    if not graph_hits and retrieval_audit.get("graph_constraints_active"):
        answer = "I don't know based on the provided recipe evidence."
        generation_audit = {
            "model": model,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens": int(args.max_tokens),
            "grounded_draft": answer,
            "final_answer": answer,
            "citations_in_answer": [],
            "citation_audit": {
                "allowed_citations": [],
                "repair_attempted": False,
                "reason": "No graph-approved recipes matched the constraints.",
            },
        }
    else:
        if graph_hits:
            messages_a = build_grounding_prompt(question, graph_hits=graph_hits, observability=bool(args.observability))
        else:
            messages_a = build_vector_only_prompt(question, vec_hits=vec_hits, observability=bool(args.observability))
        messages_a = _with_conversation_messages(messages_a, conversation_messages)

        grounded_draft = call_llm_chat(
            llm,
            messages=messages_a,
            model=model,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_tokens),
        )

        if graph_hits and vec_hits:
            messages_b = build_refine_prompt(
                question,
                grounded_draft=grounded_draft,
                vec_hits=vec_hits,
                observability=bool(args.observability),
            )
            messages_b = _with_conversation_messages(messages_b, conversation_messages)
            answer = call_llm_chat(
                llm,
                messages=messages_b,
                model=model,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_tokens),
            )
        else:
            answer = grounded_draft

        answer, citation_audit = _maybe_repair_citations(
            question=question,
            answer=answer,
            graph_hits=graph_hits,
            vec_hits=vec_hits,
            llm=llm,
            args=args,
            model=model,
        )

        generation_audit = {
            "model": model,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens": int(args.max_tokens),
            "grounded_draft": grounded_draft,
            "final_answer": answer,
            "citations_in_answer": extract_citations(answer),
            "citation_audit": citation_audit,
        }

    audit = {
        "question": question,
        "retrieval_question": retrieval_question,
        "indices": {
            "neo4j_database": getattr(graph_client, "database", ""),
            "vector_chunks": retrieval_audit.get("vector_index"),
        },
        "retrieval": retrieval_audit,
        "generation": generation_audit,
        "timing": {"retrieval_s": retrieval_s},
    }
    return answer, audit


def append_jsonl(path: str, record: Dict[str, Any]) -> None:
    """Append one audit record as JSONL."""

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")



@dataclass
class CachedSessionEvidence:
    """Graph/vector evidence retrieved once for an interactive CLI session."""

    initial_question: str
    retrieval_question: str
    intent: PromptIntent
    graph_hits: List[RetrievalHit]
    vec_hits: List[RetrievalHit]
    retrieval_audit: Dict[str, Any]
    retrieval_s: float




def _citation_handles_from_messages(messages: Optional[List[Dict[str, str]]]) -> set[str]:
    """Extract citation handles already present in an evidence-seeded message history."""

    handles: set[str] = set()
    for message in messages or []:
        content = str(message.get("content") or "")
        if content:
            handles.update(extract_citations(content))
    return handles


def _validate_seeded_message_answer(answer: str, messages: List[Dict[str, str]]) -> str:
    """Validate direct LLM output when the message history already contains evidence."""

    allowed = _citation_handles_from_messages(messages)
    if not allowed:
        return answer
    if invalid_citations(answer, allowed) or contains_closing_citations(answer):
        return strip_invalid_citations(answer, allowed)
    return answer

def _validate_cached_session_citations(
    *,
    answer: str,
    graph_hits: List[RetrievalHit],
    vec_hits: List[RetrievalHit],
) -> Tuple[str, Dict[str, Any]]:
    """Validate follow-up citations against cached evidence without a second LLM call."""

    allowed = allowed_citation_handles(graph_hits, vec_hits)
    invalid_before = invalid_citations(answer, allowed)
    closing_before = contains_closing_citations(answer)
    final_answer = answer
    stripped = False
    if invalid_before or closing_before:
        final_answer = strip_invalid_citations(answer, allowed)
        stripped = True

    audit = {
        "allowed_citations": _sort_citation_tokens(allowed),
        "citations_before_validation": extract_citations(answer),
        "invalid_before_validation": invalid_before,
        "closing_tags_before_validation": closing_before,
        "repair_attempted": False,
        "repair_skipped_reason": "Cached CLI follow-up uses one LLM call; unsupported citations are stripped locally.",
        "citations_after_validation": extract_citations(final_answer),
        "invalid_after_validation": invalid_citations(final_answer, allowed),
        "closing_tags_after_validation": contains_closing_citations(final_answer),
        "stripped_invalid_citations": stripped,
    }
    return final_answer, audit



_QUIT_COMMANDS = {"quit", "exit", ":q", "q"}
_CLEAR_COMMANDS = {"clear", ":clear", "/clear"}


def _print_cli_answer(answer: str, audit: Dict[str, Any], *, elapsed: float) -> None:
    print("\n" + "=" * 100)
    print("ANSWER:")
    print(answer)
    print("\n" + "=" * 100)
    print(f"Query time: {elapsed:.2f}s")
    cites = audit.get("generation", {}).get("citations_in_answer", []) or []
    print("\nCitations used in answer:", ", ".join(cites) if cites else "(none)")


def _read_next_cli_question() -> Optional[str]:
    try:
        value = input("\nrecipe-rag> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not value:
        return ""
    if value.lower() in _QUIT_COMMANDS:
        return None
    return value


def run_cli_conversation(initial_question: str, *, args: argparse.Namespace) -> None:
    """Run the CLI as a cached-evidence chat until the user quits or clears."""

    from common.opensearch_client import create_vector_client

    settings = load_settings()
    graph_client = make_graph_client()
    vec_client, _ = create_vector_client()
    llm = None if bool(args.retrieval_only) else load_llm()
    model = settings.llm_server_model
    conversation_messages: List[Dict[str, str]] = []
    cache: Optional[CachedSessionEvidence] = None
    question = initial_question
    turn = 1
    session_epoch = 1

    print("Interactive Recipe RAG started. Type 'clear' to reset cached recipes; type 'quit' to exit.")

    def _retrieve_for_session(seed_question: str) -> CachedSessionEvidence:
        start_retrieval = time.time()
        retrieval_question = seed_question
        intent, graph_hits, vec_hits, retrieval_audit = retrieve_evidence(
            retrieval_question,
            graph_client=graph_client,
            vec_client=vec_client,
            args=args,
        )
        return CachedSessionEvidence(
            initial_question=seed_question,
            retrieval_question=retrieval_question,
            intent=intent,
            graph_hits=graph_hits,
            vec_hits=vec_hits,
            retrieval_audit=retrieval_audit,
            retrieval_s=time.time() - start_retrieval,
        )

    def _print_retrieval_cache(active_cache: CachedSessionEvidence, *, reused: bool) -> None:
        if reused:
            print("\nCached retrieval context reused; graph/vector retrieval was not run. Astonishing restraint from the machinery.")
        print_output(
            build_output(
                query_text=active_cache.intent.graph_content_query,
                notquery_text=active_cache.intent.graph_content_notquery,
                graph_records=active_cache.retrieval_audit.get("graph_records") or [],
                vector_hits=active_cache.retrieval_audit.get("vector_dicts") or [],
            ),
            bool(args.verbose),
        )

    def _build_turn_audit(
        *,
        active_cache: CachedSessionEvidence,
        current_question: str,
        generation_audit: Optional[Dict[str, Any]],
        elapsed: float,
        cache_initialized_this_turn: bool,
    ) -> Dict[str, Any]:
        retrieval_payload: Dict[str, Any]
        if cache_initialized_this_turn:
            retrieval_payload = active_cache.retrieval_audit
        else:
            retrieval_payload = {
                "cache_reused": True,
                "refresh_skipped_until_clear": True,
                "note": "No graph recipe or vector chunk retrieval was run on this turn.",
                "initial_question": active_cache.initial_question,
                "initial_retrieval_question": active_cache.retrieval_question,
                "graph_hit_count": len(active_cache.graph_hits),
                "vector_hit_count": len(active_cache.vec_hits),
            }
        return {
            "question": current_question,
            "retrieval_question": active_cache.retrieval_question if cache_initialized_this_turn else None,
            "conversation_turn": turn,
            "conversation_session": session_epoch,
            "conversation_history_turns": len(conversation_messages),
            "message_history_contains_seed_evidence": messages_contain_seed_evidence(conversation_messages),
            "evidence_cache": {
                "initialized_this_turn": cache_initialized_this_turn,
                "reused_this_turn": not cache_initialized_this_turn,
                "refresh_skipped_until_clear": not cache_initialized_this_turn,
                "clear_command": "clear",
                "initial_question": active_cache.initial_question,
                "graph_hit_count": len(active_cache.graph_hits),
                "vector_hit_count": len(active_cache.vec_hits),
            },
            "indices": {
                "neo4j_database": getattr(graph_client, "database", ""),
                "vector_chunks": active_cache.retrieval_audit.get("vector_index"),
            },
            "retrieval": retrieval_payload,
            "generation": generation_audit or {},
            "timing": {
                "retrieval_s": active_cache.retrieval_s if cache_initialized_this_turn else 0.0,
                "cached_initial_retrieval_s": active_cache.retrieval_s,
            },
            "timing_s": elapsed,
        }

    def _run_llm_turn(active_cache: CachedSessionEvidence, current_question: str, *, cache_initialized_this_turn: bool) -> Tuple[str, Dict[str, Any]]:
        if cache_initialized_this_turn:
            conversation_messages.clear()
            conversation_messages.extend(
                build_conversation_seed_prompt(
                    current_question,
                    graph_hits=active_cache.graph_hits,
                    vec_hits=active_cache.vec_hits,
                    observability=bool(args.observability),
                )
            )
        else:
            conversation_messages.append({"role": "user", "content": current_question})

        raw_answer = call_llm_chat(
            llm,
            messages=conversation_messages,
            model=model,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            max_tokens=int(args.max_tokens),
        )
        answer, citation_audit = _validate_cached_session_citations(
            answer=raw_answer,
            graph_hits=active_cache.graph_hits,
            vec_hits=active_cache.vec_hits,
        )
        conversation_messages.append({"role": "assistant", "content": answer})
        generation_audit = {
            "model": model,
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_tokens": int(args.max_tokens),
            "raw_answer": raw_answer,
            "final_answer": answer,
            "citations_in_answer": extract_citations(answer),
            "citation_audit": citation_audit,
            "llm_call_count": 1,
            "evidence_added_to_message_history_this_turn": cache_initialized_this_turn,
            "evidence_repeated_in_new_message_this_turn": False,
            "message_history_contains_seed_evidence": messages_contain_seed_evidence(conversation_messages),
        }
        return answer, generation_audit

    try:
        while True:
            print("\n" + "=" * 100)
            print(f"TURN {turn} QUESTION: {question}")
            print("=" * 100)

            start = time.time()
            cache_initialized_this_turn = cache is None
            if cache is None:
                cache = _retrieve_for_session(question)
                # if args.observability:
                #     print("\n[EVIDENCE_CACHE] initialized from one graph/vector retrieval")
            # elif args.observability:
            #     print("\n[EVIDENCE_CACHE] reused; graph/vector retrieval skipped until 'clear'")

            if args.retrieval_only:
                _print_retrieval_cache(cache, reused=not cache_initialized_this_turn)
                elapsed = time.time() - start
                audit = _build_turn_audit(
                    active_cache=cache,
                    current_question=question,
                    generation_audit=None,
                    elapsed=elapsed,
                    cache_initialized_this_turn=cache_initialized_this_turn,
                )
            else:
                answer, generation_audit = _run_llm_turn(
                    cache,
                    question,
                    cache_initialized_this_turn=cache_initialized_this_turn,
                )
                elapsed = time.time() - start
                audit = _build_turn_audit(
                    active_cache=cache,
                    current_question=question,
                    generation_audit=generation_audit,
                    elapsed=elapsed,
                    cache_initialized_this_turn=cache_initialized_this_turn,
                )
                _print_cli_answer(answer, audit, elapsed=elapsed)

            if args.save_results:
                audit["created_at_ms"] = int(time.time() * 1000)
                append_jsonl(args.save_results, audit)
                print(f"\nSaved JSONL record to: {args.save_results}")

            while True:
                next_question = _read_next_cli_question()
                if next_question is None:
                    return
                if not next_question:
                    continue
                if next_question.lower() in _CLEAR_COMMANDS:
                    conversation_messages.clear()
                    cache = None
                    session_epoch += 1
                    print("\nConversation context cleared. Enter a new recipe request to retrieve a fresh recipe set.")
                    continue
                question = next_question
                turn += 1
                break
    finally:
        _close_graph_client(graph_client)
        vec_client.close()


# --------------------------------------------------------------------------------------
# REST agent service
# --------------------------------------------------------------------------------------


@dataclass
class ServiceResources:
    """Long-lived clients reused by the REST service between requests."""

    graph_client: MyNeo4j
    vec_client: MyOpenSearch
    llm: Any

    def close(self) -> None:
        _close_graph_client(self.graph_client)
        try:
            self.vec_client.close()
        except Exception:
            pass


def _init_service_resources(args: argparse.Namespace) -> ServiceResources:
    from common.opensearch_client import create_vector_client

    graph_client = make_graph_client()
    vec_client, _ = create_vector_client()
    llm = load_llm()
    return ServiceResources(graph_client=graph_client, vec_client=vec_client, llm=llm)


def _normalize_messages(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected non-empty 'messages' list.")
    normalized: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("Each message must be a JSON object.")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Each message requires string 'role' and 'content'.")
        normalized.append({"role": role, "content": content})
    return normalized


def _extract_question_from_messages(messages: List[Dict[str, str]]) -> str:
    if not messages:
        raise ValueError("No messages provided.")
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[-1].get("content", "")


def _messages_before_current_user(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return messages[:idx]
    return messages[:-1]



def _build_chat_response(*, model: str, content: str) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _error(status: int, message: str) -> tuple[Dict[str, Any], int]:
    return {"error": {"message": message, "type": "invalid_request_error"}}, status


def _coerce_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, *, default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_string_list(value: Any, *, default: List[str]) -> List[str]:
    if value is None:
        return list(default)
    if isinstance(value, list):
        return _dedupe_clean(value)
    if isinstance(value, str):
        return _dedupe_clean([value])
    return list(default)


def _build_request_args(base_args: argparse.Namespace, payload: Dict[str, Any]) -> argparse.Namespace:
    """Merge REST payload options into an argparse-style namespace."""

    merged = vars(base_args).copy()
    merged["temperature"] = _coerce_float(payload.get("temperature"), default=float(base_args.temperature))
    merged["top_p"] = _coerce_float(payload.get("top_p"), default=float(base_args.top_p))
    merged["max_tokens"] = int(_coerce_int(payload.get("max_tokens"), default=int(base_args.max_tokens)) or base_args.max_tokens)
    merged["top_k"] = int(_coerce_int(payload.get("top_k"), default=int(base_args.top_k)) or base_args.top_k)
    merged["graph_k"] = _coerce_int(payload.get("graph_k"), default=base_args.graph_k)
    merged["vec_k"] = _coerce_int(payload.get("vec_k"), default=base_args.vec_k)
    candidate_k = _coerce_int(payload.get("candidate_k"), default=base_args.candidate_k)
    merged["candidate_k"] = candidate_k if candidate_k is not None and candidate_k > 0 else None
    merged["max_graph_text_chars"] = int(
        _coerce_int(payload.get("max_graph_text_chars"), default=int(base_args.max_graph_text_chars)) or base_args.max_graph_text_chars
    )
    merged["max_vector_text_chars"] = int(
        _coerce_int(payload.get("max_vector_text_chars"), default=int(base_args.max_vector_text_chars)) or base_args.max_vector_text_chars
    )

    # Optional structured filter overrides for clients that know the taxonomy.
    merged["exclude_caution"] = _coerce_string_list(payload.get("exclude_caution"), default=list(base_args.exclude_caution or []))
    merged["require_caution"] = _coerce_string_list(payload.get("require_caution"), default=list(base_args.require_caution or []))
    merged["require_health_label"] = _coerce_string_list(
        payload.get("require_health_label"), default=list(base_args.require_health_label or [])
    )
    merged["require_diet_label"] = _coerce_string_list(payload.get("require_diet_label"), default=list(base_args.require_diet_label or []))
    merged["cuisine_type"] = _coerce_string_list(payload.get("cuisine_type"), default=list(base_args.cuisine_type or []))
    merged["meal_type"] = _coerce_string_list(payload.get("meal_type"), default=list(base_args.meal_type or []))
    merged["dish_type"] = _coerce_string_list(payload.get("dish_type"), default=list(base_args.dish_type or []))
    merged["query"] = str(payload.get("query") or base_args.query or "")
    merged["notquery"] = str(payload.get("notquery") or base_args.notquery or "")
    merged["prompt"] = None
    merged["question"] = None
    merged["retrieval_only"] = False
    return argparse.Namespace(**merged)


def _resolve_service_bindings(args: argparse.Namespace) -> Tuple[str, int]:
    host = args.service_host or os.getenv("RAG_AGENT_HOST", "0.0.0.0")
    port = args.service_port or int(os.getenv("RAG_AGENT_PORT", "8002"))
    return host, int(port)


def create_service_app(args: argparse.Namespace) -> Any:
    """Create the Flask app exposing OpenAI-compatible Recipe RAG endpoints."""

    try:
        from flask import Flask, jsonify, request
    except Exception as exc:
        raise RuntimeError("Flask is required to run the Recipe RAG REST service. Install `flask`.") from exc

    app = Flask(__name__)
    settings = load_settings()
    resources = _init_service_resources(args)
    app.config["RECIPE_RAG_RESOURCES"] = resources
    atexit.register(resources.close)

    @app.route("/health", methods=["GET"])
    def health() -> tuple[Dict[str, Any], int]:
        host, port = _resolve_service_bindings(args)
        return jsonify(
            {
                "status": "ok",
                "model": settings.llm_server_model,
                "server": {"host": host, "port": port},
                "vector_index": settings.opensearch_vector_index,
            }
        ), 200

    @app.route("/v1/models", methods=["GET"])
    def models() -> tuple[Dict[str, Any], int]:
        return jsonify(
            {
                "object": "list",
                "data": [
                    {
                        "id": settings.llm_server_model,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "local",
                    }
                ],
            }
        ), 200

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions() -> tuple[Dict[str, Any], int]:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return _error(400, "Expected JSON object payload.")
        if payload.get("stream") is True:
            return _error(400, "Streaming responses are not supported.")

        try:
            messages = _normalize_messages(payload)
            question = _extract_question_from_messages(messages)
        except ValueError as exc:
            return _error(400, str(exc))

        request_args = _build_request_args(args, payload)
        model = str(payload.get("model") or settings.llm_server_model)
        if messages_contain_seed_evidence(messages):
            answer = call_llm_chat(
                resources.llm,
                messages=messages,
                model=model,
                temperature=float(request_args.temperature),
                top_p=float(request_args.top_p),
                max_tokens=int(request_args.max_tokens),
            )
            answer = _validate_seeded_message_answer(answer, messages)
        else:
            answer, _audit = run_one(
                question,
                graph_client=resources.graph_client,
                vec_client=resources.vec_client,
                llm=resources.llm,
                args=request_args,
                conversation_messages=_messages_before_current_user(messages),
            )
        return jsonify(_build_chat_response(model=model, content=answer)), 200

    return app


def run_service(args: argparse.Namespace) -> None:
    """Start the Recipe RAG REST service."""

    app = create_service_app(args)
    host, port = _resolve_service_bindings(args)
    app.run(host=host, port=port, debug=False)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for service mode or interactive chatbot mode."""

    args = parse_args(argv)

    if args.service:
        run_service(args)
        return

    question = args.prompt or args.question
    if not question and args.query:
        question = f"Find recipe matches for: {args.query}"
        if args.notquery:
            question += f" while excluding: {args.notquery}"
    if not question:
        print("No prompt provided. Use --prompt or --question for the RAG path, or --query for retrieval-only matching.")
        return

    run_cli_conversation(question, args=args)


if __name__ == "__main__":
    main()
