#!/usr/bin/env python3
"""Two-pass recipe RAG inference demonstration using BM25 and vector evidence.

This file intentionally does not query OpenSearch. Retrieval is represented by
hardcoded BM25 and vector hits so prompt construction and answer refinement can
be evaluated in isolation from retrieval behavior.

Stage 1 uses BM25 evidence as the authoritative truth-grounding pass.
Stage 2 uses vector evidence for semantic detail and human-readable phrasing
while preserving BM25-grounded facts.

Environment variables:
    EXTERNAL_LLM_URL         OpenAI-compatible base URL, default: http://127.0.0.1:8001/v1
    EXTERNAL_LLM_API_KEY     API key for the endpoint, default: not-needed
    EXTERNAL_LLM_MODEL       Chat model name, default: local-llm
    LLM_TEMPERATURE          Sampling temperature, default: 0.2
    LLM_TOP_P                Nucleus sampling value, default: 0.9
    EXTERNAL_LLM_MAX_TOKENS  Max output tokens per call, default: 65536
    USE_EXTERNAL_AI          Set to 1/true/yes to use EXTERNAL_LLM_URL instead of local internal endpoint
    OBSERVABILITY            Set to 1/true/yes to print prompt payloads

Run:
    python inference.py
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Set

from openai import OpenAI


# -----------------------------------------------------------------------------
# Retrieval model passed to prompt builders.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalHit:
    """Normalized evidence item passed to the prompt builders."""

    channel: str = ""  # bm25 | vector
    handle: str = ""  # B1..Bn or V1..Vn
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


# -----------------------------------------------------------------------------
# OpenAI-compatible client helpers used by the standalone inference demo.
# -----------------------------------------------------------------------------

def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def load_llm() -> Any:
    """Construct an OpenAI-compatible client for the configured LLM endpoint."""
    base_url = os.getenv("EXTERNAL_LLM_URL", "http://127.0.0.1:8001/v1").strip()
    api_key = os.getenv("EXTERNAL_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "not-needed"

    # Default to the local workshop LLM service. USE_EXTERNAL_AI switches the same
    # OpenAI-compatible call path to the configured external provider.
    use_external = _env_bool("USE_EXTERNAL_AI", False)
    if not use_external:
        print("Using internal AI provider")
        client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="not-needed")
        return client

    print("Using external AI provider")
    client = OpenAI(base_url=base_url, api_key=api_key)
    return client


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Fallback formatting for prompt observability."""

    parts: List[str] = []
    for message in messages:
        role = (message.get("role") or "user").upper()
        content = message.get("content") or ""
        parts.append(f"{role}:\n{content}".strip())
    return "\n\n".join(parts).strip()


def _extract_llm_text(resp: Any) -> str:
    """Normalize common completion response shapes into a plain string."""

    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()

    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                msg = c0.get("message")
                if isinstance(msg, dict) and "content" in msg:
                    return str(msg.get("content") or "").strip()
                if "text" in c0:
                    return str(c0.get("text") or "").strip()
                delta = c0.get("delta")
                if isinstance(delta, dict) and "content" in delta:
                    return str(delta.get("content") or "").strip()
        if isinstance(resp.get("content"), str):
            return str(resp["content"]).strip()
        return str(resp).strip()

    choices = getattr(resp, "choices", None)
    if choices and isinstance(choices, list):
        c0 = choices[0]
        msg = getattr(c0, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str):
                return content.strip()
        text = getattr(c0, "text", None)
        if isinstance(text, str):
            return text.strip()

    return str(resp).strip()


def call_llm_chat(
    llm: Any,
    *,
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """Run a chat completion against an OpenAI-compatible client."""

    resp = llm.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return _extract_llm_text(resp)


def _format_hits(hits: List[RetrievalHit], *, title: str) -> str:
    """Render retrieval hits into tag-delimited evidence blocks.

    Each block includes metadata and the actual chunk text. Including the text is
    important because the inference prompt needs recipe content, not just hit
    metadata, to ground or refine the generated answer.
    """

    if not hits:
        return f"=== {title} ===\n(none)"

    blocks: List[str] = [f"=== {title} ==="]
    for hit in hits:
        open_tag = f"[{hit.handle}]"
        close_tag = f"[/{hit.handle}]"
        meta: List[str] = []

        if hit.recipe_id:
            meta.append(f"recipe_id={hit.recipe_id}")
        if hit.recipe_name:
            meta.append(f"recipe_name={hit.recipe_name}")
        if hit.source:
            meta.append(f"source={hit.source}")
        if hit.url or hit.path:
            meta.append(f"url={hit.url or hit.path}")
        if hit.path:
            meta.append(f"document={hit.path}")
        if hit.chunk_index is not None and hit.chunk_count is not None:
            meta.append(f"chunk={hit.chunk_index}/{hit.chunk_count}")
        if hit.score:
            meta.append(f"score={hit.score:.4f}")
        if hit.explicit_terms:
            terms = [str(t).strip() for t in hit.explicit_terms if str(t).strip()]
            if terms:
                preview = ", ".join(terms[:10])
                if len(terms) > 10:
                    preview += ", ..."
                meta.append(f"matched_terms=[{preview}]")

        meta_line = f"META: {', '.join(meta)}" if meta else ""
        text = (hit.text or "").strip()
        body_parts = [part for part in (meta_line, text) if part]
        body = "\n".join(body_parts)
        blocks.append(f"{open_tag}\n{body}\n{close_tag}".strip())

    return "\n\n".join(blocks).strip()


def _distinct_recipe_count(hits: List[RetrievalHit]) -> int:
    """Count distinct recipe options represented by retrieval hits."""

    seen: Set[str] = set()
    fallback = 0
    for hit in hits or []:
        key = (hit.recipe_id or hit.recipe_name or hit.url or hit.path or "").strip()
        if key:
            seen.add(key)
        else:
            fallback += 1
    return len(seen) + fallback


def _allowed_citation_tags(hits: List[RetrievalHit]) -> List[str]:
    """Return opening citation tags for evidence handles."""

    out: List[str] = []
    for hit in hits:
        handle = (hit.handle or "").strip()
        if handle:
            out.append(f"[{handle}]")
    return out


# -----------------------------------------------------------------------------
# Prompt builders.
# -----------------------------------------------------------------------------


def build_grounding_prompt(
    question: str,
    bm25_hits: List[RetrievalHit],
    observability: Optional[bool] = False,
) -> List[Dict[str, str]]:
    """Build the BM25-only grounding prompt for recipe answers.

    The first pass treats BM25 chunks as authoritative evidence for recipe facts,
    constraints, and citations.
    """

    context = _format_hits(
        bm25_hits,
        title="Recipe BM25 Evidence (authoritative recipe facts; each [B#] block is a candidate recipe chunk)",
    )
    allowed = " ".join(_allowed_citation_tags(bm25_hits)) if bm25_hits else "(none)"
    option_count = _distinct_recipe_count(bm25_hits)
    system = (
        "You are a recipe retrieval assistant. Answer using ONLY the Recipe BM25 Evidence.\n"
        "Recipe BM25 evidence is authoritative for recipe identity, ingredients, labels, cautions, nutrition, source, and URL.\n"
        "Evidence chunks are delimited as [B#] ... [/B#]. Each [B#] block represents one grounded candidate recipe chunk.\n"
        f"Candidate recipe option count: {option_count}.\n"
        "\n"
        "MULTI-OPTION RULES (mandatory):\n"
        "- If the candidate recipe option count is greater than 1, present every grounded option as a separate recipe option.\n"
        "- Do not collapse multiple grounded recipes into a single recommendation unless the user explicitly asks for exactly one recipe.\n"
        "- For each option, include the recipe name and the supported reason it matches the request.\n"
        "- Keep comparisons concise; do not rank options unless the evidence directly supports the ranking.\n"
        "\n"
        "CITATION RULES (mandatory):\n"
        f"- Allowed citation tags: {allowed}\n"
        "- After EVERY sentence containing a factual recipe claim, append one or more allowed [B#] tags.\n"
        "- Cite each recipe option with its own [B#] tag; do not cite one option with another option's tag.\n"
        "- Use opening tags only, for example [B1]. Never use closing tags like [/B1].\n"
        "- Do not mention BM25, vector, database, retrieval, or evidence mechanics in the answer.\n"
        "\n"
        "RECIPE SAFETY RULES:\n"
        "- Allergy/caution, diet, health-label, calorie, serving, ingredient, and source claims must come from [B#] evidence.\n"
        "- If the evidence does not prove that a recipe satisfies a user constraint, do not say that it does.\n"
        "- If no provided recipe satisfies the request, say that the provided recipe data is insufficient.\n"
        "\n"
        "If the evidence does not support an answer, write exactly: I don't know based on the provided recipe evidence.\n"
        "Output ONLY the answer text."
    )
    user = f"USER REQUEST:\n{question}\n\nGROUNDING_EVIDENCE:\n{context}\n"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        print("\n[GROUNDING_PROMPT]")
        print(_messages_to_prompt(messages))

    return messages


def build_refine_prompt(
    question: str,
    grounded_draft: str,
    vec_hits: List[RetrievalHit],
    observability: Optional[bool] = False,
) -> List[Dict[str, str]]:
    """Build the refinement prompt that may use vector context for phrasing.

    The second pass may improve wording with vector chunks, but it is instructed
    to preserve BM25-grounded facts and citations from the draft.
    """

    vec_context = _format_hits(
        vec_hits,
        title="Recipe Vector Context (phrasing and semantic support; not authoritative for recipe facts)",
    )
    allowed_v = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    option_count = _distinct_recipe_count(vec_hits)
    system = (
        "Rewrite the grounded draft for clarity, concision, and helpful recipe phrasing.\n"
        "The grounded draft may contain multiple recipe options. Preserve that multi-option structure.\n"
        f"Distinct vector recipe option count available for semantic support: {option_count}.\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Preserve every existing [B#] citation exactly. Do not delete, renumber, merge, or move BM25 citations away from their claims.\n"
        "- Preserve every BM25-grounded recipe option from the draft; do not collapse multiple options into one.\n"
        "- Recipe facts, allergy/caution claims, diet labels, health labels, nutrition, source, and URL remain grounded only by [B#].\n"
        "- You may use vector context only for wording, terminology alignment, and brief non-factual clarifications.\n"
        "\n"
        "VECTOR CITATIONS:\n"
        f"- Allowed vector citation tags: {allowed_v}\n"
        "- Add [V#] only when a sentence uses semantic context from the vector chunks.\n"
        "- If vector context improves wording for a specific option, cite the [V#] tag tied to that same recipe when available.\n"
        "- Use opening tags only. Never use closing tags like [/V1].\n"
        "- Do not mention BM25, vector, database, retrieval, or evidence mechanics.\n"
        "\n"
        "Output ONLY the revised answer text."
    )
    user = (
        f"USER REQUEST:\n{question}\n\n"
        f"GROUNDED_DRAFT:\n{grounded_draft}\n\n"
        f"{vec_context}\n"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        print("\n[REFINE_PROMPT]")
        print(_messages_to_prompt(messages))

    return messages


# -----------------------------------------------------------------------------
# Hardcoded retrieval inputs used to isolate inference/prompt behavior.
# -----------------------------------------------------------------------------


HARDCODED_USER_PROMPT = (
    "We are looking for a noodle recipe that is sulfites free and does not contain any soba. "
    "What are the ingredients needed for this recipe?"
)

HARDCODED_BM25_HITS: List[RetrievalHit] = [
    RetrievalHit(
        channel="bm25",
        handle="B1",
        index="recipes-bm25",
        os_id="9f052d8b134e302d",
        score=1.0,
        recipe_id="9f052d8b134e302d",
        recipe_name="Chicken noodle stir fry",
        source="legenrecipes.com",
        url="https://legenrecipes.com/recipe/chicken-noodle-stir-fry",
        servings=10,
        calories=2485.9525,
        text="""
    Recipe: Chicken noodle stir fry
    Source: legenrecipes.com
    URL: https://legenrecipes.com/recipe/chicken-noodle-stir-fry/
    Servings: 10
    Calories: 2485.95
    Allergy cautions: none listed
    Diet labels: Low-Fat, Low-Sodium
    Health labels: Dairy-Free, Peanut-Free, Tree-Nut-Free, Soy-Free, Fish-Free, Shellfish-Free
    Cuisine type: asian
    Meal type: lunch/dinner
    Dish type: main course
    """,
    ),
]

HARDCODED_VECTOR_HITS: List[RetrievalHit] = [
    RetrievalHit(
        channel="vector",
        handle="V1",
        index="recipes-vector",
        os_id="2356ddc18b6a782a408d06bd634e97443f787391",
        score=0.6905856,
        recipe_id="9f052d8b134e302d",
        recipe_name="Chicken noodle stir fry",
        source="legenrecipes.com",
        url="https://legenrecipes.com/recipe/chicken-noodle-stir-fry",
        servings=10,
        calories=2485.9525,
        chunk_index=1,
        chunk_count=2,
        text=("""
    Recipe: Chicken noodle stir fry

    Ingredients:
    - 400 grams hokkien noodles
    - 1 teaspoon sesame oil
    - 2 chicken breasts, thickly sliced
    - 1 zucchini, chopped into large match sticks
    - 1 bunch broccolini, ends trimmed
    - 2 large handfuls of snow peas, ends trimmed
    - 1 garlic clove, thinly sliced

    Instructions:
    1. Prepare noodles according to packet instructions, set aside.
    2. Heat non stick fry pan to medium-high heat, add half a tablespoon of sesame oil and chicken and cook for 4-5 minutes or until chicken is golden and cooked through. Remove and set aside.
    3. Add half a tablespoon of sesame oil to the fry pan, add the zucchini, broccolini and snow peas and cook for 2-3 minutes or until the vegetables are golden. Add garlic and cook for a further minute. Add the chicken and noodles and stir to combine.
    """),
    ),
    RetrievalHit(
        channel="vector",
        handle="V2",
        index="recipes-vector",
        os_id="2356ddc18b6a782a408d06bd634e97443f787391",
        score=0.5905856,
        recipe_id="9f052d8b134e302d",
        recipe_name="Chicken noodle stir fry",
        source="legenrecipes.com",
        url="https://legenrecipes.com/recipe/chicken-noodle-stir-fry",
        servings=10,
        calories=2485.9525,
        chunk_index=2,
        chunk_count=2,
        text=("""
    Visible page text:
    Chicken noodle stir fry
    Preparing vegetables that are easy for bub's to handle and eat can be quite the baby led weaning challenge .
    Over time and though much trial and error I have discovered a few lovely little tips and tricks that has helped Grace munch away on her fair share of wonderful vegetables.
    To start I find the shape of the sliced/chopped vegetable is pretty important; especially at the beginning of the baby led weaning journey when grabbing and holding food is very new to our little ones.
    Most vegetables I serve to Grace I chop into large match stick shapes .
    5 Stars 4 Stars 3 Stars 2 Stars 1 Star
    No reviews
    Ingredients
    400 grams hokkien noodles
    1 teaspoon sesame oil
    2 chicken breasts, thickly sliced
    1 zucchini, chopped into large match sticks
    1 bunch broccolini, ends trimmed
    """),
    ),
]


# -----------------------------------------------------------------------------
# Main two-pass inference demonstration.
# -----------------------------------------------------------------------------


def main() -> None:
    """Run the two-stage inference demonstration against the configured LLM."""

    model = os.getenv("EXTERNAL_LLM_MODEL", "local-llm").strip() or "local-llm"
    temperature = _env_float("LLM_TEMPERATURE", 0.2)
    top_p = _env_float("LLM_TOP_P", 0.9)
    max_tokens = _env_int("EXTERNAL_LLM_MAX_TOKENS", 65536)
    observability = _env_bool("OBSERVABILITY", False)

    llm = load_llm()

    print("\n\nPrompt:")
    print(HARDCODED_USER_PROMPT)
    print("\n")

    # Step 1: user request + BM25 hits -> grounded draft with BM25 citations.
    grounding_messages = build_grounding_prompt(
        HARDCODED_USER_PROMPT,
        bm25_hits=HARDCODED_BM25_HITS,
        observability=observability,
    )
    truth_grounding_result = call_llm_chat(
        llm,
        messages=grounding_messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # Step 2: grounded draft + vector hits -> clearer final answer without changing facts.
    refine_messages = build_refine_prompt(
        HARDCODED_USER_PROMPT,
        grounded_draft=truth_grounding_result,
        vec_hits=HARDCODED_VECTOR_HITS,
        observability=observability,
    )
    final_answer = call_llm_chat(
        llm,
        messages=refine_messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    print(final_answer)


if __name__ == "__main__":
    main()
