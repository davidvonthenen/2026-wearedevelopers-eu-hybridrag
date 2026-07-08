"""LLM client, prompts, and citation-aware generation helpers for recipes."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .config import Settings, load_settings
from .logging import get_logger
from .models import RetrievalHit

LOGGER = get_logger(__name__)

_CLOSING_CITATION_RE = re.compile(r"\[/([GV]\d+)\]")
_BRACKET_GROUP_RE = re.compile(r"\[([^\]]+)\]")
_PAREN_GROUP_RE = re.compile(r"\(([^\)]+)\)")
_CITATION_TOKEN_RE = re.compile(r"\b([GV]\d+)\b")


def load_llm(settings: Optional[Settings] = None) -> Any:
    """Construct an OpenAI-compatible client for the configured LLM endpoint."""

    if settings is None:
        settings = load_settings()

    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "The OpenAI Python package is required for RAG generation. Install `openai` "
            "or run query.py with --retrieval-only."
        ) from exc

    LOGGER.info("Connecting to LLM server at %s", settings.llm_server_url)
    client = OpenAI(
        base_url=settings.llm_server_url,
        api_key=settings.llm_server_api_key,
    )
    setattr(client, "default_model", settings.llm_server_model)
    return client


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Fallback formatting for logging prompt payloads."""

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
    """Render retrieval hits into tag-delimited evidence blocks."""

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
        if meta_line and text:
            blocks.append(f"{open_tag}\n{meta_line}\n{text}\n{close_tag}".strip())
        elif meta_line:
            blocks.append(f"{open_tag}\n{meta_line}\n{close_tag}".strip())
        elif text:
            blocks.append(f"{open_tag}\n{text}\n{close_tag}".strip())
        else:
            blocks.append(f"{open_tag}\n{close_tag}".strip())

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


def allowed_citation_handles(graph_hits: List[RetrievalHit], vec_hits: Optional[List[RetrievalHit]] = None) -> Set[str]:
    """Return allowed citation handles without brackets."""

    handles = {hit.handle for hit in graph_hits if hit.handle}
    if vec_hits:
        handles.update(hit.handle for hit in vec_hits if hit.handle)
    return handles


def extract_citations(answer: str) -> List[str]:
    """Extract citation tokens such as ``G1`` and ``V2`` from model output."""

    if not answer:
        return []

    cites: set[str] = set()
    for group in _BRACKET_GROUP_RE.findall(answer):
        for token in _CITATION_TOKEN_RE.findall(group):
            cites.add(token)
    for group in _PAREN_GROUP_RE.findall(answer):
        if "G" not in group and "V" not in group:
            continue
        for token in _CITATION_TOKEN_RE.findall(group):
            cites.add(token)

    def _key(tag: str) -> tuple[int, int, str]:
        prefix = tag[:1]
        try:
            number = int(tag[1:])
        except Exception:
            number = 10**9
        return (0 if prefix == "G" else 1, number, tag)

    return sorted(cites, key=_key)


def invalid_citations(answer: str, allowed_handles: Set[str]) -> List[str]:
    """Return citation handles not present in the retrieved evidence."""

    return [tag for tag in extract_citations(answer) if tag not in allowed_handles]


def contains_closing_citations(answer: str) -> bool:
    """Return true when a model leaked closing evidence tags into the answer."""

    return bool(_CLOSING_CITATION_RE.search(answer or ""))


def strip_invalid_citations(answer: str, allowed_handles: Set[str]) -> str:
    """Remove unsupported citation tokens while keeping supported citations intact."""

    allowed = set(allowed_handles)

    def _replace_group(match: re.Match[str]) -> str:
        raw = match.group(1)
        tokens = _CITATION_TOKEN_RE.findall(raw)
        if not tokens:
            return match.group(0)
        kept = [tok for tok in tokens if tok in allowed]
        if not kept:
            return ""
        return "[" + ", ".join(kept) + "]"

    text = _CLOSING_CITATION_RE.sub("", answer or "")
    text = _BRACKET_GROUP_RE.sub(_replace_group, text)
    return re.sub(r"\s+", " ", text).strip()


def build_grounding_prompt(
    question: str,
    graph_hits: List[RetrievalHit],
    observability: Optional[bool] = False,
) -> List[Dict[str, str]]:
    """Build the graph-only grounding prompt for recipe answers."""

    if observability:
        LOGGER.info("Building recipe grounding prompt with %d graph hits", len(graph_hits))

    context = _format_hits(graph_hits, title="Recipe Graph Evidence (authoritative recipe facts; each [G#] block is a candidate recipe option)")
    allowed = " ".join(_allowed_citation_tags(graph_hits)) if graph_hits else "(none)"
    option_count = _distinct_recipe_count(graph_hits)

    system = (
        "You are a recipe retrieval assistant. Answer using ONLY the Recipe Graph Evidence.\n"
        "Recipe graph evidence is authoritative for recipe identity, ingredients, labels, cautions, nutrition, source, and URL.\n"
        "Evidence chunks are delimited as [G#] ... [/G#]. Each [G#] block represents one grounded candidate recipe option.\n"
        f"Candidate recipe option count: {option_count}.\n"
        "\n"
        "MULTI-OPTION RULES (mandatory):\n"
        "- If the candidate recipe option count is greater than 1, present every grounded option as a separate recipe option.\n"
        "- Do not collapse multiple grounded recipes into a single recommendation.\n"
        "- For each option, include the recipe name and the supported reason it matches the request.\n"
        "- Keep comparisons concise.\n"
        "\n"
        "CITATION RULES (mandatory):\n"
        f"- Allowed citation tags: {allowed}\n"
        "- After EVERY sentence containing a factual recipe claim, append one or more allowed [G#] tags.\n"
        "- Cite each recipe option with its own [G#] tag; do not cite one option with another option's tag.\n"
        "- Use opening tags only, for example [G1]. Never use closing tags like [/G1].\n"
        "- Do not mention graph, vector, database, retrieval, or evidence mechanics in the answer.\n"
        "\n"
        "If the evidence does not support an answer, write exactly: I don't know based on the provided recipe evidence.\n"
        "Output ONLY the answer text."
    )
    user = f"USER REQUEST:\n{question}\n\nGROUNDING_EVIDENCE:\n{context}\n"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        LOGGER.info("=====================================================")
        LOGGER.info(_messages_to_prompt(messages))
        LOGGER.info("=====================================================")

    return messages

def build_vector_only_prompt(
    question: str,
    vec_hits: List[RetrievalHit],
    observability: Optional[bool] = False,
) -> List[Dict[str, str]]:
    """Build a constrained vector-only fallback prompt."""

    if observability:
        LOGGER.info("Building recipe vector-only prompt with %d vector hits", len(vec_hits))

    context = _format_hits(vec_hits, title="Recipe Vector Evidence (semantic fallback; group chunks by recipe option when possible)")
    allowed = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    option_count = _distinct_recipe_count(vec_hits)

    system = (
        "Answer using ONLY the Recipe Vector Evidence.\n"
        "Evidence chunks are delimited as [V#] ... [/V#]. Vector evidence is a fallback when graph-grounded recipe evidence is unavailable.\n"
        f"Distinct recipe option count represented by the vector evidence: {option_count}.\n"
        "\n"
        "MULTI-OPTION RULES (mandatory):\n"
        "- If the evidence contains more than one distinct recipe option, present all supported options separately.\n"
        "- Group chunks from the same recipe together when they share the same recipe name, recipe ID, source, or URL.\n"
        "- Do not collapse multiple recipes into one recommendation.\n"
        "\n"
        "CITATION RULES (mandatory):\n"
        f"- Allowed citation tags: {allowed}\n"
        "- After EVERY factual sentence, append one or more allowed [V#] tags.\n"
        "- Cite each recipe option with the [V#] tag or tags that describe that option.\n"
        "- Use opening tags only, for example [V1]. Never use closing tags like [/V1].\n"
        "\n"
        "If evidence does not support the answer, write exactly: I don't know based on the provided recipe evidence.\n"
        "Output ONLY the answer text."
    )
    user = f"USER REQUEST:\n{question}\n\nEVIDENCE:\n{context}\n"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        LOGGER.info("=====================================================")
        LOGGER.info(_messages_to_prompt(messages))
        LOGGER.info("=====================================================")

    return messages

def build_refine_prompt(
    question: str,
    grounded_draft: str,
    vec_hits: List[RetrievalHit],
    observability: Optional[bool] = False,
) -> List[Dict[str, str]]:
    """Build the refinement prompt that may use vector context for phrasing."""

    if observability:
        LOGGER.info("Building recipe refine prompt with %d vector hits", len(vec_hits))

    vec_context = _format_hits(vec_hits, title="Recipe Vector Context (phrasing and semantic support; not authoritative for recipe facts)")
    allowed_v = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    option_count = _distinct_recipe_count(vec_hits)

    system = (
        "Rewrite the grounded draft for clarity, concision, and helpful recipe phrasing.\n"
        "The grounded draft may contain multiple recipe options. Preserve that multi-option structure.\n"
        f"Distinct vector recipe option count available for semantic support: {option_count}.\n"
        "\n"
        "CRITICAL RULES:\n"
        "- Preserve every existing [G#] citation exactly. Do not delete, renumber, merge, or move graph citations away from their claims.\n"
        "- Preserve every graph-grounded recipe option from the draft; do not collapse multiple options into one.\n"
        "- Recipe facts, allergy/caution claims, source, and URL remain grounded only by [G#].\n"
        "- You may use vector context only for wording, terminology alignment, and brief non-factual clarifications.\n"
        "\n"
        "VECTOR CITATIONS:\n"
        f"- Allowed vector citation tags: {allowed_v}\n"
        "- After EVERY factual sentence, append one or more allowed [V#] tags.\n"
        "- If vector context improves wording for a specific option, cite the [V#] tag tied to that same recipe when available.\n"
        "- Use opening tags only. Never use closing tags like [/V1].\n"
        "- Do not mention graph, vector, database, retrieval, or evidence mechanics.\n"
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
        LOGGER.info("=====================================================")
        LOGGER.info(_messages_to_prompt(messages))
        LOGGER.info("=====================================================")

    return messages

def build_conversation_seed_prompt(
    question: str,
    graph_hits: List[RetrievalHit],
    vec_hits: List[RetrievalHit],
    observability: Optional[bool] = False,
) -> List[Dict[str, str]]:
    """Build the one-time evidence-seeded prompt for interactive chat sessions.

    The returned message list is meant to stay in the OpenAI message history.
    Later CLI turns append plain user/assistant messages and do not repeat the
    graph recipes or vector embedding chunks. Terribly advanced concept: memory.
    """

    if observability:
        LOGGER.info(
            "Building one-time recipe conversation seed prompt with %d graph hits and %d vector hits",
            len(graph_hits),
            len(vec_hits),
        )

    graph_context = _format_hits(
        graph_hits,
        title="Recipe Graph Evidence (authoritative recipe facts; each [G#] block is a candidate recipe option)",
    )
    vector_context = _format_hits(
        vec_hits,
        title="Recipe Vector Embedding Context (semantic chunks returned by vector similarity; additive support only)",
    )
    allowed_g = " ".join(_allowed_citation_tags(graph_hits)) if graph_hits else "(none)"
    allowed_v = " ".join(_allowed_citation_tags(vec_hits)) if vec_hits else "(none)"
    graph_option_count = _distinct_recipe_count(graph_hits)
    vector_option_count = _distinct_recipe_count(vec_hits)

    if graph_hits:
        authority_rule = (
            "Recipe graph evidence is authoritative for recipe identity, ingredients, cautions, and URL.\n"
            "Recipe vector embedding context is additive semantic support only: it may help with wording, recipe-text nuance, and chunk-level context, but it must never override graph facts.\n"
        )
        factual_citation_rule = "- After EVERY sentence containing a factual recipe claim, append one or more allowed [G#] tags.\n"
        option_rule = "- If the graph candidate recipe option count is greater than 1, present every grounded graph recipe option that answers the user's request.\n"
        safety_rule = "- Allergy/caution, diet, and health-label must come from [G#] evidence.\n"
    else:
        authority_rule = (
            "No graph-grounded recipe evidence is available for this session. Recipe vector embedding context is a semantic fallback only.\n"
            "Avoid allergy-safe, diet-compliant, or health-label unless the vector text explicitly supports them.\n"
        )
        factual_citation_rule = "- After EVERY sentence containing a factual recipe claim, append one or more allowed [V#] tags.\n"
        option_rule = "- If the vector embedding context represents more than one recipe option, present every supported option separately.\n"
        safety_rule = "- Treat vector-only answers as cautious summaries, not authoritative allergy, diet, or nutrition guarantees.\n"

    system = (
        "You are a recipe retrieval assistant running inside an evidence-seeded conversation.\n"
        "The initial user message contains the complete recipe evidence for this session. Use that same evidence for follow-up questions.\n"
        "Do not ask for or assume refreshed retrieval data. Do not invent recipes, ingredients, labels, cautions, nutrition, sources, or URLs.\n"
        f"{authority_rule}"
        f"Graph candidate recipe option count: {graph_option_count}.\n"
        f"Distinct vector recipe option count represented by embedding context: {vector_option_count}.\n"
        "\n"
        "MULTI-OPTION RULES (mandatory):\n"
        f"{option_rule}"
        "- Do not collapse multiple graph-grounded recipes into a single recommendation unless the user explicitly asks for exactly one recipe.\n"
        "- For each option, include the recipe name and the supported reason it matches the request.\n"
        "- Keep comparisons concise; do not rank options unless the evidence directly supports the ranking.\n"
        "\n"
        "CITATION RULES (mandatory):\n"
        f"- Allowed graph citation tags: {allowed_g}\n"
        f"- Allowed vector citation tags: {allowed_v}\n"
        f"{factual_citation_rule}"
        "- Use [V#] only when a sentence uses semantic context from vector embedding chunks.\n"
        "- Cite each recipe option with its own evidence tag; do not cite one option with another option's tag.\n"
        "- Use opening tags only, for example [G1] or [V2]. Never use closing tags like [/G1].\n"
        "- Never invent citation numbers.\n"
        "- Do not mention graph, vector, database, retrieval, embedding, chunk, or evidence mechanics in the answer.\n"
        "\n"
        "SAFETY RULES:\n"
        f"{safety_rule}"
        "- If vector context conflicts with graph evidence, ignore the vector context.\n"
        "- If the evidence does not prove that a recipe satisfies a user constraint, do not say that it does.\n"
        "- If the current question cannot be answered from the evidence already in this conversation, write exactly: I don't know based on the provided recipe evidence.\n"
        "\n"
        "Output ONLY the answer text."
    )
    user = (
        f"INITIAL USER REQUEST:\n{question}\n\n"
        "The following recipe evidence is the complete evidence context for this conversation. "
        "It should remain available through the message history for later follow-up questions.\n\n"
        f"RECIPE_GRAPH_EVIDENCE:\n{graph_context}\n\n"
        f"RECIPE_VECTOR_EMBEDDING_CONTEXT:\n{vector_context}\n"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        LOGGER.info("=====================================================")
        LOGGER.info(_messages_to_prompt(messages))
        LOGGER.info("=====================================================")

    return messages


def messages_contain_seed_evidence(messages: Optional[List[Dict[str, str]]]) -> bool:
    """Return true when a chat history already carries the one-time evidence seed."""

    for message in messages or []:
        content = str(message.get("content") or "")
        if "RECIPE_GRAPH_EVIDENCE:" in content or "RECIPE_VECTOR_EMBEDDING_CONTEXT:" in content:
            return True
    return False

def build_citation_repair_prompt(
    question: str,
    answer: str,
    graph_hits: List[RetrievalHit],
    vec_hits: List[RetrievalHit],
    observability: Optional[bool] = False,
) -> List[Dict[str, str]]:
    """Build a repair prompt when the model emits unsupported citation handles."""

    graph_context = _format_hits(graph_hits, title="Allowed Graph Evidence")
    vec_context = _format_hits(vec_hits, title="Allowed Vector Evidence")
    allowed = " ".join(_allowed_citation_tags(graph_hits) + _allowed_citation_tags(vec_hits)) or "(none)"
    system = (
        "Repair the answer so every citation is valid and every factual recipe claim has support.\n"
        f"Allowed citation tags: {allowed}\n"
        "Use opening tags only, such as [G1] or [V2]. Remove unsupported claims rather than inventing citations.\n"
        "Graph citations [G#] are required for recipe facts, constraints, cautions, labels, nutrition, source, and URL.\n"
        "Vector citations [V#] may support phrasing or semantic context but cannot replace graph evidence for recipe facts.\n"
        "Do not mention graph, vector, database, retrieval, or evidence mechanics.\n"
        "Output ONLY the repaired answer text."
    )
    user = (
        f"USER REQUEST:\n{question}\n\n"
        f"ANSWER_TO_REPAIR:\n{answer}\n\n"
        f"{graph_context}\n\n{vec_context}\n"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if observability:
        LOGGER.info("=====================================================")
        LOGGER.info(_messages_to_prompt(messages))
        LOGGER.info("=====================================================")

    return messages


def generate_answer(
    llm: Any,
    question: str,
    model: str,
    context: str,
    *,
    observability: bool = False,
    max_tokens: int = 65536,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """Compatibility helper for callers that still pass a single context string."""

    if not context.strip():
        return "No supporting recipe data found."
    messages = [
        {"role": "system", "content": "Answer using ONLY the provided recipe context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    if observability:
        LOGGER.info("LLM prompt context length=%d chars", len(context))
    return call_llm_chat(
        llm,
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def ask(question: str, *, max_tokens: int = 65536, temperature: float = 0.2, top_p: float = 0.9) -> str:
    """Small direct chat helper used for smoke tests and direct terminal calls."""

    settings = load_settings()
    llm = load_llm(settings)
    return call_llm_chat(
        llm,
        messages=[{"role": "user", "content": question}],
        model=settings.llm_server_model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


__all__ = [
    "load_llm",
    "generate_answer",
    "ask",
    "call_llm_chat",
    "build_grounding_prompt",
    "build_conversation_seed_prompt",
    "messages_contain_seed_evidence",
    "build_vector_only_prompt",
    "build_refine_prompt",
    "build_citation_repair_prompt",
    "allowed_citation_handles",
    "extract_citations",
    "invalid_citations",
    "contains_closing_citations",
    "strip_invalid_citations",
]
