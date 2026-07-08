#!/usr/bin/env python3
"""Shared retrieval, ranking, context-building, and generation helpers.

The query path uses one OpenSearch index at a time, defaulting to the chunk
index created by ``ingest.py``. Candidate retrieval is handled by OpenSearch;
the default orchestration then re-ranks those candidates locally with ``bm25s``
before building the prompt for the local GGUF model.
"""

from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError
from llama_cpp import Llama

import bm25s

try:
    import Stemmer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Stemmer = None  # type: ignore


_BM25_STOPWORDS = "en"
_STEMMER = Stemmer.Stemmer("english") if Stemmer else None  # type: ignore[attr-defined]


##############################################################################
# Configuration helpers
##############################################################################

def env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


# OpenSearch
OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "")
OPENSEARCH_SSL  = os.getenv("OPENSEARCH_SSL", "false").lower() == "true"
# Chunk-level documents are the default retrieval unit for the RAG path. The
# full-document index still exists from ingest, but query-time context is built
# from chunks unless callers override CHUNK_INDEX_NAME.
INDEX_NAME = env_str("CHUNK_INDEX_NAME", "bbc-bm25-chunks")


##############################################################################
# NER client
##############################################################################

NER_URL = env_str("NER_URL", "http://127.0.0.1:8000/ner")
NER_TIMEOUT_SECS = float(env_str("NER_TIMEOUT_SECS", "5"))

def post_ner(text: str, timeout: float = NER_TIMEOUT_SECS) -> Dict[str, Any]:
    """Call the external NER service and return its JSON payload.

    The RAG query path uses the returned ``entities`` list to bias retrieval
    toward documents that mention the same explicit terms as the question.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}
    r = requests.post(NER_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise SystemExit(f"[NER] HTTP {r.status_code}: {msg}") from e
    try:
        return r.json()
    except Exception as e:
        raise SystemExit(f"[NER] Non-JSON response: {r.text[:800]}") from e


def normalize_entities(ner_result: Dict[str, Any]) -> List[str]:
    """Return normalized (lowercased, de-duped) entity strings."""
    seen, out = set(), []
    for name in ner_result.get("entities", []):
        k = name.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


##############################################################################
# OpenSearch connection
##############################################################################

def connect() -> Tuple[OpenSearch, str]:
    """Create an OpenSearch client and return it with the configured index name."""
    http_auth = (OPENSEARCH_USER, OPENSEARCH_PASS) if OPENSEARCH_USER and OPENSEARCH_PASS else None
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=http_auth,
        use_ssl=OPENSEARCH_SSL,
        verify_certs=OPENSEARCH_SSL,
        ssl_assert_hostname=False if not OPENSEARCH_SSL else None,
        ssl_show_warn=OPENSEARCH_SSL,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client, INDEX_NAME


##############################################################################
# Query builder (lexical-first, auditable)
##############################################################################

def build_query_opensearch_ranking(question: str, entities: List[str]) -> Dict[str, Any]:
    """
    - If no entities -> dis_max over a single match on the full question.
    - If entities exist -> dis_max with two bool branches:
        (A) STRICT / AND-style:
            - terms_set on explicit_terms requiring ALL
            - match on explicit_terms_text with operator='and'
            - multi_match on content/category^0.5 with operator='and'
          boost 30.0
        (B) OR-style:
            - terms on explicit_terms (entities)
            - match on explicit_terms_text (joined)
            - multi_match on content/category^0.5
          boost 10.0
    - No fallback full-question clause when entities are present.
    """
    if not entities:
        return {
            "dis_max": {
                "tie_breaker": 0.0,
                "queries": [
                    {"match": {"content": {"query": question}}}
                ]
            }
        }

    joined = " ".join(entities)

    strict_bool = {
        "bool": {
            "should": [
                {
                    "terms_set": {
                        "explicit_terms": {
                            "terms": entities,
                            "minimum_should_match_script": {"source": "params.num_terms"}
                        }
                    }
                },
                {
                    "match": {
                        "explicit_terms_text": {
                            "query": joined,
                            "operator": "and"
                        }
                    }
                },
                {
                    "multi_match": {
                        "query": joined,
                        "fields": ["content^1.0", "category^0.5"],
                        "operator": "and"
                    }
                }
            ],
            "minimum_should_match": 1,
            "boost": 30.0
        }
    }

    or_bool = {
        "bool": {
            "should": [
                {"terms": {"explicit_terms": entities}},
                {"match": {"explicit_terms_text": joined}},
                {"multi_match": {"query": joined, "fields": ["content^1.0", "category^0.5"]}},
            ],
            "minimum_should_match": 1,
            "boost": 10.0
        }
    }

    return {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": [strict_bool, or_bool]
        }
    }



def build_query_external_ranking(question: str, entities: List[str]) -> Dict[str, Any]:
    """
    Build the broader OpenSearch candidate query used before local BM25 re-ranking.

    - If no entities are found, use a single match query over the full question.
    - If entities exist, use an OR-style branch over explicit entity fields plus
      content/category matches. The stricter AND branch is intentionally left to
      ``build_query_opensearch_ranking`` because this path expects the local
      ``bm25s`` stage to do the final ordering.
    - No fallback full-question clause is added when entities are present.
    """
    if not entities:
        return {
            "dis_max": {
                "tie_breaker": 0.0,
                "queries": [
                    {"match": {"content": {"query": question}}}
                ]
            }
        }

    joined = " ".join(entities)

    or_bool = {
        "bool": {
            "should": [
                {"terms": {"explicit_terms": entities}},
                {"match": {"explicit_terms_text": joined}},
                {"multi_match": {"query": joined, "fields": ["content^1.0", "category^0.5"]}},
            ],
            "minimum_should_match": 1,
            "boost": 10.0
        }
    }

    return {
        "dis_max": {
            "tie_breaker": 0.0,
            "queries": [or_bool]
        }
    }

##############################################################################
# LLM loader and answering
##############################################################################

@lru_cache(maxsize=1)
def load_llm() -> Llama:
    """Load the local GGUF model once per process using llama.cpp."""
    return Llama(
        model_path=str(Path(env_str("MODEL_PATH",
                                    str(Path.home() / "models" / "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"))).expanduser()),
        n_ctx=65536,
        n_threads=max(1, (os.cpu_count() or 4) - 1),
        temperature=0.2,
        top_p=0.80,
        repeat_penalty=1.2,
        chat_format="chatml",
        verbose=False,
    )


##############################################################################
# Search + ranking utilities
##############################################################################

SEARCH_SIZE = env_int("SEARCH_SIZE", 10)
ALPHA       = float(env_str("ALPHA", "0.5"))
PREFERENCE  = env_str("PREFERENCE_TOKEN", "governance-audit-v1")
DO_EXPLAIN  = env_bool("OS_EXPLAIN", False)
DO_PROFILE  = env_bool("OS_PROFILE", False)

HIGHLIGHT = {
    "fields": {"content": {}},
    "fragment_size": 160,
    "number_of_fragments": 2,
    "pre_tags": ["<em>"],
    "post_tags": ["</em>"]
}

SOURCE_FIELDS = [
    "filepath", "category", "content",
    "explicit_terms", "explicit_terms_text",
    "ingested_at_ms", "doc_version"
]

def search_one(
    label: str,
    client: OpenSearch,
    index_name: str,
    query: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute one OpenSearch search and attach metadata used by renderers."""
    body = {
        "query": query,
        "_source": SOURCE_FIELDS,
        "highlight": HIGHLIGHT,
        "size": SEARCH_SIZE,
        "explain": DO_EXPLAIN,
        "profile": DO_PROFILE,
        "track_total_hits": True,
    }
    try:
        res = client.search(index=index_name, body=body, preference=PREFERENCE, request_timeout=10)
    except TransportError as e:
        # Soft-fail: return empty result + diagnostic
        return {
            "_store_label": label,
            "_index_used": index_name,
            "_query": query,
            "_error": f"{e.__class__.__name__}: {getattr(e, 'error', str(e))}",
            "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}
        }
    res["_store_label"] = label
    res["_index_used"]  = index_name
    res["_query"]       = query
    return res


def rerank_hits_with_bm25(
    question: str,
    res: Dict[str, Any],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Re-rank OpenSearch candidate hits with a local BM25 scorer.

    Args:
        question: Natural-language query posed by the user.
        res: Raw OpenSearch response dictionary containing the hits payload.
        top_k: Maximum number of combined hits to return.

    Returns:
        Re-ranked hits limited to ``top_k`` documents. The returned hit objects
        preserve the OpenSearch source payload and include ``_bm25_score``.
    """
    actual_hits = res.get("hits", {}).get("hits", []) or []
    
    corpus: List[str] = []
    
    # Re-rank only the candidate set returned by OpenSearch. This keeps the
    # expensive local scorer bounded while letting OpenSearch handle recall.
    for h in actual_hits:
        content = h.get("_source", {}).get("content", "")
        corpus.append(content)

    corpus_tokens = bm25s.tokenize(corpus, stopwords=_BM25_STOPWORDS, stemmer=_STEMMER)
    has_tokens = False
    for doc_tokens in corpus_tokens:
        if len(doc_tokens) > 0:
            has_tokens = True
            break
    query_tokens = bm25s.tokenize(question, stemmer=_STEMMER)

    if not has_tokens or not query_tokens:
        sorted_hits = sorted(
            actual_hits,
            key=lambda h: h.get("_score", float("-inf")),
            reverse=True,
        )
        top_hits = sorted_hits[: min(top_k, len(sorted_hits))]
        for hit in top_hits:
            hit.setdefault("_bm25_score", hit.get("_score"))
    else:
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        k = min(top_k, len(actual_hits))
        
        if k == 0:
            return []
            
        results, scores = retriever.retrieve(query_tokens, k=k)

        # bm25s returns arrays shaped (n_queries, k). We only issue one query.
        doc_ids = list(results[0])
        doc_scores = list(scores[0])

        top_hits = []
        for doc_id, score in zip(doc_ids, doc_scores):
            doc_index = int(doc_id)
            if doc_index < 0 or doc_index >= len(actual_hits):
                continue
            hit = actual_hits[doc_index]
            if "_original_score" not in hit and "_score" in hit:
                hit["_original_score"] = hit["_score"]
            hit["_score"] = float(score)
            hit["_bm25_score"] = float(score)
            top_hits.append(hit)

        if not top_hits:
            sorted_hits = sorted(
                actual_hits,
                key=lambda h: h.get("_score", float("-inf")),
                reverse=True,
            )
            top_hits = sorted_hits[: min(top_k, len(sorted_hits))]

    combined = top_hits[: min(top_k, len(top_hits))]
    for hit in combined:
        hit.setdefault("_bm25_score", hit.get("_score"))

    return combined


def rank_hits(res: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Keep OpenSearch-ranked hits whose score is at least ``ALPHA * top_score``.

    This path is used when local BM25 re-ranking is disabled.
    """
    hits = res.get("hits", {}).get("hits", []) or []
    if not hits:
        return []
    top1 = hits[0]["_score"]
    keep = [h for h in hits if h["_score"] >= ALPHA * top1]
    return keep


def combine_hits(hits: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Return the first ``top_k`` hits while preserving the incoming rank order."""
    combined: List[Dict[str, Any]] = []
    ia = ib = 0
    while len(combined) < top_k and ia < len(hits):
        if ia < len(hits):
            combined.append(hits[ia]); ia += 1
        if len(combined) >= top_k:
            break
    return combined


def render_observability_summary(res: Dict[str, Any]) -> str:
    """Render a compact status line for an OpenSearch response."""
    err = res.get("_error")
    store = res.get("_store_label", "?")
    idx   = res.get("_index_used", "?")
    total = res.get("hits", {}).get("total", {}).get("value", 0)
    if err:
        return f"[SUMMARY] STORE={store} INDEX={idx} ERROR: {err}"
    return f"[SUMMARY] STORE={store} INDEX={idx} TOTAL={total}"


def render_matches(hits: List[Dict[str, Any]]) -> str:
    """Render retrieved hits with source paths, scores, and highlights."""
    lines: List[str] = []
    lines.append("\n================= MATCH EXPLANATIONS =================")
    if not hits:
        lines.append("(no documents kept from either store)")
        lines.append("======================================================\n")
        return "\n".join(lines)
    for i, h in enumerate(hits, 1):
        store = h.get("_store_label", "?")
        idx   = h.get("_index_used", "?")
        fp    = h.get("_source", {}).get("filepath", "<unknown>")
        score = h.get("_score")
        lines.append(f"\n[{i}] STORE={store} INDEX={idx} SCORE={score:.4f}")
        lines.append(f"     DOC={fp}")
        if "highlight" in h and "content" in h["highlight"]:
            frag = h["highlight"]["content"][0]
            lines.append(f"     highlight: {frag}")
        lines.append("")
    lines.append("======================================================\n")
    return "\n".join(lines)


def build_context(hits: List[Dict[str, Any]], max_chars_per_doc: int = 0) -> str:
    """Render retrieved chunks into the plain-text context block sent to the LLM."""
    if not hits:
        return ""
    
    out = []
    for h in hits:
        src = h.get("_source", {})
        fp  = src.get("filepath", "<unknown>")
        store = h.get("_store_label", "?")
        content = (src.get("content") or "").strip()
        if max_chars_per_doc > 0:
            content = content[:max_chars_per_doc]
        out.append(f"---\nStore: {store}\nDoc: {fp}\n{content}\n")
    return "\n".join(out)


def save_results(path: str, payload: Dict[str, Any]) -> None:
    """Append one compact JSONL observability record."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


##############################################################################
# Orchestrator
##############################################################################

def generate_answer(llm: Llama, question: str, context: str, observability: bool = False) -> str:
    """Ask the local model to answer using only the retrieved context."""
    if not context.strip():
        return "No supporting documents found."
    
    sys_msg = "Answer using ONLY the provided context below."
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        # f"If context lacks the answer, reply with the following and nothing else: No relevant information found."
    )

    if observability:
        print("\n\n=== Prompt to LLM ===")
        print(user_prompt)
        print("==========\n\n")

    resp = llm.create_chat_completion(
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user", "content": user_prompt}],
        temperature=0.2, top_p=0.8, max_tokens=65536,
    )
    return resp["choices"][0]["message"]["content"].strip()


def ask(llm: Llama, question: str, *, observability: bool = True,  external_ranker: bool = True,
        top_k: int = 10, save_path: str | None = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Run the full RAG flow: NER, retrieval, ranking, context building, generation."""
    # 1) Extract entities from the question for structured lexical retrieval.
    ner = post_ner(question)
    entities = normalize_entities(ner)

    # 2) Build the OpenSearch candidate query. The external-ranker path keeps
    # the query broader because bm25s will re-score the candidates locally.
    if external_ranker:
        if observability:
            print("\n\nUsing EXTERNAL ranking with BM25 re-ranking after retrieval.\n\n")
        query = build_query_external_ranking(question, entities)
    else:
        if observability:
            print("\n\nUsing INTERNAL OpenSearch ranking only.\n\n")
        query = build_query_opensearch_ranking(question, entities)

    if observability:
        if entities:
            print(f"[NER] entities: {entities}")
            print("\n[QUERY] dis_max (entity path):")
        else:
            print("[NER] No entities detected; using full-question match only.")
            print("\n[QUERY] dis_max (no-entity path):")
        print(json.dumps(query, indent=2))

    # 3) Connect to the configured OpenSearch index.
    client, index = connect()

    # 4) Execute the search. The executor wrapper is retained from earlier demos,
    # but this version submits one search against one configured index.
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut = ex.submit(search_one, "LONG", client, index, query)
        res = fut.result()

    # 5) Rank candidates, either with local BM25 re-ranking or OpenSearch scores.
    if external_ranker:
        combined = rerank_hits_with_bm25(question, res, top_k=top_k)
    else:
        keep = rank_hits(res)
        combined  = combine_hits(keep, top_k=top_k)
    
    # 6) Optional observability prints
    if observability:
        print(render_observability_summary(res))
        print(f"\n[RESULTS] kept={len(keep)} of {len(res.get('hits',{}).get('hits',[]))}")
        print(render_matches(combined))

    # 7) Optional save (compact JSONL)
    if save_path:
        payload = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "question": question,
            "entities": entities,
            "alpha": ALPHA,
            "size": SEARCH_SIZE,
            "preference": PREFERENCE,
            "long": {
                "index": res.get("_index_used"),
                "total": res.get("hits", {}).get("total", {}).get("value", 0),
                "error": res.get("_error"),
                "kept_filepaths": [h.get("_source", {}).get("filepath") for h in keep],
            },
            "combined_filepaths": [h.get("_source", {}).get("filepath") for h in combined],
            # do NOT dump full content; keep this audit-friendly and light
        }
        save_results(save_path, payload)

    # 8) Build context and answer. Empty context short-circuits in generate_answer().
    context_block = build_context(combined)
    
    answer = generate_answer(llm, question, context_block, observability)
    return answer, combined
