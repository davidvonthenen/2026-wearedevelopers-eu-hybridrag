# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
query.py ― Query pipeline for the Graph-based RAG agent.

Core responsibilities
---------------------
- Detect entities in a user question via the local NER REST service.
- Use those `(name, label)` pairs to retrieve paragraph candidates from Neo4j.
- Rerank the graph-retrieved candidates with BM25.
- Build a compact context block and pass it to a local GGUF model through
  `llama-cpp-python`.

Important scope notes
---------------------
- This version uses a single Neo4j database. It does not maintain separate
  long-term and short-term graph stores.
- The local NER service does not promote data into Neo4j. The `promote` and
  `ttl_ms` request fields are forwarded only for compatibility with related
  demos that used those fields.
- `expiration` is still checked in Cypher because the ingest writes it on graph
  relationships. A value of `0` means the relationship is active.

Environment variables
---------------------
| Variable                | Default                                      | Purpose                      |
|-------------------------|----------------------------------------------|------------------------------|
| NER_SERVICE_URL         | http://127.0.0.1:8000/ner                    | Endpoint for entity extraction |
| NER_SERVICE_TIMEOUT     | 8.0                                          | NER HTTP timeout in seconds   |
| NEO4J_URI               | bolt://localhost:7687                        | Neo4j graph store             |
| NEO4J_USER              | neo4j                                        | Neo4j username                |
| NEO4J_PASSWORD          | neo4jneo4j                                   | Neo4j password                |
| MODEL_PATH              | ~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf  | GGUF checkpoint for llama-cpp |
"""

##############################################################################
# Imports
##############################################################################

import os
import sys
import time
import argparse
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Tuple

import bm25s
import Stemmer
from llama_cpp import Llama
from neo4j import GraphDatabase, Session

from common.common import NerServiceError, call_ner_service, create_indexes, parse_entity_pairs

##############################################################################
# Configuration
##############################################################################

# Neo4j graph store populated by ingest.py.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")

# Local GGUF checkpoint consumed by llama-cpp-python.
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path.home() / "models" / "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"),
)

# Labels considered useful for entity-grounded retrieval. Broader labels can
# increase recall, but they also tend to add noisy graph matches.
INTERESTING_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "PRODUCT",
    "GPE",
    "EVENT",
    "WORK_OF_ART",
    "NORP",
    "LOC",
}

# Compatibility value forwarded to the NER service. The local service ignores
# TTL, but keeping the argument lets this client share the same call shape as
# adjacent demos that support relationship expiration updates.
TTL_MS = 60 * 60 * 1000  # one hour in milliseconds

# BM25 reranker configuration. Neo4j supplies candidates; BM25 orders them by
# lexical similarity to the question before prompt construction.
BM25_TOP_K = 10
_STEMMER = Stemmer.Stemmer("english")

##############################################################################
# Helpers - LLM
##############################################################################

@lru_cache(maxsize=1)
def load_llm() -> Llama:
    """Load the local GGUF model once and reuse it for every question.

    `lru_cache` keeps repeated demo queries from reloading a multi-GB model
    for every question.
    """
    return Llama(
        model_path=MODEL_PATH,
        n_ctx=65536,
        n_threads=max(1, (os.cpu_count() or 4) - 1),
        temperature=0.2,
        top_p=0.8,
        repeat_penalty=1.2,
        chat_format="chatml",
        verbose=False,
    )

##############################################################################
# Neo4j connections
##############################################################################

def connect():
    """Return a Neo4j driver pointed at the graph store populated by ingest.py."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


##############################################################################
# Retrieval query over the Neo4j graph.
#
# The query matches question entities against paragraph-level MENTIONS edges,
# filters out expired relationships, and returns document metadata so the LLM
# context can show where each paragraph came from.
##############################################################################

FETCH_PARAS_QUERY = """
MATCH (e:Entity)-[m:MENTIONS]->(t:Paragraph)
WHERE [toLower(e.name), e.label] IN $entity_list
  AND ( m.expiration IS NULL
     OR m.expiration = 0
     OR m.expiration > $now )
OPTIONAL MATCH (t)-[:PART_OF]->(d:Document)
WITH t, d, count(e) AS matchingEntities
ORDER BY matchingEntities DESC,
         coalesce(t.index, 0) ASC
LIMIT $topK
RETURN
  t.text                                   AS text,
  t.index                                  AS idx,
  coalesce(d.title, 'Untitled')            AS title,
  coalesce(d.category, 'N/A')              AS category,
  matchingEntities
"""


def fetch_paragraphs(
    session: Session, entity_pairs: Iterable[Tuple[str, str]], top_k: int = 100
):
    """Return graph paragraph candidates ranked by matched entity count.

    Neo4j handles the structural filter: an entity from the question must point
    to the paragraph through a `MENTIONS` relationship. BM25 ranking happens in
    `rerank_paragraphs` after this broader graph candidate set is returned.
    """
    if not entity_pairs:
        return []

    now_ms = int(time.time() * 1000)
    entity_list = [[name, label] for name, label in entity_pairs]

    return [
        dict(r)
        for r in session.run(
            FETCH_PARAS_QUERY,
            entity_list=entity_list,
            now=now_ms,
            topK=top_k,
        )
    ]


def rerank_paragraphs(question: str, paragraphs: List[dict], top_k: int = BM25_TOP_K) -> List[dict]:
    """Re-rank graph-retrieved paragraphs with BM25-S.

    Args:
        question: The natural-language query supplied by the user.
        paragraphs: Raw paragraph dictionaries retrieved from Neo4j.
        top_k: Number of paragraphs to return after lexical reranking.

    Returns:
        The highest-scoring ``top_k`` paragraphs with an added ``bm25_score`` field.
    """

    if not paragraphs or not question.strip():
        return []

    # BM25-S expects tokenized documents. The corpus order is preserved so the
    # returned document IDs can index back into the original paragraph records.
    corpus = [p.get("text", "") for p in paragraphs]
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=_STEMMER)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    query_tokens = bm25s.tokenize(question, stemmer=_STEMMER)
    k = min(top_k, len(corpus))
    doc_ids, scores = retriever.retrieve(query_tokens, k=k)

    reranked = []
    for doc_id, score in zip(doc_ids[0], scores[0]):
        para = paragraphs[int(doc_id)].copy()
        para["bm25_score"] = float(score)
        reranked.append(para)

    return reranked


##############################################################################
# LLM answer generation
##############################################################################


def generate_answer(llm: Llama, question: str, context: str) -> str:
    """Generate an answer constrained to the retrieved graph context."""
    sys_msg = "You are an expert assistant answering the user's question using only the provided context."
    prompt = f"{context}\n\nQuestion: {question}\n\nAnswer concisely."
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=65536,
    )
    return resp["choices"][0]["message"]["content"].strip()


##############################################################################
# Demo loop helpers
##############################################################################


def ask(llm: Llama, question: str, driver, top_k: int = BM25_TOP_K) -> str:
    """Run the end-to-end graph RAG cycle for one question.

    Flow:
      1. Extract question entities through the NER REST service.
      2. Retrieve candidate paragraphs from Neo4j using entity relationships.
      3. Rerank those candidates with BM25.
      4. Ask the local model to answer from the compact context block.
    """
    print(f"\n\u25B6  {question}")

    try:
        ner_response = call_ner_service(
            question,
            promote=True,
            ttl_ms=TTL_MS,
            labels=INTERESTING_ENTITY_TYPES,
        )
    except NerServiceError as exc:
        print(f"  Failed to contact NER service: {exc}")
        return
    ent_pairs = parse_entity_pairs(ner_response)
    if not ent_pairs:
        print("  (no interesting entities detected)")
        return

    # Note: spelling in print statement preserved to avoid code changes
    print(f"Erntities found: {ent_pairs}")

    # Fetch graph candidates using entity overlap before applying BM25 reranking.
    with driver.session() as sess:
        paras = fetch_paragraphs(sess, ent_pairs, top_k=(top_k * 10))

    if not paras:
        print("  No relevant context found.")
        return

    reranked_paras = rerank_paragraphs(question, paras, top_k=top_k)
    if not reranked_paras:
        print("  Unable to rerank retrieved context.")
        return

    # Build a compact context block for the LLM. Snippets are deliberately
    # truncated to keep retrieved context focused instead of recreating the
    # entire article in the prompt.
    context_block = ""
    for p in reranked_paras:
        snippet = p["text"][:350].replace("\n", " ")
        context_block += (
            f"\n---\nDoc: {p['title']} | Para #{p['idx']} "
            f"| Matches: {p['matchingEntities']} | BM25: {p.get('bm25_score', 0):.2f}\n{snippet}…"
        )

    answer = generate_answer(llm, question, context_block)
    return answer


def main():
    """Run one user-supplied question or the two default demo questions."""
    driver = connect()
    llm = load_llm()

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=str, help="Ask a single question and exit")
    args = parser.parse_args()

    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        # Demo questions that contrast likely and unlikely entity-grounded facts.
        questions = [
            "How much did OpenAI purchase Windsurf for?",
            "How much did Google purchase Windsurf for?",
        ]

    for q in questions:
        t0 = time.time()
        answer = ask(llm, q, driver)
        dt = time.time() - t0

        print("\n" + "=" * 88)
        print(f"QUESTION: {q}")
        print("=" * 88)
        print("")
        print("=" * 88)
        print(f"ANSWER: {answer}")
        print("=" * 88)
        print(f"\nQuery time: {dt:.2f}s")
    else:
        print("No question provided.")
    
if __name__ == "__main__":
    main()
