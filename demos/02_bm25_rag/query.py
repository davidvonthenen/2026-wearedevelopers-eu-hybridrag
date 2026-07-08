#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line entry point for the local BM25 RAG demo.

The query path extracts entities from the user's question, searches the
OpenSearch chunk index, optionally re-ranks candidate chunks with the local
``bm25s`` library, and sends the assembled context to a local GGUF model loaded
with ``llama_cpp``.

Observability controls default to off:
  --observability           Print query JSON, match summaries, and the LLM prompt
  --save-results PATH       Append compact JSONL records for later inspection

Environment overrides used by the shared helpers:
  OPENSEARCH_HOST=localhost
  OPENSEARCH_PORT=9200
  OPENSEARCH_USER=
  OPENSEARCH_PASS=
  OPENSEARCH_SSL=false
  CHUNK_INDEX_NAME=bbc-bm25-chunks
  SEARCH_SIZE=10
  ALPHA=0.5
  PREFERENCE_TOKEN=governance-audit-v1
  OS_EXPLAIN=false
  OS_PROFILE=false
  NER_URL=http://127.0.0.1:8000/ner
  NER_TIMEOUT_SECS=5
  MODEL_PATH=~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf
"""

from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ---------------------------------------------------------------------------
# Import the shared retrieval and generation helpers.
# ---------------------------------------------------------------------------
try:
    from common.common import (
        ask,                    # orchestrator: NER -> OpenSearch -> optional bm25s rerank -> LLM answer
        load_llm,               # cached local GGUF loader via llama.cpp
    )
except Exception as e:
    raise SystemExit(
        "This script expects 'common/common.py' to be importable from the project root.\n"
        "Run it from this directory or set PYTHONPATH so the common package is visible.\n"
        f"Import error: {e}"
    )

import requests
from opensearchpy import OpenSearch
from opensearchpy.exceptions import TransportError
from llama_cpp import Llama


##############################################################################
# CLI
##############################################################################

def main():
    parser = argparse.ArgumentParser(description="Single-index BM25 RAG query path with optional local BM25 re-ranking.")
    parser.add_argument("--question", help="User question to answer.")
    parser.add_argument("--observability", action="store_true", default=False,
                        help="Print retrieval diagnostics and the prompt sent to the local model (default OFF).")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Append compact JSONL result records to this path (default OFF).")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top documents to retrieve and provide to the LLM (default 10).")
    args = parser.parse_args()

    llm = load_llm()

    questions: List[str]
    if args.question:
        questions = [args.question]
    else:
        # Demo questions (safe to replace for your corpus)
        questions = [
            "How much did OpenAI purchase Windsurf for?",
            "How much did Google purchase Windsurf for?",
        ]

    for q in questions:
        t0 = time.time()
        answer, hits = ask(llm, q, observability=args.observability, save_path=args.save_results, top_k=args.top_k)
        dt = time.time() - t0

        print("\n" + "=" * 88)
        print(f"QUESTION: {q}")
        print("=" * 88)
        print("")
        print("=" * 88)
        print(f"ANSWER: {answer}")
        print("=" * 88)
        print(f"\nQuery time: {dt:.2f}s")
        print(f"Docs provided to LLM: {len(hits)}\n\n")


if __name__ == "__main__":
    main()
