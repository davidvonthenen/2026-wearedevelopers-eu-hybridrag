#!/usr/bin/env python3
"""
Ingest BBC-style text articles into OpenSearch for BM25/full-text retrieval.

The ingest path writes both full documents and fixed-size overlapping chunks.
Each document and chunk is annotated with entities returned by the local NER
service so later queries can combine normal text matching with structured
entity-term matching.
"""
import os, glob, time
from pathlib import Path
from typing import Dict, List, Optional
import requests
import json

from opensearchpy import OpenSearch

# ----------------------------
# Config
# ----------------------------
DATA_DIR   = os.getenv("DATA_DIR", "bbc")
FULL_INDEX_NAME = os.getenv("FULL_INDEX_NAME", "bbc-bm25-full")
CHUNK_INDEX_NAME = os.getenv("CHUNK_INDEX_NAME", "bbc-bm25-chunks")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER", "")
OPENSEARCH_PASS = os.getenv("OPENSEARCH_PASS", "")
OPENSEARCH_SSL  = os.getenv("OPENSEARCH_SSL", "false").lower() == "true"

# Local NER service endpoint used during ingest.
DEFAULT_URL = "http://127.0.0.1:8000/ner"
CHUNK_SIZE = 2048
CHUNK_OVERLAP = 256

# ----------------------------
# NER
# ----------------------------
def post_ner(text: str, timeout: float = 5.0) -> dict:
    """Send text to the local NER service and return its JSON payload."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": text,
    }
    r = requests.post(DEFAULT_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    # Raise for non-2xx to surface useful diagnostics
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        # try to show server-provided JSON error if present
        try:
            msg = r.json()
        except Exception:
            msg = r.text
        raise SystemExit(f"HTTP {r.status_code} from server: {msg}") from e
    try:
        return r.json()
    except Exception as e:
        raise SystemExit(f"Server returned non-JSON response: {r.text[:2000]}") from e


def extract_ner_data(text: str) -> dict:
    """Return the full NER payload including normalized entities and structured entity pairs."""
    return post_ner(text)


def extract_normalized_entities(text_file: Path) -> List[str]:
    """Read a text file and return normalized entity names; retained for older callers."""
    text = text_file.read_text(encoding="utf-8", errors="ignore")
    return extract_ner_data(text).get("entities", [])


def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into fixed-size chunks with character overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text.strip():
        return []

    chunks: List[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start += step

    return chunks

# ----------------------------
# OpenSearch connection
# ----------------------------
def connect_long() -> OpenSearch:
    """Create the OpenSearch client used for both full and chunk indexes."""
    http_auth = (OPENSEARCH_USER, OPENSEARCH_PASS) if OPENSEARCH_USER and OPENSEARCH_PASS else None
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
        http_auth=http_auth,
        use_ssl=OPENSEARCH_SSL,
        verify_certs=OPENSEARCH_SSL,
        ssl_assert_hostname=False if OPENSEARCH_SSL else None,
        ssl_show_warn=OPENSEARCH_SSL,
        timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
    return client

# ----------------------------
# Index + mapping
# ----------------------------
def ensure_index(
    client: OpenSearch,
    index_name: str,
    extra_properties: Optional[Dict[str, dict]] = None,
) -> None:
    """Create an OpenSearch index with fields needed for BM25 and entity matching."""
    if client.indices.exists(index=index_name):
        return
    body = {
        "settings": {
            "analysis": {
                # Keyword fields are normalized to lowercase so entity matches are
                # case-insensitive while still using exact keyword matching.
                "normalizer": {
                    "lowercase_normalizer": {
                        "type": "custom",
                        "char_filter": [],
                        "filter": ["lowercase"]
                    }
                }
            },
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "category": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "filepath": {"type": "keyword"},
                "explicit_terms": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                "explicit_terms_text": {"type": "text"},
                # Keep the original entity label beside each normalized name. The
                # query path mostly uses explicit_terms, but these pairs remain
                # available for inspection or future label-aware retrieval.
                "entity_pairs": {
                    "type": "nested",
                    "properties": {
                        "name": {"type": "keyword", "normalizer": "lowercase_normalizer"},
                        "label": {"type": "keyword"}
                    }
                },
                "ingested_at_ms": {"type": "date", "format": "epoch_millis"},
                "doc_version": {"type": "long"}
            }
        },
    }
    if extra_properties:
        body["mappings"]["properties"].update(extra_properties)
    client.indices.create(index=index_name, body=body)

# ----------------------------
# Ingest
# ----------------------------
def ingest_bbc(client: OpenSearch, data_dir: str, index_name: str, chunk_index_name: str) -> None:
    """Index BBC-style files as full documents plus chunk-level retrieval records."""
    ensure_index(client, index_name)
    ensure_index(
        client,
        chunk_index_name,
        extra_properties={
            "chunk_index": {"type": "integer"},
            "chunk_count": {"type": "integer"},
            "parent_filepath": {"type": "keyword"},
        },
    )

    files = sorted(glob.glob(os.path.join(data_dir, "*", "*.txt")))
    if not files:
        print(f"[INFO] No files found under {data_dir}/*/*.txt"); return

    now_ms = int(time.time() * 1000)

    for fp in files:
        p = Path(fp)
        category = p.parent.name
        text = p.read_text(encoding="utf-8", errors="ignore")

        ner_response = extract_ner_data(text)
        explicit_terms = ner_response.get("entities", [])
        entity_pairs = ner_response.get("entity_pairs", [])

        # Use the path string as a deterministic OpenSearch id so re-ingest can
        # overwrite the same full-document record instead of creating duplicates.
        doc_id = p.as_posix()

        doc = {
            "content": text,
            "category": category,
            "filepath": doc_id,
            "explicit_terms": explicit_terms,
            "explicit_terms_text": " ".join(explicit_terms) if explicit_terms else "",
            "entity_pairs": entity_pairs,
            "ingested_at_ms": now_ms,
            "doc_version": now_ms,  # monotonic version; update on re-ingest
        }

        print(f"\n\u27A4  {doc_id}  [{category}] ({len(text)} chars)")
        client.index(index=index_name, id=doc_id, body=doc, refresh=False)

        # Fixed-size overlapping chunks are the default granularity for the RAG agent.
        chunks = split_into_chunks(text)
        if not chunks:
            continue

        # Remove stale chunks for this parent before writing the current chunk set;
        # otherwise shorter edited documents could leave old trailing chunks behind.
        client.delete_by_query(
            index=chunk_index_name,
            body={"query": {"term": {"parent_filepath": doc_id}}},
            conflicts="proceed",
            refresh=False,
        )

        chunk_count = len(chunks)
        for idx, chunk in enumerate(chunks):
            chunk_ner = extract_ner_data(chunk)
            chunk_terms = chunk_ner.get("entities", [])
            chunk_pairs = chunk_ner.get("entity_pairs", [])
            
            chunk_doc = {
                "content": chunk,
                "category": category,
                "filepath": doc_id,
                "parent_filepath": doc_id,
                "chunk_index": idx,
                "chunk_count": chunk_count,
                "explicit_terms": chunk_terms,
                "explicit_terms_text": " ".join(chunk_terms) if chunk_terms else "",
                "entity_pairs": chunk_pairs,
                "ingested_at_ms": now_ms,
                "doc_version": now_ms,
            }
            chunk_doc_id = f"{doc_id}::chunk-{idx:03d}"
            print(f" {chunk_doc_id}  [{category}] ({len(chunk)} chars)")
            client.index(index=chunk_index_name, id=chunk_doc_id, body=chunk_doc, refresh=False)

    client.indices.refresh(index=index_name)
    client.indices.refresh(index=chunk_index_name)
    print(
        f"[OK] Ingest complete. Indexed {len(files)} docs into '{index_name}' and "
        f"{chunk_index_name}"
    )

def main():
    """Run one ingest pass using environment-configured paths and indexes."""
    client = connect_long()
    ingest_bbc(client, DATA_DIR, FULL_INDEX_NAME, CHUNK_INDEX_NAME)

if __name__ == "__main__":
    main()
