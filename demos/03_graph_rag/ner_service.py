#!/usr/bin/env python3
"""
Synchronous NER service for the Graph-based RAG demo.

- Extracts entities with spaCy and normalizes them for downstream matching.
- Returns both a deduplicated list of entity names and detailed `(name, label)`
  pairs so ingest/query code can choose the shape it needs.
- Does not read from or write to Neo4j. Ingest and query scripts consume this
  service response and perform graph writes/reads themselves.

Endpoint
--------
POST /ner
    Request (JSON):
        {
          "text": "Your input text...",
          "labels": ["PERSON", "ORG", "GPE"]   # optional label allowlist
        }

    Additional fields such as `promote` and `ttl_ms` are ignored by this local
    service but tolerated by the shared client helper used across demos.

    Response (JSON):
        {
          "text": "...",
          "model": "en_core_web_sm",
          "entities": ["openai", "san francisco"],
          "entity_pairs": [{"name": "openai", "label": "ORG"}, ...],
          "request_id": "..."
        }

Run
---
$ export SPACY_MODEL=en_core_web_sm
$ pip install flask spacy

# For miniconda or venv:
$ pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
# OR, if using uv:
$ uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

$ python ner_service.py
"""

import os
import time
import uuid
from datetime import datetime, timezone
from typing import List, Tuple

from flask import Flask, jsonify, request
from functools import lru_cache
import spacy

# ---------------------------
# Environment
# ---------------------------

SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

# Default entity labels used when the caller does not supply a custom allowlist.
DEFAULT_INTERESTING_ENTITY_TYPES = {
    "PERSON",
    "ORG",
    "PRODUCT",
    "GPE",
    "EVENT",
    "WORK_OF_ART",
    "NORP",
    "LOC",
    "FAC",
}

# ---------------------------
# spaCy load
# ---------------------------

@lru_cache(maxsize=1)
def load_spacy():
    """Load the spaCy pipeline once per process.

    Model startup is expensive enough that doing it on every request would
    add unnecessary latency to the demo service.
    """
    return spacy.load(SPACY_MODEL)


nlp = load_spacy()

# ---------------------------
# Entity Extraction
# ---------------------------

def _extract_entities(
    nlp_obj: spacy.Language, text: str, allowed_labels: set[str]
) -> List[Tuple[str, str]]:
    """Return normalized `(entity_text, label)` pairs accepted by the label set.

    Entity text is lowercased so Neo4j lookups can use a stable representation
    across ingest and query paths. Very short spans are dropped because they tend
    to be weak retrieval anchors in small news-style corpora.
    """
    doc = nlp_obj(text)
    return [
        (ent.text.strip().lower(), ent.label_)
        for ent in doc.ents
        if ent.label_ in allowed_labels and len(ent.text.strip()) >= 3
    ]

# ---------------------------
# Flask App (synchronous path)
# ---------------------------

app = Flask(__name__)

@app.route("/ner", methods=["POST"])
def ner():
    """Extract named entities from the request text and return JSON.

    The response intentionally includes two views of the same extraction:
    `entities` for quick name-only matching and `entity_pairs` for code that
    needs labels to reduce false positives.
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data or not isinstance(data["text"], str):
        return (
            jsonify(
                {
                    "error": "Invalid request",
                    "detail": "Expected JSON with a 'text' string field.",
                }
            ),
            400,
        )

    text: str = data["text"]
    labels_field = data.get("labels")
    if isinstance(labels_field, list) and labels_field:
        allowed = {str(l).upper() for l in labels_field}
    else:
        allowed = DEFAULT_INTERESTING_ENTITY_TYPES

    # Preserve labels for graph writes, then separately deduplicate by entity
    # name for clients that only need a compact list of normalized strings.
    entity_pairs = _extract_entities(nlp, text, allowed)
    normalized_entities: List[str] = []
    seen_names = set()
    for name, _label in entity_pairs:
        if name not in seen_names:
            seen_names.add(name)
            normalized_entities.append(name)

    print(
        f"[{datetime.now(timezone.utc).isoformat()}] /ner called, {len(text)} chars, {len(normalized_entities)} entities"
    )
    print(f"[entities discovered] {normalized_entities}\n")

    request_id = str(uuid.uuid4())
    status_code = 200

    payload = {
        "text": text,
        "model": SPACY_MODEL,
        "entities": normalized_entities,
        "entity_pairs": [{"name": name, "label": label} for name, label in entity_pairs],
        "request_id": request_id,
    }

    return jsonify(payload), status_code

# ---------------------------
# Local Dev Entrypoint
# ---------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "8000")), debug=False)
