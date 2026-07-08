# Copyright 2025 NetApp, Inc. All Rights Reserved.

#!/usr/bin/env python3
"""
ingest.py ― Build the Neo4j graph used by the Graph-based RAG demo.

The script walks a folder of plain-text documents and turns each file into a
small knowledge graph:

    (:Document)<-[:PART_OF]-(:Paragraph)
    (:Entity)-[:MENTIONS {expiration:0}]->(:Paragraph|:Document)

Key design points
-----------------
- **Paragraph-level retrieval** - Text is split on blank lines, so downstream
  retrieval can return focused paragraphs instead of entire articles.
- **Entity grounding** - A local Flask NER service extracts `(name, label)`
  pairs. The ingest script stores those entities and links them to the
  paragraphs/documents where they appeared.
- **Governance hook** - Nodes and relationships receive an `expiration` field.
  The current ingest sets it to `0`, which the query path treats as active.
- **Fixed label allowlist** - `ALLOWED_LABELS` below controls which spaCy entity
  labels are stored. There is no `NER_TYPES` environment variable in this
  version; changing the allowlist requires editing this script.

Environment variables
---------------------
NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    Connection info for the Neo4j database.
DATA_DIR
    Root folder that holds category sub-directories of `.txt` files
    (default: ./bbc).
"""

import os
import uuid
from pathlib import Path

from neo4j import GraphDatabase
from common.common import call_ner_service, create_indexes, parse_entity_pairs

###############################################################################
# Configuration
###############################################################################

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")
DATASET_PATH   = os.getenv("DATA_DIR", "./bbc")

# Entity labels accepted from the NER service. Keeping this list small prevents
# very broad entity classes from flooding the graph with weak retrieval signals.
_raw_labels = {
    "PERSON", "ORG", "PRODUCT", "GPE", "EVENT",
    "WORK_OF_ART", "NORP", "LOC"
}
ALLOWED_LABELS = set(_raw_labels)

###############################################################################
# Cypher write helpers - each function runs inside a single driver tx
###############################################################################

def merge_entity(tx, ent_uuid: str, name: str, label: str) -> str:
    """Return the UUID for the (:Entity) identified by ``name``.

    Args:
        tx: Neo4j transaction context provided by ``session.execute_write``.
        ent_uuid: Fresh UUID candidate used when the entity is first created.
        name: Canonical (case-insensitive) entity surface form.
        label: spaCy entity label (PERSON, ORG, …).

    Returns:
        The persistent UUID stored on the entity node.  Existing nodes keep
        their original UUID; new nodes adopt ``ent_uuid``.
    """

    record = tx.run(
        """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.ent_uuid   = $ent_uuid,
            e.label      = $label,
            e.expiration = 0
        SET e.ent_uuid = coalesce(e.ent_uuid, $ent_uuid)
        RETURN e.ent_uuid AS ent_uuid
        """,
        name=name.lower().strip(),
        ent_uuid=ent_uuid,
        label=label,
    ).single()

    return record["ent_uuid"]

def create_document(tx, doc_uuid: str, title: str, content: str, category: str) -> None:
    """Store the source document body and category in Neo4j.

    The current pipeline clears the database at the start of `main`, but this
    helper still uses `MERGE` so the write operation is safe if it is reused in
    a smaller incremental ingest flow later.
    """
    tx.run(
        """
        MERGE (d:Document {doc_uuid: $doc_uuid})
        ON CREATE SET
            d.title      = $title,
            d.content    = $content,
            d.category   = $category,
            d.expiration = 0
        """,
        doc_uuid=doc_uuid,
        title=title,
        content=content,
        category=category,
    )

def create_paragraph(tx, para_uuid: str, text: str, idx: int, doc_uuid: str) -> None:
    """
    Persist a Paragraph node and link it to its parent Document with a
    [:PART_OF] relationship.
    """
    # Store the retrieval unit. The paragraph UUID is generated per ingest run,
    # while `doc_uuid` keeps the parent document available for Cypher queries.
    tx.run(
        """
        MERGE (p:Paragraph {para_uuid: $para_uuid})
        ON CREATE SET
            p.text       = $text,
            p.index      = $idx,
            p.doc_uuid   = $doc_uuid,
            p.expiration = 0
        """,
        para_uuid=para_uuid,
        text=text,
        idx=idx,
        doc_uuid=doc_uuid,
    )

    # Link each paragraph back to its document so query results can include the
    # document title and category alongside the retrieved paragraph text.
    tx.run(
        """
        MATCH (p:Paragraph {para_uuid: $para_uuid}),
              (d:Document  {doc_uuid: $doc_uuid})
        MERGE (p)-[r:PART_OF]->(d)
        ON CREATE SET r.expiration = 0
        """,
        para_uuid=para_uuid,
        doc_uuid=doc_uuid,
    )

def link_mentions(tx, ent_uuid: str, doc_uuid: str, para_uuid: str) -> None:
    """
    Connect an Entity to both the specific paragraph it appears in and the
    enclosing document.  Two edges make downstream reasoning flexible.
    """
    # Paragraph-level mention: this is the primary retrieval edge used by
    # `query.py` when matching question entities to candidate context.
    tx.run(
        """
        MATCH (e:Entity {ent_uuid: $ent_uuid}),
              (p:Paragraph {para_uuid: $para_uuid})
        MERGE (e)-[m:MENTIONS]->(p)
        ON CREATE SET m.expiration = 0
        """,
        ent_uuid=ent_uuid,
        para_uuid=para_uuid,
    )
    # Document-level mention: useful for document-scoped traversal or future
    # summaries, even though the current query path ranks paragraph hits.
    tx.run(
        """
        MATCH (e:Entity {ent_uuid: $ent_uuid}),
              (d:Document {doc_uuid: $doc_uuid})
        MERGE (e)-[m:MENTIONS]->(d)
        ON CREATE SET m.expiration = 0
        """,
        ent_uuid=ent_uuid,
        doc_uuid=doc_uuid,
    )

###############################################################################
# Ingest logic
###############################################################################

def ingest_file(session, category: str, path: Path) -> None:
    """
    Parse one text file and write its document, paragraph, and entity graph.

    File convention:
    * First line      → document title
    * Remaining text  → body
    * Blank lines     → paragraph boundaries

    The NER service is called per paragraph so entity relationships point at
    the smallest context unit this demo retrieves.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    title, body = lines[0], "\n".join(lines[1:])
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    doc_uuid = str(uuid.uuid4())

    print(f"\u27A4  {title}  [{category}]")

    # Store the full body once at the document level for provenance and display.
    session.execute_write(create_document, doc_uuid, title, body, category)

    # Create retrieval paragraphs, then attach every allowed entity extracted
    # from that paragraph. The `promote` flag is accepted by the shared client
    # for compatibility with other demos; this local NER service ignores it.
    for idx, text in enumerate(paragraphs):
        para_uuid = str(uuid.uuid4())
        session.execute_write(create_paragraph, para_uuid, text, idx, doc_uuid)

        response = call_ner_service(
            text,
            promote=False,
            labels=sorted(ALLOWED_LABELS) if ALLOWED_LABELS else None,
        )
        entity_pairs = parse_entity_pairs(response)
        if ALLOWED_LABELS:
            entity_pairs = [
                (name, label)
                for name, label in entity_pairs
                if label.upper() in ALLOWED_LABELS
            ]

        for name, label in entity_pairs:
            ent_uuid = session.execute_write(
                merge_entity, str(uuid.uuid4()), name, label
            )
            session.execute_write(link_mentions, ent_uuid, doc_uuid, para_uuid)

def main() -> None:
    print("Preparing graph ingest. Ensure the NER service is already running …")

    # One driver + one session keeps the batch ingest intentionally small.
    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as driver, driver.session() as session:

        # Start from a clean slate so reruns are deterministic and stale
        # entities/relationships from prior lab runs cannot affect retrieval.
        print("Clearing old data from the database …")
        session.run("MATCH (n) DETACH DELETE n")
        print("Database cleared.")

        # Ensure indexes exist before the heavy writes start
        print("Creating/validating indexes …")
        create_indexes(session)
        print("Indexes ONLINE.\n")

        # Each first-level directory is treated as a category label from the
        # BBC dataset layout, for example `tech` or another corpus section.
        for category in sorted(os.listdir(DATASET_PATH)):
            category_path = Path(DATASET_PATH) / category
            if not category_path.is_dir():
                continue

            print(f"\n=== Category: {category} ===")
            for txt in sorted(category_path.glob("*.txt")):
                ingest_file(session, category, txt)

    # Recap which entity labels were allowed into the graph.
    if ALLOWED_LABELS:
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        print(f"\nFinished. Ingest restricted to entity types: {allowed}")
    else:
        print("\nFinished. All entity types ingested.")

if __name__ == "__main__":
    main()
