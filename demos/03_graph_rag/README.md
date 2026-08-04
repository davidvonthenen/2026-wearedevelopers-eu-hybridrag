# Section 1 Part 3: Local Graph-based RAG Agent

This project provides a compact Graph-based Retrieval Augmented Generation (RAG) workflow. It uses Neo4j as the knowledge graph store, a small Flask-based Named Entity Recognition (NER) service for entity extraction, a BM25 reranker for retrieved graph candidates, and a local GGUF model through `llama-cpp-python` for answer generation.

The retrieval path is intentionally graph-first: documents are ingested into Neo4j as `Document`, `Paragraph`, and `Entity` nodes. At query time, the user question is sent to the NER service, matching entities are used to retrieve related paragraphs from Neo4j, and BM25 reranks those paragraph candidates before the local model receives the final context.

## Installation Prerequisite Software

Please see the README.md at the root of the repo.

## Step 1: Prerequisite Setup

### Start Our Named Entity Reognition Service

<span style="color: cyan;">**IN A NEW TERMINAL:** We need to start our NER service. Run the following commands FROM THIS DIRECTORY and don't forget to start your virtual environment if you are using one:</span>

```bash
python ner_service.py
```

The service listens on `http://127.0.0.1:8000/ner`. It extracts named entities with spaCy and returns both a deduplicated entity-name list and detailed `(name, label)` pairs. It does not write to Neo4j; the ingest and query scripts consume the service response and perform graph writes or reads themselves.

### Ingest the British Broadcasting Corporation (BBC) Dataset 

<span style="color: cyan;">**IMPORTANT NOTE:** This is a new step. We need to create Knowledge Graphs using the BBC dataset.</span>

We need to ingest the dataset to Neo4j. This BBC dataset is a collection of simple Technology related news articles.

```bash
python ingest.py
```

The ingest script expects files under `./bbc/<category>/*.txt`. For each file, it treats the first line as the document title and the remaining text as the body. The body is split on blank lines into paragraphs. Each document, paragraph, and detected entity is written to Neo4j, with `MENTIONS` relationships connecting entities to both paragraphs and documents.

This project does **not** use OpenSearch, vector embeddings, or fixed-size 2048/256 character chunking. Paragraphs are the retrieval units, and BM25 is used only as a reranker over graph-retrieved candidates.

## Step 2: Query the RAG Pipeline

The query script performs this flow:

1. The project already provides defaults for all parameters to run.
2. Send the question to the NER service.
3. Use the extracted `(name, label)` pairs to retrieve candidate paragraphs from Neo4j.
4. Rerank those candidates with BM25.
5. Build a compact context block from the highest-ranked paragraphs.
6. Ask the local GGUF model to answer using only the retrieved context.

Run a question through the graph retrieval and local model pipeline:

```bash
python query.py --question "How much did OpenAI purchase Windsurf for?"
```

Note the results of this question.

Try the comparison query:

```bash
python query.py --question "How much did Google purchase Windsurf for?"
```

Note the results of this question.

## Clean Up

> **IMPORTANT NOTE:** Stop the **NER Service** in this step.

## Takeways

A graph-based RAG pipeline can provide useful retrieval without a vector database when the query contains recognizable entities that map cleanly into the corpus. **HOWEVER**, naively using Knowledge Graphs can yield the same exact results as in the **vector embedding** and **BM25** examples despite having relationships between data points.

The big question now is... How do we address this issue? We will discuss this in Section 2.
