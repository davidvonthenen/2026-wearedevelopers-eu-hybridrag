# Section 1 Part 3: Local Graph-based RAG Agent

This project provides a compact Graph-based Retrieval Augmented Generation (RAG) workflow. It uses Neo4j as the knowledge graph store, a small Flask-based Named Entity Recognition (NER) service for entity extraction, a BM25 reranker for retrieved graph candidates, and a local GGUF model through `llama-cpp-python` for answer generation.

The retrieval path is intentionally graph-first: documents are ingested into Neo4j as `Document`, `Paragraph`, and `Entity` nodes. At query time, the user question is sent to the NER service, matching entities are used to retrieve related paragraphs from Neo4j, and BM25 reranks those paragraph candidates before the local model receives the final context.

## Prerequisites

- Python 3.10 only
- Neo4j 5.26.16 running locally
- A **modified** BBC dataset located at `./bbc` (original version from [derekgreene/bbc-datasets](https://github.com/derekgreene/bbc-datasets))
- Small Language Model: [Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf) saved to this path: `~/models`

## Installation

<span style="color: red;">**IMPORTANT NOTE:** If you are running this hands-on lab in sequence, you do not need to perform these **Installation Step** again.</span>

**If** you haven't created a [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [venv](https://docs.python.org/3/library/venv.html), or [uv](https://pydevtools.com/handbook/how-to/how-to-install-uv/) virtual environment yet, please do so now. This single environment can be used throughout this entire lab exercise. Call this environment `2026-wearedevs-eu-workshop`.

```bash
conda create -n 2026-wearedevs-eu-workshop python=3.10
```

Please activate this environment:

```bash
conda activate 2026-wearedevs-eu-workshop
```

To install the software dependencies:

```bash
# if you are using: miniconda or venv
pip install -r requirements.txt

# OR, if using  uv
uv pip install -r requirements.txt 
```

**If** you need to start your `podman` VM instance to host containers, run the following command:

```bash
podman machine start
```

<span style="color: cyan;">**IMPORTANT NOTE:** This is a NEW step. We need to download and run our Neo4j database.</span>

If you don't have Neo4j running (you can check by running: `podman ps`). You can run the following command:

```bash
# create the network for neo4j
podman network create graph-net

# deploy the container images
podman run -d \
    --name "neo4j-single" \
    --network "graph-net" \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/neo4jneo4j \
    -e NEO4JLABS_PLUGINS='["apoc"]' \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -v "$HOME/neo4j/data:/data" \
    -v "$HOME/neo4j/plugins:/plugins" \
    -v "$HOME/neo4j/import:/var/lib/neo4j/import" \
    neo4j:5.26.16
```

## Step 1: Prerequisite Setup

### Start Our Named Entity Reognition Service

<span style="color: cyan;">**IN A NEW TERMINAL:** We need to start our NER service. Run the following commands FROM THIS DIRECTORY and don't forget to start your virtual environment if you are using one:</span>

```bash
# If you are using the same virtual environment: 2026-wearedevs-eu-workshop, you don't need to run the command below:
# For miniconda or venv:
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
# OR, if using uv:
# uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# BUT YOU DO NEED TO START THE NER SERVICE USING THE COMMAND BELOW
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
