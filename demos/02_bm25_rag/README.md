# Section 1 Part 2: Local BM25 RAG Agent

This project provides an end-to-end Retrieval Augmented Generation (RAG) workflow using **BM25-style lexical retrieval** with OpenSearch and a local GGUF model for generation. OpenSearch stores both full BBC-style documents and fixed-size overlapping chunks. A small Flask service provides Named Entity Recognition (NER) using spaCy, and `query.py` loads the local model directly with `llama.cpp` for inference.

## Prerequisites

- Python 3.10 only
- OpenSearch 3.5 running locally with security disabled
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

Install all the Python libraries we will be using in this section by running the following command:

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

If you don't have OpenSearch running (you can check by running: `podman ps`). You can run the following command:

```bash
# create directories
mkdir -p \
    "$HOME/models" \
    "$HOME/opensearch/data" \
    "$HOME/opensearch/snapshots"

# create the network for opensearch
podman network create opensearch-net

# deploy the container images
podman run -d \
    --name "opensearch-single" \
    --network "opensearch-net" \
    -p 9200:9200 -p 9600:9600 \
    -e "discovery.type=single-node" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -e "cluster.routing.allocation.disk.threshold_enabled=false" \
    --userns="keep-id" \
    -v "$HOME/opensearch/data:/usr/share/opensearch/data" \
    -v "$HOME/opensearch/snapshots:/mnt/snapshots" \
    docker.io/opensearchproject/opensearch:3.5.0

podman run -d \
    --name "opensearch-single-dashboards" \
    --network "opensearch-net" \
    -p 5601:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-single:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    --userns="keep-id" \
    docker.io/opensearchproject/opensearch-dashboards:3.5.0
```

## Step 1: Prerequisite Setup

### Start Our Named Entity Reognition Service

<span style="color: cyan;">**IN A NEW TERMINAL:** We need to start our NER service. Run the following commands FROM THIS DIRECTORY and don't forget to start your virtual environment if you are using one:</span>

```bash
# For miniconda or venv:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
# OR, if using uv:
uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# start the service
python ner_service.py
```

The NER service exposes:

- `POST /ner`, which accepts text and returns normalized entity names plus `{name, label}` pairs
- A default endpoint URL of `http://127.0.0.1:8000/ner`

Both `ingest.py` and `query.py` call this service, so keep it running while ingesting and querying.

### Ingest the British Broadcasting Corporation (BBC) Dataset 

<span style="color: cyan;">**IMPORTANT NOTE:** This is a new step. Even through we have already used OpenSearch for vector retrieval, this ingests the BBC dataset using BM25 retrieval.</span>

We need to ingest the dataset to OpenSearch. This BBC dataset is a collection of simple Technology related news articles.

We are saving our **BM25 / full-text search** to the OpenSearch index named `bbc-bm25-chunks`.

```bash
python ingest.py
```

For demonstration purposes, documents are chunked into 2048-character windows with a 256-character overlap. Chunking keeps the retrieval unit small enough for prompt construction while still preserving nearby context, because apparently documents enjoy being longer than context windows.

## Step 2: Introduction to Named Entity Recognition

Named entity recognition (NER) is a component of natural language processing (NLP) that identifies predefined categories of objects in a body of text.

This process, also known as entity chunking or extraction, serves as a sub-task of information extraction and can be implemented using statistical models, rule-based systems, or custom-trained models for specialized use cases.

NER extracts and classifies named entities like names, locations, organizations, etc turning unstructured text into structured information.

To understand what NER does for us, let's run a couple of examples below:

```bash
# example 1
python ner_client.py --text "My name is David and I am from Long Beach, California."
```

```bash
# example 2
python ner_client.py --text "Microsoft has its headquarters in Redmond, Washington."
```

```bash
# example 3
python ner_client.py --text "The Eiffel Tower is located in Paris."
```

## Step 3: Query the RAG Pipeline

For our retrieval pipeline, `query.py` performs the following flow:

1. The project already provides defaults for all parameters to run.
2. Loads the local GGUF model with `llama_cpp.Llama`.
3. Sends the user question to the NER service.
4. Builds an OpenSearch query from the extracted entities. If no entities are found, it falls back to a normal `match` query over the chunk content.
5. Searches the `bbc-bm25-chunks` index by default.
6. Re-ranks the OpenSearch candidate chunks with the local `bm25s` library when external ranking is enabled, which is the default path in `ask()`.
7. Builds a context block from the selected chunks and asks the local model to answer using only that context.

This is BM25-style lexical retrieval, not vector retrieval. The NER step improves exact-name matching by storing entity terms as structured fields during ingest and extracting comparable entity terms from the question during retrieval.

We are going to run a few queries for inference using OpenSearch

```bash
python query.py --question "How much did OpenAI purchase Windsurf for?"
```

Note the results of this question.

Now, run the next query below:

```bash
python query.py --question "How much did Google purchase Windsurf for?"
```

Note the results of this question. Notice that if you didn't get an answer in the Step 1 using vector embeddings, you did get an answer here. The reason why is that Named Entity Recognition (NER) is a deterministic retrieval mechanism.

## Clean Up

> **IMPORTANT NOTE:** Stop the **NER Service** in this step.

## Takeways

Using the same news dataset, searching for data is easy... but just like the previous **vector embedding** example, **BM25** suffers from the same data reconcillation problem. It's not able to reconcile events over time and contradictory information.

- BM25/full-text retrieval works well when important terms appear directly in the source text.
- Named Entity Recognition helps make retrieval more precise by extracting names, organizations, places, products, events, and similar terms into structured OpenSearch fields.
- The project stores both full documents and chunks, but the query path uses the chunk index by default.
