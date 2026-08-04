# WeAreDevelopers Europe 2026 - (WORKSHOP) From Vector Search to Better Understanding: How Hybrid RAG Improves Answers, Not Just Matches

Welcome to the landing page for the workshop `From Vector Search to Better Understanding: How Hybrid RAG Improves Answers, Not Just Matches` at `WeAreDevelopers Europe 2026`.

## What to Expect

This repo intends to provide slides, instructions and materials for our workshop. Instruction will be provided (1hr to 1h2 20min) and tasks are self-paced (about 30-40mins with help from me should you need it). This workshop is broken down into 3 majors sections:

1. Introduction to Alternative Data Retrieval Methods for RAG
2. Highlight RAG Challenges When Answering User's Questions
3. How Hybrid RAG Architectures Provide Answers, Not Just Semantically Similar Text Parading as Answers

## Workshop Prerequisites

Participants should ensure they have the minimum requirements:

- A Linux or Mac Developer’s Laptop with enough memory (16GB minimum) to run 2 databases containers plus a [Quantized 7B Small Language Model](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF).
  - No GPU is required. Will strictly be using CPU only.
  - Windows Users should use a VM or Cloud Instance
    - If you opt for this, you must provide your own instances
- Python Version: **Using Only 3.10** 
  - **Sections of the workshop will not function using a version other than 3.10**
  - (HIGHLY Recommended) Use [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [uv](https://docs.astral.sh/uv/getting-started/installation/), or [venv](https://docs.python.org/3/library/venv.html) virtual development environment. (I prefer miniconda and will provide instructions usig it)
- (HIGHLY Recommended) [Huggingface developer token](https://huggingface.co/settings/tokens) saved to `HF_TOKEN` environment variable.
- Install [Podman](https://podman.io/docs/installation) (or equivalent container runtime).
  - Will use [OpenSearch](https://opensearch.org/) and [Neo4j](https://neo4j.com/) containers.

Software Downloads:
- [Qwen2.5-7B-Instruct-1M-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF)
  – Specifically the [Q5_K_M version](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf)
  - Drop the file into your ~/models folder. (You might need to create this.)
- Pre-pull the following containers:
  - `podman pull docker.io/opensearchproject/opensearch:3.5.0`
  - `podman pull docker.io/opensearchproject/opensearch-dashboards:3.5.0`
  - `podman pull docker.io/neo4j:5.26.16`

## Python Software Prerequisities

This workshop will **only work with Python version 3.10**. I would highly recommend using [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [uv](https://docs.astral.sh/uv/getting-started/installation/), or [venv](https://docs.python.org/3/library/venv.html) so you can run **Python 3.10** in its own isolated environment.

**If** you have never used one of these tools before, [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [venv](https://docs.python.org/3/library/venv.html) is the absolutely easiest and I will provide instructions using it going forward. If you prefer another tool, then use what you are familiar with.

We will use `2026-wearedevs-eu-workshop` as the environment for our workshop today.

```bash
conda create -n 2026-wearedevs-eu-workshop python=3.10
```

Please activate this environment:

```bash
conda activate 2026-wearedevs-eu-workshop
```

To install the software dependencies:

```bash
pip install -r requirements.txt
```

Then we need to install the model for spacey:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
```

Done!

## Starting OpenSearch and Neo4j Containers

If you haven't installed [Podman](https://podman.io/docs/installation) (or you can use whatever container runtime you are familiar with, like Docker), you can find instructions for install [here](https://podman.io/docs/installation).

You need to start your `podman` VM instance to host containers (if you haven't done so already).

```bash
podman machine start
```

Download the container images:

```bash
podman pull docker.io/opensearchproject/opensearch:3.5.0
podman pull docker.io/opensearchproject/opensearch-dashboards:3.5.0
podman pull docker.io/neo4j:5.26.16
```

Start the OpenSearch by running the following command:

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

Start Neo4j by running the following command:

```bash
# create directories
mkdir -p \
    "$HOME/neo4j/data" \
    "$HOME/neo4j/import" \
    "$HOME/neo4j/plugins"

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
    --userns="keep-id" \
    -v "$HOME/neo4j/data:/data" \
    -v "$HOME/neo4j/plugins:/plugins" \
    -v "$HOME/neo4j/import:/var/lib/neo4j/import" \
    docker.io/neo4j:5.26.16
```

## Workshop Materials

The workshop will consist of 4 sections. Each section will have a series of hands-on tasks for you to complete.

- Section 1: The "R" In RAG: Use Cases & Methods
  - Part 1: [Simple Vector-based Agent](./demos/01_vector_rag/README.md)
  - Part 2: [Simple BM25-based Agent](./demos/02_bm25_rag/README.md)
  - Part 3: [Simple Graph-based Agent](./demos/03_graph_rag/README.md)

- Section 2: Data Retrieval Challenges
  - Part 1: [Exploring Vector Retrieval](./demos/04_vector_search/README.md)
  - Part 2: [Exploring BM25 Retrieval](./demos/05_bm25_retireval/README.md)
  - Part 3: [Exploring Graph Retrieval](./demos/06_graph_retrieval/README.md)

- Section 3: What Is Hybrid RAG?
  - Part 1: [How Data Feeds Inference](./demos/07_hybrid_bm25_vector_instaclustr/README.md)
  - Part 2: [Decouple Data & Inference](./demos/08_hybrid_graph_vector_retrieval/README.md)

- Section 4: Fact-based Answers Wrapped in AI Governance
  - Part 1: [Answers with Governance](./demos/09_hybrid_rag_graph_vector/README.md)
  - Part 2: [The "Crazy" Architecture](./demos/10_hybrid_rag_graph_vector_instaclustr/README.md)
