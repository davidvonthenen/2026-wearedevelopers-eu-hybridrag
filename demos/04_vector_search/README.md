# Section 2 Part 1: Vector Retrieval using the Recipe Dataset

This project demonstrates a small vector-based retrieval pipeline for recipe data. The ingest path reads recipe metadata from a CSV file, fetches readable recipe page text from each recipe URL, chunks that text, embeds the chunks, and stores them in an OpenSearch `knn_vector` index. The query path embeds a user question and runs a pure vector k-NN search against that index.

No LLM is involved in this section. The point is to observe what dense vector retrieval does well, where it becomes fuzzy, and why negation is not the same thing as filtering. Apparently geometry did not volunteer to become a Boolean logic engine. Rude, but mathematically consistent.

## Prerequisites

- Python 3.10 **ONLY**
- OpenSearch 3.5 running locally with security disabled
- [recipes-with-nutrition on Huggingface](https://huggingface.co/datasets/datahiveai/recipes-with-nutrition). This already exists within this directory.

## Installation

<span style="color: red;">**IMPORTANT NOTE:** If you are running this hands-on lab in sequence, you do not need to perform these **Installation Step** again.</span>

**If** you haven't created a [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [venv](https://docs.python.org/3/library/venv.html), or [uv](https://pydevtools.com/handbook/how-to/how-to-install-uv/) virtual environment yet, please do so now. This single environment can be used throughout this entire lab exercise. Call this environment `2026-wearedevs-eu-workshop`.

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
# create the network for opensearch
podman network create opensearch-net

# deploy the OpenSearch container
podman run -d \
    --name "opensearch-single" \
    --network "opensearch-net" \
    -p 9200:9200 -p 9600:9600 \
    -e "discovery.type=single-node" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -e "cluster.routing.allocation.disk.threshold_enabled=false" \
    -v "$HOME/opensearch/data:/usr/share/opensearch/data" \
    -v "$HOME/opensearch/snapshots:/mnt/snapshots" \
    opensearchproject/opensearch:3.5.0

# deploy OpenSearch Dashboards
podman run -d \
    --name "opensearch-single-dashboards" \
    --network "opensearch-net" \
    -p 5601:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-single:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.5.0
```

## Step 1: Prerequisite Setup

We need to ingest the dataset to OpenSearch. This Recipe dataset is a collection of simple recipes capturing instructions, ingredients, allergens, etc.

We are saving our **vector embeddings** to the OpenSearch index named `recipes-vector`.

We are using the [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) as our embedding model.

```bash
python ingest.py
```

Embeddings are L2-normalized before indexing and stored in an OpenSearch `knn_vector` field named `embedding`. The index uses HNSW with cosine similarity.

For demonstration purposes, the script targets 2048-character chunks with 256 characters of overlap. Each chunk repeats a compact recipe metadata header, so the final stored chunk text may be slightly larger than the body chunk size.

## Step 2: Data Retrieval Using Vector Embeddings

The query script embeds the user query and sends it directly to OpenSearch as a k-NN vector search. This script does not perform hybrid retrieval, graph lookup, Boolean filtering, or LLM generation.

We are going to query for simple "noodle" recipes.

```bash
python query.py --query "I am looking for noodle recipes."
```

This should return recipes that are semantically close to the query. Results may contain exact noodle terms, related ingredients, or similar recipe concepts depending on the embedding space and the indexed recipe text.

Now try a negated request:

```bash
python query.py --query "I am looking for noodles recipes without soba."
```

This demonstrates a common dense-vector retrieval limitation: the embedding still contains a strong semantic signal for both `noodles` and `soba`. Since this script performs pure vector similarity, it does not treat `without soba` as a hard exclusion.

Try adding another exclusion:

```bash
python query.py --query "I am looking for noodles recipes without soba that doesn't contain Sulfites."
```

This often compounds the problem. The query vector can be pulled toward the very concepts the user tried to exclude because terms like `soba` and `Sulfites` remain semantically important tokens in the query text.

## Takeways

Vector search and semantic similarity have fundamentally transformed how unstructured data is queried, turning raw text into high-dimensional geometric coordinates that map conceptual meaning. However, this architectural elegance hits a fascinating, mathematically predictable wall when encountering conversational nuances like negation. While dense embeddings excel at identifying what an intent includes, they notoriously flounder when explicitly instructed on what a user wishes to avoid.

### The Mechanics of the Collapse: Why Negation Fails

To dissect this failure mode, one must look at how embedding models process tokens contextually. When a sequence is passed through an embedding transformer, the entire phrase is compressed into a singular dense vector.

- The Suffix Weight Trap: In tokenization and high-dimensional space allocation, nouns like "soba" or "Sulfites" possess massive semantic weight.

- Proximity Over Logic: When a user requests "noodle recipes without soba," the model yields a vector that sits tightly clustered within both the "noodle" and "soba" neighborhoods. Because "soba" and "noodle" share immense contextual overlap in the training corpus, mathematical operations calculating distance—such as cosine similarity or dot product—prioritize this semantic proximity over the logical modifier "without." The embedding space lacks a native representation for a logical NOT.

- The Compounding Failure Mode: As demonstrated by the final query ("without soba that doesn't contain Sulfites"), adding multiple layers of exclusion severely exacerbates this mathematical drift. The query vector is pulled simultaneously toward the "soba" cluster and the "Sulfites" cluster. Instead of filtering these elements out, the system treats them as primary semantic hooks, optimizing the results for the exact constraints the user tried to banish.

### Architectural Reflection and Call to Action

In engineering large-scale conversational **answer** platforms, watching a team deploy pure vector search only to watch it buckle under a deluge of user-specified exclusions is an absolute rite of passage. It illustrates a stark reality: human linguistic logic does not map cleanly to pure vector geometry. Relying strictly on dense retrieval for constraint-driven applications is an anti-pattern for production environments.

The lie you have been told is that vector embeddings solves all problems and you can work with unstructed data. In fact, the **ONLY** way to build meaningful AI solutions is either:

- putting additional layers of linguistic nuace into the search layers, or
- structure your unstructured data (or better yet, don't start with unstructured data)

> **IMPORTANT NOTE:** Many vector databases have solutions for negation by either: 1) key/value pair exculsion, OR 2) post-filtering exclusion. Those mechanisms, although they do work, tend not to scale for large datasets. Will discuss this in the next few sections.
