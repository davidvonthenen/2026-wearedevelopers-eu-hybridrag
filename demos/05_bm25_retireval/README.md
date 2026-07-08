# Section 2 Part 2: BM25 Retrieval using the Recipe Dataset

This project demonstrates a BM25-based retrieval pipeline over a recipe dataset. It uses OpenSearch for lexical search, a small spaCy-based NER service for optional entity extraction, and structured OpenSearch keyword fields for deterministic filters such as allergy cautions and health labels.

It is focused on observing how lexical retrieval behaves, especially when a natural-language query contains exclusions such as "without soba" or "doesn’t contain Sulfites."

## What the code does

There are two primary scripts:

- `ingest.py` loads recipe rows from the CSV dataset, fetches the recipe web pages, extracts human-readable recipe text, prepends recipe metadata, chunks the result, calls the NER service for each chunk, and indexes BM25-searchable documents into OpenSearch.
- `query.py` sends a query to OpenSearch using BM25 full-text search

## Prerequisites

- Python 3.10 **ONLY**
- OpenSearch 3.5 running locally with security disabled
- TODO: DATASET TODO

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

# Start OpenSearch.
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

# Start OpenSearch Dashboards.
podman run -d \
    --name "opensearch-single-dashboards" \
    --network "opensearch-net" \
    -p 5600:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-single:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    opensearchproject/opensearch-dashboards:3.5.0
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

The service listens on `http://127.0.0.1:8000/ner` by default. `ingest.py` uses it to attach extracted entity strings to indexed chunks. `query.py` also calls it for the query text before executing the BM25 search helper.

### Ingest the Recipe Dataset 

<span style="color: cyan;">**IMPORTANT NOTE:** This is a new step. Even through we have already used OpenSearch for BM25 retrieval in Section 1 Step 2, this ingests the RECIPE dataset using BM25 retrieval.</span>

We need to ingest the dataset to OpenSearch. This Recipe dataset is a collection of simple recipes capturing instructions, ingredients, allergens, etc.

We are saving our **BM25 chunks** to the OpenSearch index named `recipes-bm25`.

```bash
python ingest.py
```

The ingest process writes documents to the OpenSearch index named `recipes-bm25` by default. Each document represents one chunk of recipe text and includes:

- BM25-searchable text in the `text` field
- Recipe metadata such as name, source, URL, servings, calories, cuisine type, meal type, and dish type
- We normalized some fields such as `cautions`
- Original display values for labels where useful
- NER output in the flat `named_entities` keyword field
- Chunk provenance fields such as `chunk_id`, `chunk_index`, `chunk_number`, and `chunk_count`

For demonstration purposes, documents are chunked at 2048 characters with a 256-character overlap unless you override the CLI options.

## Step 2: Data Retrieval Using BM25

This demonstrates BM25 data retrieval using the recipe dataset. We are going to query for simple "noodle" recipes.

```bash
python query.py --query "I am looking for noodle recipes."
```

This query behaves exactly like you would expect. This mimics the data returned from using `vector embeddings` (ie semantic similarity search).

```bash
python query.py --query "I am looking for noodles recipes without soba."
```

If we treat the "negated search" case the same with `BM25` or `FullText Search` as `vector embeddings`, the results are going to be the same. The result of this prompt returns both "noodle" and "soba" recipes despite that being the exact opposite of what we want.

```bash
python query.py --query "I am looking for noodles recipes without soba that doesn't contain Sulfites."
```

Compounding the "negated search" case with multiple fields yields zero results. If you remember in the `vector embedding` result, we actually got results that only contained `Sulfites`. At least in the `BM25` case, zero results is better than a confidently incorrect result. In real world scenarios, this kind of error can kill someone with a peanut allgery.

```bash
python query.py --query "I am looking for noodles recipes without soba." --exclude-caution "Sulfites"
```

Now if we go through an apply some structure to our data before processing, we can actually pull this type of data retrieval off. If we extract "allegens" and create `tags` or `key/value pairs` for entries in our `BM25` data, we can get the desired effect "without Sulfites". This does require that we have special logic to understand this "negated context" and explicitly use the `--exclude-caution` filter. The million dollar question is... How do you strategically make this jump? This can be done, but now requires you understand more of the context of the users prompt/question.

**HOWEVER**, notice that the top recipes returned are those containing "noodle" and "soba". This is not ideal for the exact same reasons. You would need to add even more structure (via tags or ke/value pairs) to your "unstructured data".

## Clean Up

> **IMPORTANT NOTE:** Stop the **NER Service** in this step.

## Takeways

Lexical search via algorithms like BM25 represents the bedrock of text retrieval. While vector search attempts to map abstract semantic concepts in high-dimensional space, BM25 operates on the ground floor of absolute literalism: it counts tokens, evaluates term frequency, and balances inverse document frequency. Transitioning from dense embeddings to keyword-matching engines fundamentally transforms how negation queries fail, trading the fluid hallucinations of vector space for the uncompromising rigidity of exact token matching.

## The Mechanics of Lexical Collapse

Analyzing how BM25 processes conversational modifiers reveals a stark contrast to vector-based systems. The failure modes here are deterministic rather than geometric.

- The Tokenization Paradox: BM25 handles text strings as bags of words. When evaluating the phrase "recipes without soba", the engine indexes "recipes", "without", and "soba" as individual, positive search terms. Because the algorithm lacks native grammatical awareness, "without" cannot modify or negate "soba". Instead, the engine treats "soba" as a highly relevant, low-frequency keyword, actively surfacing documents that prominently feature the exact ingredient the user intended to avoid.

- Fail-Safe vs. Toxic Hallucination: A critical architectural distinction emerges during compound negation ("without soba that doesn't contain Sulfites"). Where vector search drifts into a confidently incorrect state—returning matching profiles because the semantic "vibe" aligns—BM25 searches for all literal tokens and ultimately returns zero results. In high-stakes domains like health, medicine, or allergen management, returning an empty result set is infinitely superior to serving a lethal ingredient under the guise of relevance.

- The Structured Solution: Introducing a dedicated attribute flag like --exclude-caution "Sulfites" outlines the correct engineering trajectory. By intercepting high-risk variables before query execution and routing them into categorical key-value metadata tags, the retrieval pipeline enforces strict compliance.

However, notice how the remaining text string "without soba" continues to pollute the BM25 core. This confirms that relying on unstructured text search for exclusion logic is fundamentally flawed unless the entire query undergoes explicit syntax translation.

### Architectural Reflection and Call to Action

Building discovery systems for complex domains frequently exposes a common developer pitfall: assuming that legacy text engines possess an inherent understanding of human conversational context. They do not. They match characters. Watching a platform fail to parse basic user constraints because the input interface mimics a conversational chat window serves as a stark reminder that interface design must align with backend indexing capabilities.

To resolve this bottleneck, engineers must decouple conversational intent from raw backend queries:

- Implement Query Understanding (QU): Deploy an aggressive classification layer upstream to parse unstructured text inputs into strict Abstract Syntax Trees (ASTs).

- Translate to Boolean Logics: Programmatically map terms following conversational negations directly into hard database exclusions (e.g., Lucene's MUST_NOT clauses).

Stop forcing Language Models to guess intent; map conversational exclusions to deterministic index filters.
