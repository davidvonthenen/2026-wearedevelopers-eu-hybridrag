# Section 4 Part 1: Complete RAG Agent Using the Recipe Dataset

This project demonstrates a chatbot-style Hybrid RAG pipeline for recipe question answering. It uses Neo4j as the graph-backed truth channel for structured recipe facts, labels, cautions, nutrition, source, and URL metadata, and OpenSearch as the vector-backed semantic channel for recipe wording, ingredients, instructions, and fuzzy matching.

The retrieval pattern builds on the earlier Graph + Vector recipe retrieval exercise, but this section focuses on turning that retrieval flow into an interactive chatbot. The important behavior to notice is the division of responsibility: graph retrieval is used for deterministic constraints, while vector retrieval adds semantic detail for more natural answers.

The projects' main entry point is `query.py`, which runs the chatbot-style application. It can run as an interactive CLI or as an OpenAI-compatible REST service.

The data retrieval mechanism is the same as `Section 3, Step 2: Breaking Down Hybrid (Graph + Vector) RAG using the Recipe Dataset`. We have already discussed via the workshop presentation and through the various materials in each section, what the data retrieval looks like; therefore, `Section 4` will only focus on inference.

## Prerequisites

- Python 3.10 **ONLY**
- Neo4j 5.26.16 running locally
- OpenSearch 3.5 running locally
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

If Neo4j and OpenSearch are not running, you can start local containers with the following commands. The Neo4j password shown here matches the default password in `common/config.py`; if you use a different password, set `NEO4J_PASSWORD` before running the scripts.

```bash
# create the network for Neo4j
podman network create graph-net

# deploy Neo4j
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
    neo4j:5.26.16

# create the network for OpenSearch
podman network create opensearch-net

# deploy OpenSearch
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
    opensearchproject/opensearch:3.5.0

podman run -d \
    --name "opensearch-single-dashboards" \
    --network "opensearch-net" \
    -p 5600:5601 \
    -e 'OPENSEARCH_HOSTS=["http://opensearch-single:9200"]' \
    -e 'DISABLE_SECURITY_DASHBOARDS_PLUGIN=true' \
    --userns="keep-id" \
    opensearchproject/opensearch-dashboards:3.5.0
```

## Step 1: Prerequisite Setup

### Ingest the Recipe Dataset 

<span style="color: red;">**IMPORTANT NOTE:** If you are running this hands-on lab in sequence, you do not need to perform data ingesting again.</span>

Since we are reusing the same recipe dataset and we already performed the ingest to following Neo4j and OpenSearch container, we can just move onto step 2:

- `neo4j database`
- `recipes-vector`

**If** you are starting over (like you destroyed your Neo4j and OpenSearch container), you can go back to the following folders to run `python ingest.py`. **Again, do not to this unless you are absolutely sure you need to!**

- `05_bm25_retireval`
- `06_graph_retrieval`

## Step 2: Our Recipe Chatbot

We will make use of Neo4j for truth grounding and vector embeddings for contextual detail.

### Start Our Local LLM

<span style="color: cyan;">**IN A NEW TERMINAL:** We need to start our LLM service. Run the following commands FROM THIS DIRECTORY and don't forget to start your virtual environment if you are using one:</span>

```bash
python llm_service.py
```

### Let's Run the Chatbot

**BUT BEFORE BE BEGIN,** one thing we haven't address is how to handle the "negation" of ingredients or allergens (example: without "soba" noodles). This is needed in order to map search terms (or things we are interested in) to `--query` and ingredients we aren't intersted in to `--notquery` (or the negation). We also need a mechanism to track the removal of allergens like "Sulfites" to pass into `--exclude-caution`.

Since a peanut allergy (for example) can actually kill someone, this needs to be a deterministic mechanism. We are going to overload or add functionality to our `Named Entity Recognition` to understand negation. We are going to do this by using a project called [negspacy](https://github.com/jenojp/negspacy), or a "spaCy pipeline object for negating concepts in text".

To demonstrate an example of `negspacy`'s behavior, run the following command:

```bash
python mynegspacy.py 
```

You can see based on the results how we can map `attribution` or `negation` of ingredients or things in sentences. In each sentence in the prompt, we are going to run it through this NER classifier that now provides us with `attribution` or `negation`.

If a sentence is an ingredient we are looking for (`negation` is `False`), then we map the ingredient to `--query`.
If a sentence is an ingredient we want to avoid for (`negation` is `True`), then we map the ingredient to `--notquery`.

At a high level, `query.py` does the following:

- Loads `en_core_web_sm` and a custom food NER model from `model/`.
- Adds the custom food NER component into the spaCy pipeline.
- Uses `negspacy` to mark negated `FOOD` entities.
- Maps negated allergy-context entities to `--exclude-caution` behavior.
- Maps other negated food entities to graph exclusion behavior.
- Removes sentences with negated entities from the vector embedding query.
- Retrieves graph evidence and vector evidence once for the initial CLI turn.

Start the chatbot with an initial prompt:

```bash
python query.py --prompt "I would like a recipe with noodle in it. But I want a recipe without soba in it. I want recipes without Sulfites for allergens."
```

You can then ask follow-up questions in the same terminal:

```text
When heating the non stick fry pan, what temperature setting should I set the heat to?
```

The CLI keeps the conversation history and cached graph/vector evidence. Type `clear` to discard the cached recipe set and force retrieval for a new recipe request. Type `quit` to exit.

### Run the Chatbot with Observability

If you are interested in seeing the debug statements, the prompts being used, and decisions the LLM is making for how `attribution` and `negation` are being mapped to the requests, run the following command (NOTE: the `--observability` flag):

```bash
python query.py --prompt "I would like a recipe with noodle in it. But I want a recipe without soba in it. I want recipes without Sulfites for allergens."  --observability
```

Again, ask the follow up question:

```text
When heating the non stick fry pan, what temperature setting should I set the heat to?
```

You should get the same results as before.

## Clean Up

> **IMPORTANT NOTE:** Stop the **Chatbot** in this step.
> **IMPORTANT NOTE:** Stop the **LLM Service** in this step.

## Takeways

When we built this chatbot, we had to confront the reality that missing a dietary restriction is not an oopsie... it can be a catastrophic failure. Since something like a peanut allergy requires a completely deterministic mechanism, we overloaded our Named Entity Recognition (NER) pipeline using `negspacy` to map attribution and negation. That exact technique is what allows our agent to parse sentences, identify if the `negation` flag is `True` or `False`, and dynamically route ingredients to either the `--query` or `--notquery` parameters. We are effectively turning unpredictable natural language into a highly controlled, safe query router.

Beyond parsing single commands, the true mark of an intelligent agent is its ability to maintain conversation history. If a user asks a follow-up question, like asking what temperature setting to use for a non-stick fry pan, the chatbot seamlessly retains the context of that prior interaction. When I design slide decks and technical workshop materials for fellow software engineers and data scientists, I always emphasize that building this statefulness is what separates a sterile search tool from a dynamic conversational partner. By running the chatbot with the `--observability` flag, you can watch the LLM map out these exact logic paths and see the debug statements in real-time. Dive into those debug logs, scrutinize the decision trees, and go build some unbreakable systems!
