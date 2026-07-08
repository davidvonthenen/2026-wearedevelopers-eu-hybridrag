# Section 2 Part 3: Graph Retrieval using the Recipe Dataset

This project demonstrates Knowledge Graph retrieval over a recipe dataset using Neo4j. It has two primary scripts:

- `ingest.py` loads recipe CSV rows into Neo4j, optionally fetches recipe page text from each URL, and stores both structured metadata and graph content chunks.
- `query.py` shows how recipe records can be retrieved from Neo4j using graph metadata filters and lightweight text predicates.

## Prerequisites

- Python 3.10 **ONLY**
- Neo4j 5.26.16 running locally
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
# create the network for neo4j
podman network create graph-net

# deploy the Neo4j container image
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

### Ingest the Recipe Dataset 

<span style="color: cyan;">**IMPORTANT NOTE:** This is a new step. Even through we have already used Neo4j for Knowledge Graph retrieval in Section 1 Step 3, this ingests the RECIPE dataset using structured data.</span>

We need to ingest the dataset to Neo4j. This Recipe dataset is a collection of simple recipes capturing instructions, ingredients, allergens, etc.

```bash
python ingest.py
```

The ingest step creates a graph shaped around recipes and their structured metadata:

- `(:Recipe)` nodes store recipe identity, source, URL, servings, calories, normalized label lists, fetch metadata, and combined readable content in `Recipe.content`.
- `(:RecipeChunk)` nodes store chunked recipe text linked back to the recipe with `(:Recipe)-[:HAS_CHUNK]->(:RecipeChunk)`.
- Metadata nodes are created for `AllergyWarning`, `DietLabel`, `HealthLabel`, `CuisineType`, `MealType`, `DishType`, and `RecipeSource`.
- Typed relationships such as `HAS_CAUTION`, `HAS_HEALTH_LABEL`, and `HAS_CUISINE_TYPE` make structured filtering explicit.

This section still performs content chunking, but those chunks are stored in Neo4j as graph nodes. They are not embedded and they are not written to a vector index.

## Step 2: Data Retrieval Using Knowledge Graphs

This demonstrates Graph data retrieval using the recipe dataset. We are going to query for simple "noodle" recipes.

```bash
python query.py --query noodle
```

Observe the results.

Now query for `noodle` recipes while excluding matches for `soba`:

```bash
python query.py --query noodle --notquery soba
```

The first thing to notice is that we don't need to mince words (excuse the pun). We can directly query for recipes with "noodle", but without "soba". Notice the difference in the results.

Now, let's exclude recipes with the `Sulfites` allergy caution:

```bash
python query.py --query noodle --notquery soba --exclude-caution "Sulfites"
```

This is the first time that we can return valid results that are based on facts (and the desired result)... and not accidentally kill someone with an allergy in the process. Notice that there is only a single recipe that meets this criteria. This is the difference is dealing with structured data and why, as AI engineers and data scientists, we need to understand this narrative more.

The question now is... how do we map this to an AI solution? Let's explore this in the next section.

## Takeways

Vector databases and lexical search engines are inherently probabilistic; they guess what a user means by operating on high-dimensional distances or token statistics. Knowledge Graphs completely rewrite this paradigm by introducing explicit, deterministic relationships. Instead of hoping a neural network accurately captures the subtle nuance of exclusion, graph architectures treat data as discrete entities (Nodes) connected by defined, typed relationships (Edges).

### The Mechanics of Relational Precision

When executing a parameterized graph query (such as isolating "noodle" while strictly eliminating "soba" and filtering out "Sulfites") the underlying engine completely abandons linguistic speculation. It performs graph traversal and relational set algebra.

- Set Theory Over Proximity: The query engine identifies the specific node representing the ingredient "Noodle," follows its relationships to associated recipes, and then systematically prunes any paths intersecting with the "Soba" or "Sulfites" nodes.

- Absolute Factual Certainty: This architectural precision explains why the workflow returned exactly one solitary, verified recipe. In high-stakes applications like food allergen tracking, medical dosing, or legal compliance, relying on a model's "semantic vibe" is an existential hazard. Probabilistic architectures optimize for plausibility; structured graphs execute on absolute truth.

You should know we also have this in other systems as well. A great example... SQL. Yes, I said it. A technology from the 1970s.

### Architectural Reflection and Call to Action

It is a profound irony of modern AI architecture that engineering teams frequently spend months fine-tuning dense embedding models or chaining complex prompt templates to prevent hallucinations, entirely ignoring that structured databases solve data integrity out of the box. Forcing a neural network to calculate what not to surface is a structural misuse of the technology.

The engineering objective must shift from optimization to translation: how do we gracefully transform a chaotic, conversational user prompt into a structured graph query? Bridging this chasm requires a hybrid orchestration layer. By deploying an upstream language model strictly as an interface translator to generate graph queries, you combine natural human language with rigid, safe database execution.

We will cover this in the next section...
