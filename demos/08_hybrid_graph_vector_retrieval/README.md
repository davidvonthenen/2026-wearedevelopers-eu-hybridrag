# Section 3 Part 2: Breaking Down Hybrid (Graph + Vector) RAG using the Recipe Dataset

This project breaks a Hybrid RAG pipeline into separate retrieval and inference steps so each part can be inspected independently:

- **Retrieval:** Neo4j provides graph-grounded recipe facts and OpenSearch provides semantic vector chunks.
- **Inference:** a local or external OpenAI-compatible chat endpoint performs a two-pass answer generation flow.
- **Governance:** prompts and outputs use citation handles such as `[G1]` and `[V1]` so claims can be traced back to retrieved evidence.

The point of the exercise is not to hide retrieval behind a magical "ask the model" button, because apparently one opaque box was not enough for civilization. The point is to show how the shape and presentation of retrieved data changes the final answer.

This directory contains three main scripts:

- `ingest.py` loads recipe CSV rows, fetches readable recipe page text, writes structured recipe facts to Neo4j, and writes embedded recipe chunks to OpenSearch.
- `query.py` shows retrieval in isolation by printing graph matches and vector hits separately.
- `inference.py` shows two-pass inference in isolation using hardcoded graph and vector evidence, not live database retrieval.

The intended retrieval design is:

1. Use the **graph** for authoritative facts and constraints, such as allergy cautions, diet labels, health labels, cuisine type, meal type, dish type, and exact include/exclude text filters.
2. Use **vectors** for semantic recipe text, phrasing, ingredients, preparation language, and human-readable detail.
3. Use the **LLM/SLM** only after retrieval has already produced the evidence it is allowed to use.

## Installation Prerequisite Software

Please see the README.md at the root of the repo.

## Step 1: Prerequisite Setup

### Ingest the Recipe Dataset 

<span style="color: red;">**IMPORTANT NOTE:** If you are running this hands-on lab in sequence, you do not need to perform data ingesting again.</span>

Since we are reusing the same recipe dataset and we already performed the ingest to following Neo4j and OpenSearch container, we can just move onto step 2:

- `neo4j database`
- `recipes-vector`

**If** you are starting over (like you destroyed your Neo4j and OpenSearch container), you can go back to the following folders to run `python ingest.py`. **Again, do not to this unless you are absolutely sure you need to!**

- `05_bm25_retireval`
- `06_graph_retrieval`

## Step 2: Data Retrieval Using Graph and Vector Embeddings

This demonstrates data retrieval using Knowledge Graphs and Vector Embeddings. We are going to query for simple "noodle" recipes.

```bash
python query.py --query noodle
```

The output is split into two sections:

- `Graph Matches`: recipe records returned by Neo4j after applying exact graph/content constraints.
- `Vector Hits`: OpenSearch kNN chunk results for the semantic query.

By default, the output hides large text fields so the terminal remains readable. Use `--verbose` to print full graph `content` and vector chunk `text` fields.

Run a constrained query:

```bash
python query.py --query noodle --notquery soba --exclude-caution "Sulfites"
```

In this command:

- `--query noodle` is used as a graph content filter and as the semantic vector query.
- `--notquery soba` excludes graph records whose recipe content contains `soba`.
- `--exclude-caution "Sulfites"` excludes recipes with that caution label.
- Vector search is constrained to the recipe IDs that survived the graph pass.

This is the important distinction: the graph handles exact constraints, while vectors provide semantic chunks for the surviving recipe candidates. Asking a vector index to enforce negation by vibes alone is how demos become lawsuits with prettier screenshots.


**NOTE:** Since the graph ontology (or structure for our data) is completely filled out, We are able to get exact matches/results as we did in `Section 2, Step 3`. The graph structure contains all relationships between any data points. We know what recipes contain what ingredients. We know what recipes contain with allergens. This is also why we can handle negation or exclusion so easily.

Now the problem becomes how do we understand this "negation" (or I don't want to search of things with XYZ) in our data retrieval? This is where we bring it all together in `Section 4` of this workshop.

## Step 3: Providing Data for Inference

Now that we have data to provide to our Small Language Model (SLM) to synthesize the data and extract an answer, this section aims to show how we go about doing that through the lens of AI governance (using repeatable, observable steps) by through the lens of using Knowledge Graphs.

### Start Our Local LLM

<span style="color: cyan;">**IN A NEW TERMINAL:** We need to start our LLM service. Run the following commands FROM THIS DIRECTORY and don't forget to start your virtual environment if you are using one:</span>

```bash
python llm_service.py
```

### Let's Run Inference

Using our local [Qwen2.5-7B-Instruct-1M-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF) SLM, we are going to structure the prompt and perform that 2-phase truth grounding step using `BM25 data` and then use `Vector Embeddings` to provide semantic detail to provide:

- a more human-like response
- more detail or color to the answer

Let's run infernce using the command below:

```bash
python inference.py
```

We are using hardcoded graph and vector evidence (obtained from the query above) so the inference behavior can be studied by itself.

The two-pass flow is:

1. **Graph grounding pass:** the model answers using only `[G#]` graph evidence. Recipe identity, labels, cautions, ingredients, nutrition, source, and URL must come from graph evidence.
2. **Vector refinement pass:** the model rewrites the grounded draft for clarity using `[V#]` vector evidence only for wording and semantic support. It must preserve graph-grounded facts and citations.

To inspect the exact prompts sent to the model, enable observability:

```bash
OBSERVABILITY=true python inference.py
```

## Clean Up

> **IMPORTANT NOTE:** Stop the **NER Service** in this step.
> **IMPORTANT NOTE:** Stop the **LLM Service** in this step.

## Takeways

Let us talk about the absolute powerhouse that is the Knowledge Graph. By structuring our data as a graph ontology, we are handing our system a complete, deterministic map of relationships. I have spent countless late nights untangling semantic messes in production systems because flat data structures completely failed to handle basic concepts like negation. When a user asks for a recipe with "no soba" and "sulfites free," vector similarity alone will often retrieve recipes loaded with soba because the keyword matches highly in the embedding space. The graph, however, deterministically knows exactly what ingredients are included and excluded, treating constraints as exact graph traversals rather than fuzzy probabilistic guesses. This deterministic structure provides an ironclad foundation for our truth-grounding phase.

### Deterministic vs Non-deterministic

Next, inspect the magnificent quality of the final inference output generated by our two-pass pipeline. The Small Language Model (SLM) does not guess; it builds its answer entirely upon the graph's unshakeable facts, then uses the vector chunks to weave in semantic textual reference. We are successfully decoupling the heavy lifting of retrieval from the creative synthesis of the LLM. In my experience deploying these architectures, forcing the language model to rely exclusively on highly structured graph data for facts, while using vectors for phrasing and context, eliminates the hallucinations that keep data scientists or AI engineers awake at night. The result is a human-like, richly detailed response that remains rigorously accurate.

### More AI Governance

Finally, let us marvel at the true hero of enterprise AI: governance and observability. If you look closely at the generated text and the prompt rules, you will notice the precise tracking of citation handles like `[G1]` and `[V1]`. By forcing the model to append these exact tags to every factual claim, we establish a non-negotiable audit trail. Embrace this level of traceability... it is the difference between an unreliable demo and a robust, production-ready system.
