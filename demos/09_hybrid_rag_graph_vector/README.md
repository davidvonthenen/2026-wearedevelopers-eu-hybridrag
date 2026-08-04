# Section 4 Part 1: Complete RAG Agent Using the Recipe Dataset

This project demonstrates a chatbot-style Hybrid RAG pipeline for recipe question answering. It uses Neo4j as the graph-backed truth channel for structured recipe facts, labels, cautions, nutrition, source, and URL metadata, and OpenSearch as the vector-backed semantic channel for recipe wording, ingredients, instructions, and fuzzy matching.

The retrieval pattern builds on the earlier Graph + Vector recipe retrieval exercise, but this section focuses on turning that retrieval flow into an interactive chatbot. The important behavior to notice is the division of responsibility: graph retrieval is used for deterministic constraints, while vector retrieval adds semantic detail for more natural answers.

The projects' main entry point is `query.py`, which runs the chatbot-style application. It can run as an interactive CLI or as an OpenAI-compatible REST service.

The data retrieval mechanism is the same as `Section 3, Step 2: Breaking Down Hybrid (Graph + Vector) RAG using the Recipe Dataset`. We have already discussed via the workshop presentation and through the various materials in each section, what the data retrieval looks like; therefore, `Section 4` will only focus on inference.

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
