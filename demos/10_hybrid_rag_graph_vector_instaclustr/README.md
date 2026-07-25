# Section 4 Part 2: Complete RAG Agent Used Remote Vector Embeddings

This lesson is functionality identical to the previous one, `Section 4, Step 1: Complete RAG Agent Using the Recipe Dataset`. The difference is that we are going to intentionally move or distribute various components of the RAG pipeleine to remote locations.

Here is a breakdown of where everything is "running" from:

- Vector Embeddings is retrieved from our OpenSearch instance hosted by [NetApp Instaclustr](https://www.instaclustr.com/)
- Neo4j Knowledge Graphs on your local laptop
- [Qwen2.5-7B-Instruct-1M-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF) is running on your laptaop
  - **OPTION:** Will Provide instructions on using an OpenAI-compatable inference platforms like OpenAI, Nebius, Fireworks AI, etc.
- Chatbot Process on local Laptop

The purpose of this is to demonstrate decoupling various components of inference to open up the ideas for architecture possibilities.

## Prerequisites

- Python 3.10 **ONLY**
- Neo4j 5.26.16 running locally
- OpenSearch 3.5 running in AWS via [NetApp Instaclustr](https://www.instaclustr.com/)
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

If you don't have Neo4j running (you can check by running: `podman ps`). You can run the following command:

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
    -v "$HOME/neo4j/data:/data" \
    -v "$HOME/neo4j/plugins:/plugins" \
    -v "$HOME/neo4j/import:/var/lib/neo4j/import" \
    docker.io/neo4j:5.26.16
```

## Step 1: Prerequisite Setup

### Remember Your Instaclustr USER_INDEX and USER_PASSWORD?

<span style="color: red;">**IMPORTANT NOTE:** If you have already claimed your Instaclustr USER_INDEX and USER_PASSWORD then you can move on to the next part which is starting your local LLM.</span>

We are going to reuse the Instaclustr USER_INDEX you claimed in`Section 3` and the we are going to reuse the same dataset that we already ingested.

If you need a reminder, the [Google Sheet at this location](XXXXXXXX), has the list of `USER_INDEX` and `USER_PASSWORD` that you claimed.

For the commands in this section, you will always preface the `python` command with your `USER_INDEX` and `USER_PASSWORD` values.

Example:

```bash
USER_INDEX=0 USER_PASSWORD=abcxyz123 python ingest.py
```

### (Option 1) Start Our Local LLM

<span style="color: cyan;">**IN A NEW TERMINAL:** We need to start our LLM service. Run the following commands FROM THIS DIRECTORY and don't forget to start your virtual environment if you are using one:</span>

```bash
python llm_service.py
```

### (Option 2) Use an OpenAI-compatible API Provide (like OpenAI, Nebius, Fireworks AI, etc)

If you are interested in using a remote LLM provider that is OpenAI-compatible, set the following environment variables:

```bash
USE_EXTERNAL_AI=1
EXTERNAL_LLM_URL=YOUR_URL_VALUE
EXTERNAL_LLM_API_KEY=YOUR_APIKEY_VALUE
EXTERNAL_LLM_MODEL=YOUR_MODEL_VALUE
EXTERNAL_LLM_MAX_TOKENS=245760
```

Personally, I like [Nebius](https://nebius.com/) as a provider, so this might looks like:

```bash
USE_EXTERNAL_AI=1
EXTERNAL_LLM_URL="https://api.tokenfactory.nebius.com/v1/"
EXTERNAL_LLM_API_KEY=YOUR_APIKEY_VALUE
EXTERNAL_LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"
EXTERNAL_LLM_MAX_TOKENS=245760
```

**OR**, you can also combine this on the command line in the next step:

```bash
USE_EXTERNAL_AI=1 EXTERNAL_LLM_URL="https://api.tokenfactory.nebius.com/v1/" EXTERNAL_LLM_API_KEY=YOUR_APIKEY_VALUE EXTERNAL_LLM_MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507" EXTERNAL_LLM_MAX_TOKENS=245760 python query.py --prompt "I would like a recipe with noodle in it. But I want a recipe without soba in it. I want recipes without Sulfites for allergens."
```

### Let's Run the Chatbot

Let's run our chatbot using the following command:

```bash
USER_INDEX=YOUR_INDEX USER_PASSWORD=YOUR_INDEX_PASSWORD python query.py --prompt "I would like a recipe with noodle in it. But I want a recipe without soba in it. I want recipes without Sulfites for allergens."
```

You can now ask the chatbot a follow up conversation because it remembers the conversation history.

```text
When heating the non stick fry pan, what temperature setting should I set the heat to?
```

You should get the same results as before.

## Clean Up

> **IMPORTANT NOTE:** Stop the **Chatbot** in this step.
> **IMPORTANT NOTE:** Stop the **LLM Service** in this step.

## Takeways

We have achieved the architectural flexibility by deliberately shattering the monolith and distributing our hybrid RAG components across the network. In this lesson, we proved that your system's components do not need to be tightly bound to a single local machine to operate cohesively. By offloading our vector embeddings to a remote OpenSearch instance hosted by [NetApp Instaclustr](https://www.instaclustr.com/) while keeping our Neo4j Knowledge Graph local, we demonstrate the power of a highly decoupled architecture. Designing systems this capability in mind is a massive win for real-world enterprise engineering, as it allows infrastructure teams to optimize, secure, and scale each layer of the data retrieval pipeline without forcing a complete rewrite of the application logic.

This architectural independence extends beautifully into our generative layer, unlocking massive possibilities for how we handle compute resource constraints. While running our local [Qwen2.5-7B-Instruct-1M-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF) model on a laptop is phenomenal for privacy and offline development. This section also shows how seamlessly we can pivot to high-performance, OpenAI-compatible remote inference platforms like Nebius, Fireworks AI, or OpenAI by merely swapping out a few environment variables. Toggling between local execution and a massive cloud-hosted model without altering a single line of chatbot logic is exactly how robust production systems are built. It grants you the strategic leverage to balance cost, latency, and model capacity on the fly. Ultimately, this distributed topology proves that an intelligent RAG agent is not a rigid piece of software, but an adaptable orchestration of modular services capable of living anywhere across your infrastructure.
