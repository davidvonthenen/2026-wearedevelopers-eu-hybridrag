# WeAreDevelopers Europe 2026 - (WORKSHOP) From Vector Search to Better Understanding: How Hybrid RAG Improves Answers, Not Just Matches

Welcome to the landing page for the workshop `From Vector Search to Better Understanding: How Hybrid RAG Improves Answers, Not Just Matches` at `WeAreDevelopers Europe 2026`.

## What to Expect

This repo intends to provide instructions and materials for our workshop. Instruction will be provided and tasks are self-paced (with help from me should you need it). This workshop is broken down into 3 majors sections:

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
  - (HIGHLY Recommended) Use [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [uv](https://docs.astral.sh/uv/getting-started/installation/), or [venv](https://docs.python.org/3/library/venv.html) virtual development environment. (I prefer miniconda or uv)
- (HIGHLY Recommended) [Huggingface developer token](https://huggingface.co/settings/tokens) saved to `HF_TOKEN` environment variable.
- Install [Podman](https://podman.io/docs/installation) (or equivalent container runtime).
  - Will use [OpenSearch](https://opensearch.org/) and [Neo4j](https://neo4j.com/) containers.

LLM / Model to Pre-Download :
- [Qwen2.5-7B-Instruct-1M-GGUF](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF) – Specifically the [Q5_K_M version](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf)
  - On the HuggingFace page, navigate to “files” and [download this model](https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-1M-GGUF/blob/main/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf).
  - Drop the file into your ~/models folder. (You might need to create this.)
- Pre-pull the following containers:
  - podman pull opensearchproject/opensearch:3.5.0
  - podman pull opensearchproject/opensearch-dashboards:3.5.0
  - podman pull neo4j:5.26.16

## Workshop Materials

Coming Soon!
