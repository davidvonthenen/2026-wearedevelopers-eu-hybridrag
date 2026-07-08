PYTHON ?= python
DATA_DIR ?= ./bbc
HOST ?= 0.0.0.0
PORT ?= 8000
QUESTION1 ?= I am looking for noodle recipes.
QUESTION2 ?= I am looking for noodles recipes without soba.
QUESTION3 ?= I am looking for noodles recipes without soba that doesn't contain Sulfites.
QUESTION4 ?= I am looking for noodles recipes without soba.
ALLERGEN4 ?= Sulfites

.PHONY: help ner-server ingest server query client env

help:
	@echo "Start your OpenSearch instance first."
	@echo "Then, run in the following order:"
	@echo "  make ingest    - Ingest data"
	@echo "  make server    - Start the NER server"
	@echo "  make query     - Run queries"

ingest:
	$(PYTHON) ingest.py

server:
	$(PYTHON) ner_service.py

ner-server: server

query:
	$(PYTHON) query.py --query "$(QUESTION1)"
	$(PYTHON) query.py --query "$(QUESTION2)"
	$(PYTHON) query.py --query "$(QUESTION3)"
	$(PYTHON) query.py --query "$(QUESTION4)" --exclude-caution "$(ALLERGEN4)"

client: query

env:
	@echo "OPENSEARCH_HOST=$${OPENSEARCH_HOST:-127.0.0.1}"
	@echo "OPENSEARCH_PORT=$${OPENSEARCH_PORT:-9201}"
	@echo "OPENSEARCH_INDEX=$${OPENSEARCH_INDEX:-$(INDEX)}"
	@echo "EMBEDDING_MODEL=$${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
	@echo "LLAMA_MODEL_PATH=$${LLAMA_MODEL_PATH:-./models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf}"
	@echo "LLAMA_CTX=$${LLAMA_CTX:-65536}"
	@echo "LLAMA_N_THREADS=$${LLAMA_N_THREADS:-$$($(PYTHON) -c 'import os; print(os.cpu_count() or 1)')}"
	@echo "LLAMA_N_GPU_LAYERS=$${LLAMA_N_GPU_LAYERS:-20}"
	@echo "RAG_TOP_K=$${RAG_TOP_K:-3}"
	@echo "RAG_NUM_CANDIDATES=$${RAG_NUM_CANDIDATES:-50}"
	@echo "SERVER_HOST=$${SERVER_HOST:-$(HOST)}"
	@echo "SERVER_PORT=$${SERVER_PORT:-$(PORT)}"
