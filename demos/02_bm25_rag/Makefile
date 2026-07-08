PYTHON ?= python
DATA_DIR ?= ./bbc
HOST ?= 0.0.0.0
PORT ?= 8000
QUESTION1 ?= How much did OpenAI purchase Windsurf for?
QUESTION2 ?= How much did Google purchase Windsurf for?

.PHONY: help ingest ner-server server query env

help:
	@echo "Make sure you have the OpenSearch server running."
	@echo "Then run the following commands in this order:"
	@echo "1. make server"
	@echo "2. make ingest"
	@echo "3. make query"

ingest:
	$(PYTHON) ingest.py

server:
	$(PYTHON) ner_service.py

ner-server: server

query:
	$(PYTHON) query.py

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
