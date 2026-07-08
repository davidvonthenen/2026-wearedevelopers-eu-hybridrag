PYTHON ?= python
DATA_DIR ?= ./bbc
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: help ingest query client chat server llm-server env

help:
	@echo "Start your OpenSearch instance first."
	@echo "Then, run in the following order:"
	@echo "  make ingest    - Ingest data"
	@echo "  make query     - Run queries"
	@echo "  make llm-server    - Start the LLM server (keep this window open)"
	@echo "  make client    - Run inference"
	
ingest:
	$(PYTHON) ingest.py

query:
	$(PYTHON) query.py --query noodle --exclude-caution "Sulfites"
	$(PYTHON) query.py --query noodle --notquery soba --exclude-caution "Sulfites"

llm-server:
	$(PYTHON) llm_service.py

server: llm-server

query:
	$(PYTHON) query.py

client: query

chat: query

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
