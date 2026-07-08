PYTHON ?= python
DATA_DIR ?= ./bbc
QUESTION1 ?= How much did OpenAI purchase Windsurf for?
QUESTION2 ?= How much did Google purchase Windsurf for?

.PHONY: help ingest ner-server server query client env

help:
	@echo "Make sure you have the Neo4j container running."
	@echo "Run the following commands in this order:"
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

client: query

env:
	@echo "NEO4J_URI=$${NEO4J_URI:-bolt://localhost:7687}"
	@echo "NEO4J_USER=$${NEO4J_USER:-neo4j}"
	@echo "NEO4J_PASSWORD=$${NEO4J_PASSWORD:-neo4jneo4j}"
	@echo "DATA_DIR=$${DATA_DIR:-$(DATA_DIR)}"
	@echo "NER_SERVICE_URL=$${NER_SERVICE_URL:-http://127.0.0.1:8000/ner}"
	@echo "NER_SERVICE_TIMEOUT=$${NER_SERVICE_TIMEOUT:-8.0}"
	@echo "MODEL_PATH=$${MODEL_PATH:-~/models/Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf}"
	@echo "PORT=$${PORT:-$(PORT)}"
