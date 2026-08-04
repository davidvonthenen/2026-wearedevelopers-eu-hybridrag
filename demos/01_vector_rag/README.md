# Section 1 Part 1: Local Vector RAG Agent

This project provides an end-to-end Retrieval-Augmented Generation (RAG) workflow. It stores BBC article chunks as vectors in OpenSearch 3.5, retrieves relevant chunks with pure vector k-NN search, and answers questions with a local GGUF model loaded directly through `llama-cpp-python`.

The main scripts are:

- `ingest.py`: reads `./bbc/<category>/*.txt`, chunks articles, embeds each chunk, and writes the chunk plus metadata to OpenSearch.
- `query.py`: loads the local GGUF model, embeds the question, retrieves vector hits from OpenSearch, builds a grounded prompt, and prints the answer plus retrieved hit metadata.

## Installation Prerequisite Software

Please see the README.md at the root of the repo.

## Step 1: Ingest the BBC Dataset

### Ingest the British Broadcasting Corporation (BBC) Dataset 

The ingest script reads the BBC dataset from `./bbc`, chunks each `.txt` document, embeds each chunk, and stores the chunks in the OpenSearch index named `bbc-vector-chunks` by default.

The default embedding model is [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). The code normalizes embeddings before storing them in a `knn_vector` field configured for cosine similarity.

```bash
python ingest.py
```

Before re-indexing a document, the ingest process deletes existing chunks for that document path so repeated runs replace old chunks instead of duplicating them.

## Step 2: Query the RAG Pipeline

`query.py` performs one local RAG pass:

1. The project already provides defaults for all parameters to run.
2. Loads the local GGUF model with `llama-cpp-python`.
3. Embeds the user question with the configured sentence-transformers model.
4. Retrieve top-k chunks from OpenSearch using vector k-NN against the `embedding` field.
5. Build a prompt containing the question and retrieved context.
6. Ask the local model to answer only from the retrieved snippets.

Let's run our first question on our RAG Agent:

```bash
python query.py --question "How much did OpenAI purchase Windsurf for?"
```

Note the results of this question.

Now, run the next query below:

```bash
python query.py --question "How much did Google purchase Windsurf for?"
```

Note the results of this question. (It's possible you might have gotten a "I don't know" answer. This is normal.)

## Takeways

Using a single news dataset, pure vector semantic similarity search alone can cause an LLM to provide contradictory answers.

The dataset includes news articles as they happened over a period of time without reconciling the information contained within.

Facts that are true:

- On May 06, 2025: OpenAI made an announcement to acquire Windsurf
- On July 12, 2025: Google announceed they are paying $2.4 billion to license some of Windsurf's technology and acqui-hired key Windsurf employees.
- On July 14, 2025: Cognition agreed to buy Windsurf days after Google poached CEO in $2.4 billion licensing deal.

> **NOTE:** Depending on how lucky or unlucky you are, one of the questions above might actually provide an "I don't know." answer. This happens because the vector similarity search wasn't able to use kNN to find documents that were a close enough match despite the information existing in the dataset. Why? At a very high-level and simplistic terms, most everything AI and ML is probabilistic not deterministic.

Assuming you got answers for both questions... If you were strictly looking for answers based on the data, each answer is correct. **HOWEVER**, in the real world, these aren't the answer we are expecting. We are expecting our AI systems (RAG or otherwise) to provide factual, truth based answers which take into account contraditions over time.
