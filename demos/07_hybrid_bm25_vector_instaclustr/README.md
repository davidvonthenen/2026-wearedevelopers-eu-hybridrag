# Section 3 Part 1: Breaking Down Hybrid (BM25 + Vector) RAG using the Recipe Dataset

This section demonstrates a Hybrid RAG pipeline where retrieval and inference are intentionally separated. That separation is the point of the exercise, because it lets you inspect two things independently:

- what hybrid retrieval looks like when BM25 and vector search are both backed by OpenSearch
- how the shape and presentation of retrieved data affects inference

The project has three entry points:

- `ingest.py` loads recipe data into OpenSearch BM25 and vector indexes.
- `query.py` inspects retrieval results without calling an LLM.
- `inference.py` runs a two-stage inference demonstration using hardcoded BM25 and vector evidence so prompt behavior can be studied independently from retrieval behavior.

## Prerequisites

- Python 3.10 **ONLY**
- OpenSearch 3.5 running in AWS via [NetApp Instaclustr](https://www.instaclustr.com/)
- [recipes-with-nutrition on Huggingface](https://huggingface.co/datasets/datahiveai/recipes-with-nutrition). This already exists within this directory.

## Installation

<span style="color: red;">**IMPORTANT NOTE:** If you are running this hands-on lab in sequence, you do not need to perform these **Installation Step** again.</span>

**If** you haven't created a [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), [venv](https://docs.python.org/3/library/venv.html), or [uv](https://pydevtools.com/handbook/how-to/how-to-install-uv/) virtual environment yet, please do so now. This single environment can be used throughout this entire lab exercise. Call this environment `2026-wearedevs-eu-workshop`.

```bash
conda create -n 2026-wearedevs-eu-workshop python=3.10
```

Please activate this environment:

```bash
conda activate 2026-wearedevs-eu-workshop
```

To install the software dependencies:

```bash
# if you are using: miniconda or venv
pip install -r requirements.txt

# OR, if using  uv
uv pip install -r requirements.txt 
```

The step uses a managed OpenSearch instance provided through [NetApp Instaclustr](https://www.instaclustr.com/) that I am providing for this workshop. The code uses standard OpenSearch APIs, so the same project can also run against a compatible local or cloud-hosted OpenSearch instance when configured with the appropriate environment variables.

The only thing you will need (which will be provided in a Google Sheet in `Step 1`) is:

- an User Index
- an OpenSearch User Password

## Step 1: Prerequisite Setup

### Start Our Named Entity Reognition Service

<span style="color: cyan;">**IN A NEW TERMINAL:** We need to start our NER service. Run the following commands FROM THIS DIRECTORY and don't forget to start your virtual environment if you are using one:</span>

```bash
# If you are using the same virtual environment: 2026-wearedevs-eu-workshop, you don't need to run the command below:
# For miniconda or venv:
# pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
# OR, if using uv:
# uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz

# BUT YOU DO NEED TO START THE NER SERVICE USING THE COMMAND BELOW
python ner_service.py
```

The NER service is used by `ingest.py` to attach entity tags to BM25 chunks and by `query.py` to extract entities from the user query.

### Ingest the Recipe Dataset

<span style="color: cyan;">**IMPORTANT NOTE:** This is a new step since we are using an AWS-hosted OpenSearch cluster.</span>

We need to ingest the dataset to an Instaclustr OpenSearch instance in AWS and we are going to use the same Recipe dataset as in `Section 2`.

We are saving our **BM25 chunks** to the OpenSearch index named `recipes-bm25-USER_INDEX`.
We are saving our **vector embeddings** to the OpenSearch index named `recipes-bm25-USER_INDEX`.

```bash
USER_INDEX=YOUR_USER_INDEX USER_PASSWORD=YOUR_PASSWORD python ingest.py
```

Using the [Google Sheet at this location](https://bit.ly/4gm2h6v), claim a `USER_INDEX` and `USER_PASSWORD` by placing your name on the row you are claiming. This prevents others from using your same OpenSearch indexes.

For the commands in this section, you will always preface the `python` command with your `USER_INDEX` and `USER_PASSWORD` values.

Example:

```bash
USER_INDEX=0 USER_PASSWORD=abcxyz123 python ingest.py
```

## Step 2: Data Retrieval Using BM25 and Vector Embeddings

This step demonstrates retrieval without inference. `query.py` first runs BM25 search for grounding, then uses the grounded recipe IDs to constrain vector search.

Run a basic noodle recipe query:

```bash
USER_INDEX=YOUR_USER_INDEX USER_PASSWORD=YOUR_PASSWORD python query.py --query "I am looking for noodle recipes."
```

The output is split into two sections:

- `BM25 Grounding Matches`
- `Vector Hits`

By default, `query.py` hides the large `text` field so the terminal output stays readable. Add `--verbose` when you want to inspect the actual chunk text returned by each retrieval channel:

```bash
USER_INDEX=YOUR_USER_INDEX USER_PASSWORD=YOUR_PASSWORD python query.py --query "I am looking for noodle recipes." --verbose
```

### Explicit Exclusions and Structured Filters

This project does not automatically interpret natural-language negation such as "without soba" as a structured exclusion. The exclusion must be passed explicitly with the appropriate flag.

To exclude recipe chunks containing an allergen like `Sulfites`, use `--exclude-caution`:

```bash
USER_INDEX=YOUR_USER_INDEX USER_PASSWORD=YOUR_PASSWORD  python query.py --query "I am looking for noodles recipes without soba." --exclude-caution "Sulfites"
```

The allergy caution filter is based on normalized recipe metadata from the dataset. NER entities are stored as BM25 tags and can boost entity-aware matching, but allergy exclusion is handled through the structured caution metadata, not through free-form text guessing.

> **REMEMBER:** This DOES require that we have special logic to understand this "negated context" and explicitly use the `--exclude-caution` filter. Remember that million dollar question, "How do you strategically allow for our query to `understand` this negation?" This question is still something we need to address.

**NOTE:**, also remember that the top recipes returned are those containing "noodle" and "soba" depsite having a known qualifier "without". If we wanted to handle this use case, you would need to add even more structure (via tags or key/value pairs) to your "unstructured data" which we will address in `Part 2` using `Knowledge Graphs` with `Vector Embeddings` for Hybrid RAG.

## Step 3: Providing Data for Inference

Now that we have data to provide to our Small Language Model (SLM) to synthesize the data and extract an answer, this section aims to show how we go about doing that through the lens of AI governance (using repeatable, observable steps).

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

If important to note that we are using hardcoded BM25 and vector evidence (obtained from the query above) so the inference behavior can be studied by itself.

Take note of the answer the SLM provided in the terminal where you ran the `python inference.py` command.

## Clean Up

> **IMPORTANT NOTE:** Stop the **NER Service** in this step.
> **IMPORTANT NOTE:** Stop the **LLM Service** in this step.

## Takeways

What we have built here is an absolute architectural triumph! By decoupling the data retrieval mechanism from the generative inference engine, we have constructed a highly resilient and scalable pipeline. In my years of engineering these system, I have learned that isolating your data layer from your compute layer is paramount. Your BM25 data and embeddings can live entirely elsewhere, specifically in our AWS-hosted OpenSearch cluster, while the inference happens independently on local hardware. This means you can swap out the underlying model or scale your search infrastructure without breaking a single dependency, giving you the ultimate flexibility.

### Truth Grounding and Semantic Nuance

The real magic happens when we dissect the Hybrid RAG implementation. We are actively marrying the absolute certainty of BM25 retrieval with the semantic elegance of vector embeddings. BM25 acts as our uncompromising anchor for truth grounding; it guarantees that when we need a recipe completely free of sulfites, we retrieve factual, exact-match data. Vector embeddings are then layered on top to provide the contextual and semantic nuance required for a fluid, natural conversation. Instead of relying on a single retrieval method that might fail at keyword matching or hallucinate from pure vectors, we get the best of both worlds: rock-solid facts paired with rich textual meaning.

### AI Governance

When evaluating the final output, the quality and traceability should grab your immediate attention. Did you spot the [B1] and [V1] tags appended to the generated text? That right there is the Holy Grail of AI governance! We are forcing the system to show its work, providing an explicit audit trail that links every claim directly back to the source BM25 chunk or vector embedding. I have sat through countless compliance meetings, and I can assure you that having granular observability into why a system generated a specific response will save your sanity when stakeholders start asking questions. Take a moment to appreciate this level of transparency, and carry this exact mindset forward as we prepare to integrate Knowledge Graphs in the next phase!
