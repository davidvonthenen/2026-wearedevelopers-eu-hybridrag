"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from pathlib import Path
from typing import Optional


def _get_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_optional_int(name: str, default: Optional[int] = None) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _default_llama_n_gpu_layers() -> int:
    """Return a reasonable GPU offload default for the local LLM runtime."""

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return -1
    return 20


@dataclass
class Settings:
    """Connection and model settings for recipe graph/vector ingestion."""

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4jneo4j"
    neo4j_database: str = "neo4j"

    # OpenSearch (vector store)
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_ssl: bool = False
    opensearch_vector_index: str = "recipes-vector"
    search_preference: str = ""

    # Embeddings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # alt: thenlper/gte-small
    embedding_dimension: Optional[int] = None

    # Recipe fetch settings used by ingest.py. Do not break ingest unless you
    # enjoy archaeological debugging.
    recipe_http_timeout_secs: float = 15.0
    recipe_http_user_agent: str = "recipe-hybrid-rag/1.0 (+https://example.local; contact=dev-null)"

    # Local LLM runtime
    llm_runtime: str = "gguf"  # supported: gguf, mlx

    # Llama.cpp
    llama_model_path: str = "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"
    llama_ctx: int = 65536                  # Qwen = 65536/101000
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    llama_n_gpu_layers: int = 20             # -1 offloads all layers when GPU backend is available
    llama_n_batch: int = 256                 # prompt processing batch
    llama_n_ubatch: int = 256                # physical micro-batch; None to let llama.cpp choose
    llama_low_vram: bool = True              # reduce Metal VRAM usage

    # MLX local model directory. Relative paths are resolved under ~/models.
    mlx_model_path: str = "Qwen2.5-7B-Instruct-4bit"

    # External LLM (OpenAI-compatible endpoint). Used when USE_EXTERNAL_AI=true.
    llm_server_url: str = "http://127.0.0.1:8001/v1"
    llm_server_api_key: str = ""
    llm_server_model: str = "local-llm"
    external_base_url: str = "https://inference.do-ai.run/v1/chat/completions"
    external_model: str = "llama3-8b-instruct"

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8001


def load_settings() -> Settings:
    """Load settings from environment variables."""

    external_ai = os.getenv("USE_EXTERNAL_AI", "false").lower() in ("1", "true", "yes", "on")

    llm_runtime = _get_str("LLM_RUNTIME", Settings.llm_runtime).strip().lower()
    if llm_runtime not in {"gguf", "mlx"}:
        llm_runtime = Settings.llm_runtime

    llm_server_url = os.getenv("LLM_SERVER_URL", Settings.llm_server_url)
    if external_ai:
        llm_server_url = _get_str("EXTERNAL_LLM_URL", Settings.external_base_url)

    llm_server_api_key = os.getenv("LLM_SERVER_API_KEY", Settings.llm_server_api_key)
    if external_ai:
        llm_server_api_key = _get_str("EXTERNAL_LLM_API_KEY", "")

    llm_server_model = os.getenv("LLM_SERVER_MODEL", Settings.llm_server_model)
    if external_ai:
        llm_server_model = os.getenv("EXTERNAL_LLM_MODEL", Settings.external_model)

    llama_ctx = _get_int("LLAMA_CTX", Settings.llama_ctx)
    if external_ai:
        llama_ctx = _get_int("EXTERNAL_LLM_MAX_TOKENS", 262144)

    return Settings(
        neo4j_uri=_get_str("NEO4J_URI", Settings.neo4j_uri),
        neo4j_user=_get_str("NEO4J_USER", Settings.neo4j_user),
        neo4j_password=_get_str("NEO4J_PASSWORD", Settings.neo4j_password),
        neo4j_database=_get_str("NEO4J_DATABASE", Settings.neo4j_database),
        opensearch_host=_get_str("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_user=_get_str("OPENSEARCH_USER", Settings.opensearch_user),
        opensearch_password=_get_str("OPENSEARCH_PASS", Settings.opensearch_password),
        opensearch_ssl=_get_bool("OPENSEARCH_SSL", Settings.opensearch_ssl),
        opensearch_vector_index=_get_str("OPENSEARCH_VECTOR_INDEX", Settings.opensearch_vector_index),
        search_preference=_get_str("OPENSEARCH_SEARCH_PREFERENCE", Settings.search_preference),

        # Embeddings
        embedding_model=_get_str("EMBEDDING_MODEL", _get_str("EMBEDDING_MODEL", Settings.embedding_model)),
        embedding_dimension=_get_optional_int("EMBEDDING_DIMENSION", Settings.embedding_dimension),

        # Fetch Recipes
        recipe_http_timeout_secs=_get_float("RECIPE_HTTP_TIMEOUT_SECS", Settings.recipe_http_timeout_secs),
        recipe_http_user_agent=_get_str("RECIPE_HTTP_USER_AGENT", Settings.recipe_http_user_agent),

        # Local LLM runtime
        llm_runtime=llm_runtime,

        # LLaMA
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path.home() / "models" / Settings.llama_model_path),
        ),
        llama_ctx=llama_ctx,
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", _default_llama_n_gpu_layers()),
        llama_n_batch=_get_int("LLAMA_N_BATCH", Settings.llama_n_batch),
        llama_n_ubatch=_get_int("LLAMA_N_UBATCH", Settings.llama_n_ubatch or 0) or None,
        llama_low_vram=_get_bool("LLAMA_LOW_VRAM", Settings.llama_low_vram),

        # MLX
        mlx_model_path=os.getenv(
            "MLX_MODEL_PATH",
            str(Path.home() / "models" / Settings.mlx_model_path),
        ),

        # External LLM (OpenAI-compatible endpoint)
        llm_server_url=llm_server_url,
        llm_server_api_key=llm_server_api_key,
        llm_server_model=llm_server_model,

        # Server
        server_host=os.getenv("SERVER_HOST", Settings.server_host),
        server_port=_get_int("SERVER_PORT", Settings.server_port),
    )


__all__ = [
    "Settings",
    "load_settings",
    "_get_str",
    "_get_int",
    "_get_float",
    "_get_bool",
]
