"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

from dataclasses import dataclass
import os
import platform
from pathlib import Path
from typing import Optional


_INSTACLUSTR_OPENSEARCH_HOST = "load-balancer.8a34294b4561402299e0c2e354bff492.cnodes.io"
_INSTACLUSTR_MIN_USER_INDEX = 1
_INSTACLUSTR_MAX_USER_INDEX = 39


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
    """Connection and model settings for recipe BM25/vector retrieval."""

    # OpenSearch BM25/vector store
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_user: str = ""
    opensearch_password: str = ""
    opensearch_ssl: bool = False
    opensearch_vector_index: str = "recipes-vector"
    opensearch_bm25_index: str = "recipes-bm25"
    search_preference: str = ""

    # Embeddings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"  # alt: thenlper/gte-small
    embedding_dimension: Optional[int] = None

    # NER service
    ner_service_url: str = "http://127.0.0.1:8000/ner"
    ner_timeout_secs: float = 10.0

    # Recipe fetch settings used by ingest.py.
    recipe_http_timeout_secs: float = 15.0
    recipe_http_user_agent: str = "recipe-bm25-vector-rag/1.0 (+https://example.local; contact=dev-null)"

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

    user_index = _get_str("USER_INDEX")
    user_password = _get_str("USER_PASSWORD")

    opensearch_host = _get_str("OPENSEARCH_HOST", Settings.opensearch_host)
    opensearch_port = _get_int("OPENSEARCH_PORT", Settings.opensearch_port)
    opensearch_user = _get_str("OPENSEARCH_USER", Settings.opensearch_user)
    opensearch_password = _get_str("OPENSEARCH_PASS", Settings.opensearch_password)
    opensearch_ssl = _get_bool("OPENSEARCH_SSL", Settings.opensearch_ssl)
    opensearch_vector_index = _get_str("OPENSEARCH_VECTOR_INDEX", Settings.opensearch_vector_index)
    opensearch_bm25_index = _get_str("OPENSEARCH_BM25_INDEX", Settings.opensearch_bm25_index)

    if user_index or user_password:
        if not user_index or not user_password:
            raise ValueError("USER_INDEX and USER_PASSWORD must both be set to enable Instaclustr mode.")

        try:
            index = int(user_index)
        except ValueError as exc:
            raise ValueError("USER_INDEX must be an integer from 1 to 35.") from exc

        if not _INSTACLUSTR_MIN_USER_INDEX <= index <= _INSTACLUSTR_MAX_USER_INDEX:
            raise ValueError("USER_INDEX must be an integer from 1 to 35.")

        opensearch_host = _INSTACLUSTR_OPENSEARCH_HOST
        opensearch_port = Settings.opensearch_port
        opensearch_user = f"User{index}"
        opensearch_password = user_password
        opensearch_ssl = True
        opensearch_vector_index = f"recipes-vector-{index}"
        opensearch_bm25_index = f"recipes-bm25-{index}"

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
        opensearch_host=opensearch_host,
        opensearch_port=opensearch_port,
        opensearch_user=opensearch_user,
        opensearch_password=opensearch_password,
        opensearch_ssl=opensearch_ssl,
        opensearch_vector_index=opensearch_vector_index,
        opensearch_bm25_index=opensearch_bm25_index,
        search_preference=_get_str("OPENSEARCH_SEARCH_PREFERENCE", Settings.search_preference),

        # Embeddings
        embedding_model=_get_str("EMBEDDING_MODEL", _get_str("EMBEDDING_MODEL", Settings.embedding_model)),
        embedding_dimension=_get_optional_int("EMBEDDING_DIMENSION", Settings.embedding_dimension),

        # NER Service
        ner_service_url=_get_str("NER_SERVICE_URL", Settings.ner_service_url),
        ner_timeout_secs=_get_float("NER_TIMEOUT_SECS", Settings.ner_timeout_secs),

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
