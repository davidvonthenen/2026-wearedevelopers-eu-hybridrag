"""Configuration loading utilities for the RAG application."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class Settings:
    """Runtime configuration parameters for the local vector RAG workflow."""

    # OpenSearch
    opensearch_host: str = "127.0.0.1"
    opensearch_port: int = 9200
    opensearch_index: str = "bbc-vector-chunks"

    # Embeddings
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"


    # Llama.cpp
    llama_model_path: str = "Qwen2.5-7B-Instruct-1M-Q5_K_M.gguf"
    # Context window requested from llama.cpp. The Qwen model may support larger
    # contexts, but this project defaults to 65,536 tokens to keep memory use sane.
    llama_ctx: int = 65536
    llama_n_threads: int = max(1, (os.cpu_count() or 4) - 1)
    # Number of layers to offload to the GPU. Set to 0 for CPU-only execution.
    llama_n_gpu_layers: int = 20
    llama_n_batch: int = 256                 # llama.cpp prompt-processing batch size
    llama_n_ubatch: Optional[int] = 256      # physical micro-batch; None lets llama.cpp choose
    llama_low_vram: bool = True              # trade speed for lower VRAM usage on constrained GPUs


    # RAG
    rag_top_k: int = 3
    rag_num_candidates: int = 50



def _get_int(name: str, default_val: int) -> int:
    """Read an integer environment variable, falling back on invalid input."""
    v = os.getenv(name)
    if v is None or v == "":
        return default_val
    try:
        return int(v)
    except ValueError:
        return default_val


def _get_bool(name: str, default_val: bool) -> bool:
    """Read a boolean environment variable using common truthy strings."""
    v = os.getenv(name)
    if v is None:
        return default_val
    return v.lower() in ("1", "true", "yes", "on")


def load_settings(env_file: str | None = None) -> Settings:
    """Load runtime settings from environment variables and an optional .env file."""
    # Load an explicit .env first; otherwise check the current working directory
    # and then the project root inferred from this file location.
    if env_file:
        load_dotenv(env_file)
    else:
        for candidate in (Path(".env"), Path(__file__).resolve().parent.parent / ".env"):
            if candidate.exists():
                load_dotenv(str(candidate))
                break

    opensearch_port = _get_int("OPENSEARCH_PORT", Settings.opensearch_port)
    if opensearch_port != 9200:
        print("\n-----------------------------------------------------")
        print(f"WARNING: Using NON STANDARD OpenSearch Port: {opensearch_port}")
        print("Default OpenSearch Port is 9200")
        print("-----------------------------------------------------\n")

    return Settings(
        opensearch_host=os.getenv("OPENSEARCH_HOST", Settings.opensearch_host),
        opensearch_port=_get_int("OPENSEARCH_PORT", Settings.opensearch_port),
        opensearch_index=os.getenv("OPENSEARCH_INDEX", Settings.opensearch_index),
        embedding_model=os.getenv("EMBEDDING_MODEL", Settings.embedding_model),
        llama_model_path=os.getenv(
            "LLAMA_MODEL_PATH",
            str(Path.home() / "models" / Settings.llama_model_path),
        ),
        llama_ctx=_get_int("LLAMA_CTX", Settings.llama_ctx),
        llama_n_threads=_get_int("LLAMA_N_THREADS", Settings.llama_n_threads),
        llama_n_gpu_layers=_get_int("LLAMA_N_GPU_LAYERS", Settings.llama_n_gpu_layers),
        llama_n_batch=_get_int("LLAMA_N_BATCH", Settings.llama_n_batch),
        llama_n_ubatch=_get_int("LLAMA_N_UBATCH", Settings.llama_n_ubatch or 0) or None,
        llama_low_vram=_get_bool("LLAMA_LOW_VRAM", Settings.llama_low_vram),
        rag_top_k=_get_int("RAG_TOP_K", Settings.rag_top_k),
        rag_num_candidates=_get_int("RAG_NUM_CANDIDATES", Settings.rag_num_candidates),
    )


__all__ = ["Settings", "load_settings"]
