"""
Configuration module — resolves deployment settings from environment variables.

This mirrors the Helm _helpers.tpl logic for Docker Compose deployments,
ensuring consistent behaviour across both deployment methods.

All values can be overridden via environment variables. The defaults here
match the helm values.yaml defaults and the docker-compose_dev.yml defaults.
"""

import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deployment target
# ---------------------------------------------------------------------------
# "local"   → developer laptop / single-node; no resource limits enforced;
#             lighter HNSW searchEf; host-path volumes
# "cluster" → production k8s; full HNSW params; PVCs
DEPLOYMENT_TARGET = os.getenv("DEPLOYMENT_TARGET", "local")

# ---------------------------------------------------------------------------
# Inference backend
# ---------------------------------------------------------------------------
# "ollama"   → Ollama REST API (default). Model management included.
#              Best for: full/balanced tiers, ease of use.
# "vllm"     → vLLM OpenAI-compatible API. GPU required.
#              Best for: high-throughput serving at scale.
# "llamacpp" → llama-server OpenAI-compatible API. CPU-native GGUF.
#              Best for: lightweight/minimal tiers on constrained machines.
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "ollama").lower()

# ---------------------------------------------------------------------------
# Model tier + quantisation
# ---------------------------------------------------------------------------
MODEL_TIER = os.getenv("MODEL_TIER", "full")

# Base model per tier — matches _helpers.tpl baseModel resolution
_MODEL_TIER_BASE = {
    "full":        "mistral-nemo:12b-instruct",
    "balanced":    "deepseek-coder-v2:16b-lite-instruct",
    "lightweight": "phi3.5",
    "minimal":     "qwen2.5-coder:3b-instruct",
}

# Quantisation suffix — appended for full/balanced; suppressed for lightweight/minimal
# (those tiers use Ollama's built-in default quantisation, already Q4)
QUANTISATION = os.getenv("QUANTISATION", "q4_K_M")

def _resolve_ollama_model() -> str:
    """Compose the final Ollama model tag from MODEL_TIER + QUANTISATION."""
    base = _MODEL_TIER_BASE.get(MODEL_TIER, "mistral-nemo:12b-instruct")
    if MODEL_TIER in ("lightweight", "minimal"):
        return base
    if QUANTISATION == "fp16":
        return base
    return f"{base}-{QUANTISATION}"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", _resolve_ollama_model())

# llama.cpp / llama-server
LLAMACPP_HOST = os.getenv("LLAMACPP_HOST", "http://localhost:8081")
LLAMACPP_MODEL = os.getenv("LLAMACPP_MODEL", "qwen2.5-coder-3b-instruct-q4_k_m")

# vLLM
VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:8080")
VLLM_MODEL = os.getenv("VLLM_MODEL", OLLAMA_MODEL)

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------
# WARNING: Changing this after ingestion requires full re-ingestion.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

_EMBEDDING_DIMENSION_MAP = {
    "nomic-embed-text":  768,
    "all-minilm":        384,
    "mxbai-embed-large": 1024,
}

EMBEDDING_DIMENSION = int(
    os.getenv(
        "EMBEDDING_DIMENSION",
        str(_EMBEDDING_DIMENSION_MAP.get(EMBEDDING_MODEL, 768)),
    )
)

# ---------------------------------------------------------------------------
# Service endpoints
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://localhost:8000")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

# ---------------------------------------------------------------------------
# Graph DB (Kuzu)
# ---------------------------------------------------------------------------
GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "true").lower() == "true"
GRAPH_PATH = os.getenv("GRAPH_PATH", "/data/graph")
GRAPH_CO_CHANGE_COMMITS = int(os.getenv("GRAPH_CO_CHANGE_COMMITS", "100"))

# ---------------------------------------------------------------------------
# ChromaDB HNSW tuning
# ---------------------------------------------------------------------------
# Applied at collection creation time in vector_store.py.
# deploymentTarget=local overrides searchEf to 20 (faster, less accurate —
# acceptable for single-user dev). All other params respected from env.
CHROMA_HNSW_SPACE = os.getenv("CHROMA_HNSW_SPACE", "cosine")
CHROMA_HNSW_M = int(os.getenv("CHROMA_HNSW_M", "16"))
CHROMA_HNSW_CONSTRUCTION_EF = int(os.getenv("CHROMA_HNSW_CONSTRUCTION_EF", "100"))
CHROMA_HNSW_SEARCH_EF = int(
    os.getenv(
        "CHROMA_HNSW_SEARCH_EF",
        "20" if DEPLOYMENT_TARGET == "local" else "50",
    )
)

# ---------------------------------------------------------------------------
# Application settings
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Chunking strategy:
# "ast"  → AST-aware via tree-sitter (higher quality, more CPU/memory)
# "text" → SentenceSplitter (lighter, always works)
# minimal tier defaults to "text" (lower resource usage, matches _helpers.tpl)
_default_chunking = "text" if MODEL_TIER in ("lightweight", "minimal") else "ast"
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", _default_chunking)

# ---------------------------------------------------------------------------
# Chroma collection name
# ---------------------------------------------------------------------------
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "codebase")

# ---------------------------------------------------------------------------
# MCP server config (for reference by tools.py)
# ---------------------------------------------------------------------------
MCP_FILESYSTEM_ENABLED = os.getenv("MCP_FILESYSTEM_ENABLED", "false").lower() == "true"
MCP_FILESYSTEM_URL = os.getenv("MCP_FILESYSTEM_URL", "http://localhost:3000")
MCP_GITHUB_ENABLED = os.getenv("MCP_GITHUB_ENABLED", "false").lower() == "true"
MCP_GITHUB_URL = os.getenv("MCP_GITHUB_URL", "http://localhost:3001")
MCP_SLACK_ENABLED = os.getenv("MCP_SLACK_ENABLED", "false").lower() == "true"
MCP_SLACK_URL = os.getenv("MCP_SLACK_URL", "http://localhost:3002")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def log_config() -> None:
    """Log the active configuration at startup."""
    logger.info("=== Code Documentation Assistant Configuration ===")
    logger.info(f"  Deployment Target:  {DEPLOYMENT_TARGET}")
    logger.info(f"  Inference Backend: {INFERENCE_BACKEND}")
    logger.info(f"  Model Tier:        {MODEL_TIER}")
    logger.info(f"  Quantisation:      {QUANTISATION}")
    if INFERENCE_BACKEND == "llamacpp":
        logger.info(f"  llama-server Host: {LLAMACPP_HOST}")
        logger.info(f"  llama-server Model:{LLAMACPP_MODEL}")
    elif INFERENCE_BACKEND == "vllm":
        logger.info(f"  vLLM Host:         {VLLM_HOST}")
        logger.info(f"  vLLM Model:        {VLLM_MODEL}")
    else:
        logger.info(f"  Ollama Host:       {OLLAMA_HOST}")
        logger.info(f"  Ollama Model:      {OLLAMA_MODEL}")
    logger.info(f"  Embedding Model:   {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION}d)")
    logger.info(f"  ChromaDB Host:     {CHROMA_HOST}")
    logger.info(f"  HNSW:              space={CHROMA_HNSW_SPACE} M={CHROMA_HNSW_M} "
                f"ef_construction={CHROMA_HNSW_CONSTRUCTION_EF} ef_search={CHROMA_HNSW_SEARCH_EF}")
    logger.info(f"  Graph Enabled:     {GRAPH_ENABLED} → {GRAPH_PATH}")
    logger.info(f"  Chunk Size:        {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    logger.info(f"  Chunking Strategy: {CHUNKING_STRATEGY}")
    logger.info(f"  Top-K Retrieval:   {TOP_K}")
    logger.info(f"  MCP: filesystem={MCP_FILESYSTEM_ENABLED} "
                f"github={MCP_GITHUB_ENABLED} slack={MCP_SLACK_ENABLED}")
    logger.info("=================================================")
