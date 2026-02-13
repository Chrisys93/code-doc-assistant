"""
Configuration module — resolves deployment settings from environment variables.

This mirrors the Helm _helpers.tpl logic for Docker Compose deployments,
ensuring consistent behaviour across both deployment methods.
"""

import os
import logging

logger = logging.getLogger(__name__)

# --- Model Tier Resolution ---
MODEL_TIER = os.getenv("MODEL_TIER", "full")

MODEL_TIER_MAP = {
    "full": "mistral-nemo",
    "balanced": "qwen2.5-coder:7b",
    "lightweight": "phi3.5",
}

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", MODEL_TIER_MAP.get(MODEL_TIER, "mistral-nemo"))

# --- Embedding Model Resolution ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

EMBEDDING_DIMENSION_MAP = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "mxbai-embed-large": 1024,
}

EMBEDDING_DIMENSION = int(
    os.getenv("EMBEDDING_DIMENSION", str(EMBEDDING_DIMENSION_MAP.get(EMBEDDING_MODEL, 768)))
)

# --- Service Endpoints ---
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://localhost:8000")

# --- Application Settings ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

# --- Chunking Strategy ---
# "ast" = AST-aware via tree-sitter (higher quality, more CPU/memory)
# "text" = SentenceSplitter (lighter, always works)
# Derived from model tier in Helm; lightweight tier defaults to "text"
_tier = os.getenv("MODEL_TIER", "full")
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "text" if _tier == "lightweight" else "ast")

# --- Chroma Collection Name ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "codebase")

# --- Configure logging ---
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def log_config():
    """Log the active configuration at startup."""
    logger.info("=== Code Documentation Assistant Configuration ===")
    logger.info(f"  Model Tier:        {MODEL_TIER}")
    logger.info(f"  LLM Model:         {OLLAMA_MODEL}")
    logger.info(f"  Embedding Model:   {EMBEDDING_MODEL}")
    logger.info(f"  Embedding Dim:     {EMBEDDING_DIMENSION}")
    logger.info(f"  Ollama Host:       {OLLAMA_HOST}")
    logger.info(f"  ChromaDB Host:     {CHROMA_HOST}")
    logger.info(f"  Chunk Size:        {CHUNK_SIZE}")
    logger.info(f"  Chunk Overlap:     {CHUNK_OVERLAP}")
    logger.info(f"  Chunking Strategy: {CHUNKING_STRATEGY}")
    logger.info(f"  Top-K Retrieval:   {TOP_K}")
    logger.info("================================================")
