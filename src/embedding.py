"""
embedding.py -- the single place the embedding model is constructed.

Ingestion (ingest.py) and query-time retrieval (tools.py) both embed text and MUST use the
same model with the same settings. Building the client in one place guarantees that, and
puts the one deployment decision that matters -- where the embedding model runs -- behind
one setting.

Placement (EMBED_NUM_GPU, default 0)
------------------------------------
  0   run the embedding model on CPU (default). On a single, nearly full GPU, loading
      nomic-embed-text next to llama-server drove llama-server's decode speed from
      ~140-170 tok/s to 0.5-7 tok/s until the server was restarted (measured twice on a
      12 GB laptop GPU: VRAM 11.6 -> 11.8 GB of 12.2 GB). Results do not depend on the
      placement; only latency does. With num_gpu=0 Ollama reports size_vram=0 and the
      generation speed is unchanged. A 137M-parameter model is fast enough on CPU for
      single-query retrieval.
 -1   let Ollama decide (GPU if it fits). Use on hardware with VRAM to spare, or for a bulk
      ingest while no other model is resident on the GPU.
 >0   offload that many layers.

Task prefixes ("search_query: " / "search_document: " for nomic-embed-text) are deliberately
NOT added here: collections already in Chroma were embedded without them, and query and
index text must be embedded identically. Changing that needs a re-index and a measured
comparison -- see future_directions.md.
"""
from __future__ import annotations

import os
from functools import lru_cache

try:  # inside the app image these always exist
    from config import EMBEDDING_MODEL, OLLAMA_HOST
except ImportError:  # standalone use (tests, scripts)
    EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")


def embed_options() -> dict:
    """Ollama `options` for embedding requests, from EMBED_NUM_GPU (see module docstring)."""
    n = int(os.environ.get("EMBED_NUM_GPU", "0"))
    return {} if n < 0 else {"num_gpu": n}


@lru_cache(maxsize=1)
def get_embed_model():
    """The shared LlamaIndex embedding client (Ollama), configured once per process."""
    from llama_index.embeddings.ollama import OllamaEmbedding

    return OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
        # llama-index forwards this dict as the Ollama request's `options`
        ollama_additional_kwargs=embed_options(),
    )
