"""
End-to-end pipeline validation.

Runs the full ingestion and retrieval pipeline against this repo's own
source code, using ChromaDB in embedded (in-process) mode.

In this test environment we cannot reach Ollama, so:
  - Embeddings: uses ChromaDB's built-in or a simple local embedding function
  - LLM generation: mocked — we validate retrieval quality, not generation
  - ChromaDB: in-process (no server needed)

Usage:
    python -m tests.test_pipeline
    MODEL_TIER=lightweight python -m tests.test_pipeline
"""

import sys
import os
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import (
    MODEL_TIER, OLLAMA_MODEL, EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, log_config,
)
from src.ingest import discover_files, load_and_chunk_files


# ============================================================
# Simple local embedding for testing (no Ollama dependency)
# ============================================================
from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class LocalTestEmbedding(BaseEmbedding):
    """
    Lightweight bag-of-words embedding for pipeline testing.
    
    NOT for production — this is a deterministic, vocabulary-based
    embedding that produces consistent vectors for similar text.
    Uses character trigram hashing.
    """
    _dim: int = PrivateAttr(default=384)
    
    def __init__(self, dim: int = 384, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim
    
    @classmethod
    def class_name(cls) -> str:
        return "LocalTestEmbedding"
    
    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)
    
    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)
    
    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)
    
    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)
    
    def _embed(self, text: str) -> list[float]:
        """Character trigram hashing into a fixed-dim vector."""
        vec = np.zeros(self._dim, dtype=np.float64)
        text = text.lower()
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            idx = hash(trigram) % self._dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


# ============================================================
# Pipeline validation
# ============================================================
def main():
    log_config()
    
    print("\n" + "=" * 60)
    print("CODE DOCUMENTATION ASSISTANT — Pipeline Validation")
    print("=" * 60)
    
    # --- Step 1: Config resolution ---
    print(f"\n[1/6] Configuration")
    print(f"  Model tier: {MODEL_TIER} → {OLLAMA_MODEL}")
    print(f"  Embedding:  {EMBEDDING_MODEL} ({EMBEDDING_DIMENSION}d)")
    print(f"  Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print(f"  Top-K:      {TOP_K}")
    
    # --- Step 2: File discovery ---
    print(f"\n[2/6] File Discovery")
    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = discover_files(repo_path)
    
    code_files = [f for f in files if f["type"] == "code"]
    text_files = [f for f in files if f["type"] == "text"]
    print(f"  Code files: {len(code_files)}")
    print(f"  Text files: {len(text_files)}")
    
    languages = set(f["language"] for f in code_files if f["language"])
    print(f"  Languages:  {', '.join(sorted(languages))}")
    
    # --- Step 3: Chunking ---
    print(f"\n[3/6] Chunking")
    nodes = load_and_chunk_files(files)
    print(f"  Total chunks: {len(nodes)}")
    
    # Show chunk distribution
    by_file = {}
    for node in nodes:
        fp = node.metadata.get("file_path", "unknown")
        by_file[fp] = by_file.get(fp, 0) + 1
    for fp, count in sorted(by_file.items()):
        print(f"    {fp}: {count} chunks")
    
    # --- Step 4: Embedding + ChromaDB storage ---
    print(f"\n[4/6] Embedding & Storage (ChromaDB in-process)")
    
    embed_model = LocalTestEmbedding(dim=384)
    
    # Create ChromaDB in-process
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.get_or_create_collection(
        name="pipeline_test",
        metadata={"hnsw:space": "cosine"},
    )
    
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=False,
    )
    
    print(f"  Stored {chroma_collection.count()} chunks in ChromaDB")
    
    # --- Step 5: Retrieval testing ---
    print(f"\n[5/6] Retrieval Testing")
    
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    
    test_queries = [
        ("What models are available?", ["config.py"]),
        ("How does file discovery work?", ["ingest.py"]),
        ("How is ChromaDB connected?", ["vector_store.py"]),
        ("What does the Streamlit UI look like?", ["app.py"]),
        ("How are queries processed?", ["query_engine.py"]),
        ("What is the RAG prompt template?", ["query_engine.py"]),
    ]
    
    passed = 0
    for query_text, expected_files in test_queries:
        results = retriever.retrieve(query_text)
        retrieved_files = [r.metadata.get("file_path", "?") for r in results]
        
        # Check if any expected file appears in retrieved results
        hit = any(
            any(exp in rf for rf in retrieved_files)
            for exp in expected_files
        )
        
        status = "✅" if hit else "⚠️ "
        if hit:
            passed += 1
        
        print(f"  {status} \"{query_text}\"")
        print(f"       Retrieved: {retrieved_files[:3]}")
        if results:
            print(f"       Top score: {results[0].score:.3f}" if results[0].score else "")
    
    print(f"\n  Retrieval accuracy: {passed}/{len(test_queries)} queries hit expected files")
    
    # --- Step 6: Summary ---
    print(f"\n[6/6] Pipeline Summary")
    print(f"  {'=' * 50}")
    print(f"  Files discovered:  {len(files)}")
    print(f"  Chunks generated:  {len(nodes)}")
    print(f"  Chunks stored:     {chroma_collection.count()}")
    print(f"  Retrieval tested:  {len(test_queries)} queries")
    print(f"  Retrieval hits:    {passed}/{len(test_queries)}")
    print(f"  {'=' * 50}")
    
    # Note what's NOT tested
    print(f"\n  ⚠️  Not tested (requires running services):")
    print(f"     - Ollama LLM generation (needs Ollama server)")
    print(f"     - Ollama embeddings (using local test embeddings instead)")
    print(f"     - Streamlit UI (needs browser)")
    print(f"     - Docker Compose / Helm deployment")
    
    if passed >= len(test_queries) * 0.5:
        print(f"\n  ✅ Pipeline validation PASSED")
        return 0
    else:
        print(f"\n  ❌ Pipeline validation FAILED — retrieval quality too low")
        return 1


if __name__ == "__main__":
    sys.exit(main())
