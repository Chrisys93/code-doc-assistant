"""
Vector store abstraction layer.

ChromaDB is the concrete implementation. The abstraction preserves the
ability to swap in FAISS (for LSH at scale) or Qdrant (for production)
without changing application code.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import CHROMA_HOST, COLLECTION_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """Abstract base for vector store implementations."""

    @abstractmethod
    def get_vector_store(self):
        """Return a LlamaIndex-compatible vector store instance."""
        ...

    @abstractmethod
    def reset(self):
        """Clear all stored vectors (used during re-ingestion)."""
        ...


class ChromaVectorStoreImpl(VectorStoreBase):
    """
    ChromaDB implementation of the vector store.

    Connects to a ChromaDB instance (local or remote) and provides
    a LlamaIndex-compatible vector store for the RAG pipeline.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self._host = host or CHROMA_HOST
        self._collection_name = collection_name or COLLECTION_NAME

        # Parse host into hostname and port
        # Expected format: http://hostname:port
        clean = self._host.replace("http://", "").replace("https://", "")
        parts = clean.split(":")
        hostname = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8000

        logger.info(f"Connecting to ChromaDB at {hostname}:{port}")
        self._client = chromadb.HttpClient(host=hostname, port=port)

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{self._collection_name}' ready "
            f"({self._collection.count()} existing documents)"
        )

    def get_vector_store(self) -> ChromaVectorStore:
        """Return a LlamaIndex ChromaVectorStore wrapping our collection."""
        return ChromaVectorStore(chroma_collection=self._collection)

    def reset(self):
        """Delete and recreate the collection."""
        logger.warning(f"Resetting collection '{self._collection_name}'")
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()


def get_vector_store(host: Optional[str] = None) -> ChromaVectorStoreImpl:
    """Factory function — returns the active vector store implementation."""
    return ChromaVectorStoreImpl(host=host)
