"""
Vector store abstraction layer.

ChromaDB is the concrete implementation. The abstraction preserves the
ability to swap in Qdrant (for production scale) without changing
application code — see VectorStoreBase.

HNSW tuning
───────────
Collection metadata is set at creation time from config.py values,
which in turn read from environment variables (set by Helm or docker-compose).
deploymentTarget=local reduces searchEf to 20 (faster, acceptable for
single-user dev). deploymentTarget=cluster uses the full configured value.

See ARCHITECTURE.md Phase 14 for the full HNSW parameter rationale.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import (
    CHROMA_HOST,
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    CHROMA_HNSW_SPACE,
    CHROMA_HNSW_M,
    CHROMA_HNSW_CONSTRUCTION_EF,
    CHROMA_HNSW_SEARCH_EF,
)

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

    Connects to a ChromaDB HTTP server and provides a LlamaIndex-compatible
    vector store for the RAG pipeline.

    HNSW parameters are applied at collection creation time and cannot be
    changed without resetting (deleting and recreating) the collection.
    If you change HNSW params and want them applied, run ingestion with reset=True.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        self._host = host or CHROMA_HOST
        self._collection_name = collection_name or COLLECTION_NAME

        # Parse host into hostname and port
        clean = self._host.replace("http://", "").replace("https://", "")
        parts = clean.split(":")
        hostname = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 8000

        logger.info(f"Connecting to ChromaDB at {hostname}:{port}")
        self._client = chromadb.HttpClient(host=hostname, port=port)

        self._collection = self._get_or_create_collection()
        logger.info(
            f"ChromaDB collection '{self._collection_name}' ready "
            f"({self._collection.count()} existing documents)"
        )

    def _hnsw_metadata(self) -> dict:
        """
        Build the ChromaDB collection metadata dict from config values.

        All four HNSW parameters are set explicitly — previously only
        hnsw:space was set, leaving M, construction_ef, and search_ef at
        ChromaDB's global defaults (not tuned for this workload).

        Parameters:
          hnsw:space          → distance metric (cosine recommended for embeddings)
          hnsw:M              → bidirectional links per node; higher = better recall, more memory
          hnsw:construction_ef → candidate list at build time; higher = better recall, slower build
          hnsw:search_ef      → candidate list at query time; higher = better recall, slower query
                                 Reduced to 20 when DEPLOYMENT_TARGET=local (faster, acceptable
                                 for single-user dev).
        """
        return {
            "hnsw:space":           CHROMA_HNSW_SPACE,
            "hnsw:M":               CHROMA_HNSW_M,
            "hnsw:construction_ef": CHROMA_HNSW_CONSTRUCTION_EF,
            "hnsw:search_ef":       CHROMA_HNSW_SEARCH_EF,
        }

    def _get_or_create_collection(self):
        meta = self._hnsw_metadata()
        logger.info(
            f"ChromaDB HNSW config: space={meta['hnsw:space']} "
            f"M={meta['hnsw:M']} "
            f"construction_ef={meta['hnsw:construction_ef']} "
            f"search_ef={meta['hnsw:search_ef']}"
        )
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata=meta,
        )

    def get_vector_store(self) -> ChromaVectorStore:
        """Return a LlamaIndex ChromaVectorStore wrapping our collection."""
        return ChromaVectorStore(chroma_collection=self._collection)

    def reset(self) -> None:
        """Delete and recreate the collection with current HNSW params."""
        logger.warning(f"Resetting collection '{self._collection_name}'")
        self._client.delete_collection(self._collection_name)
        self._collection = self._get_or_create_collection()
        logger.info("Collection reset complete")

    @property
    def document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()


def get_vector_store(host: Optional[str] = None) -> ChromaVectorStoreImpl:
    """Factory function — returns the active vector store implementation."""
    return ChromaVectorStoreImpl(host=host)
