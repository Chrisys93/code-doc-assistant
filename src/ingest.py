"""
Codebase ingestion pipeline.

Handles: repo cloning → file discovery → AST-aware chunking → embedding → vector storage.

Uses LlamaIndex's CodeSplitter (tree-sitter) for AST-aware chunking with
a fixed-window fallback for files that can't be parsed.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

from git import Repo as GitRepo
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

from src.config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKING_STRATEGY,
    TOP_K,
)
from src.vector_store import ChromaVectorStoreImpl

logger = logging.getLogger(__name__)

# File extensions to ingest, mapped to tree-sitter language identifiers
LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".r": "r",
    ".R": "r",
}

# Also ingest documentation and config files (using text splitter)
TEXT_EXTENSIONS = {
    ".md", ".txt", ".rst", ".yaml", ".yml", ".toml",
    ".json", ".xml", ".html", ".css", ".sql", ".sh",
    ".bash", ".dockerfile", ".env", ".cfg", ".ini", ".conf",
}

# Directories to skip during file discovery
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".egg-info", ".eggs", "vendor", "target",
}


def clone_repo(repo_url: str, target_dir: Optional[str] = None) -> str:
    """Clone a git repository and return the local path."""
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="code-doc-")
    logger.info(f"Cloning {repo_url} to {target_dir}")
    GitRepo.clone_from(repo_url, target_dir, depth=1)
    logger.info("Clone complete")
    return target_dir


def discover_files(repo_path: str) -> list[dict]:
    """
    Walk the repo and return a list of files to ingest.

    Returns dicts with: path, relative_path, extension, language (if code)
    """
    files = []
    repo_root = Path(repo_path)

    for root, dirs, filenames in os.walk(repo_root):
        # Prune skip directories in-place
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = Path(root) / fname
            ext = fpath.suffix.lower()
            rel_path = str(fpath.relative_to(repo_root))

            if ext in LANGUAGE_MAP:
                files.append({
                    "path": str(fpath),
                    "relative_path": rel_path,
                    "extension": ext,
                    "language": LANGUAGE_MAP[ext],
                    "type": "code",
                })
            elif ext in TEXT_EXTENSIONS:
                files.append({
                    "path": str(fpath),
                    "relative_path": rel_path,
                    "extension": ext,
                    "language": None,
                    "type": "text",
                })

    logger.info(
        f"Discovered {len(files)} files "
        f"({sum(1 for f in files if f['type'] == 'code')} code, "
        f"{sum(1 for f in files if f['type'] == 'text')} text/config)"
    )
    return files


def load_and_chunk_files(files: list[dict]) -> list:
    """
    Load files and split into chunks.

    Code files: AST-aware chunking via tree-sitter (CodeSplitter)
    Text files: Sentence-based chunking (SentenceSplitter) as fallback
    """
    documents = []

    # Read all files into LlamaIndex Documents with metadata
    for file_info in files:
        try:
            with open(file_info["path"], "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            if not content.strip():
                continue

            doc = Document(
                text=content,
                metadata={
                    "file_path": file_info["relative_path"],
                    "file_type": file_info["type"],
                    "language": file_info.get("language", "unknown"),
                    "extension": file_info["extension"],
                },
            )
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Failed to read {file_info['path']}: {e}")

    logger.info(f"Loaded {len(documents)} documents")

    # Split code files with CodeSplitter, text files with SentenceSplitter
    code_docs = [d for d in documents if d.metadata.get("file_type") == "code"]
    text_docs = [d for d in documents if d.metadata.get("file_type") == "text"]

    all_nodes = []

    # AST-aware chunking for code (if strategy allows)
    if code_docs and CHUNKING_STRATEGY == "ast":
        # Group by language for correct parser selection
        by_language = {}
        for doc in code_docs:
            lang = doc.metadata.get("language", "python")
            by_language.setdefault(lang, []).append(doc)

        for language, lang_docs in by_language.items():
            try:
                code_splitter = CodeSplitter(
                    language=language,
                    chunk_lines=40,
                    chunk_lines_overlap=5,
                    max_chars=CHUNK_SIZE,
                )
                nodes = code_splitter.get_nodes_from_documents(lang_docs)
                all_nodes.extend(nodes)
                logger.info(f"  {language}: {len(lang_docs)} files → {len(nodes)} chunks (AST)")
            except Exception as e:
                # Fallback to sentence splitter if tree-sitter fails
                logger.warning(f"  {language}: AST parsing failed ({e}), using text fallback")
                fallback = SentenceSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                )
                nodes = fallback.get_nodes_from_documents(lang_docs)
                all_nodes.extend(nodes)
                logger.info(f"  {language}: {len(lang_docs)} files → {len(nodes)} chunks (fallback)")
    elif code_docs:
        # Text-based chunking for code (lightweight tier or explicit config)
        logger.info(f"  Using text-based chunking for code (strategy={CHUNKING_STRATEGY})")
        text_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        nodes = text_splitter.get_nodes_from_documents(code_docs)
        all_nodes.extend(nodes)
        logger.info(f"  code: {len(code_docs)} files → {len(nodes)} chunks (text)")

    # Text-based chunking for docs/config
    if text_docs:
        text_splitter = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        nodes = text_splitter.get_nodes_from_documents(text_docs)
        all_nodes.extend(nodes)
        logger.info(f"  text/config: {len(text_docs)} files → {len(nodes)} chunks")

    logger.info(f"Total chunks: {len(all_nodes)}")
    return all_nodes


def build_index(
    nodes: list,
    vector_store_impl: ChromaVectorStoreImpl,
) -> VectorStoreIndex:
    """
    Embed chunks and store in the vector database.

    Returns a VectorStoreIndex ready for querying.
    """
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )

    vector_store = vector_store_impl.get_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info(f"Building index with {len(nodes)} chunks using {EMBEDDING_MODEL}...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    logger.info("Index built successfully")
    return index


def load_existing_index(
    vector_store_impl: ChromaVectorStoreImpl,
) -> VectorStoreIndex:
    """Load an existing index from the vector store (no re-ingestion)."""
    embed_model = OllamaEmbedding(
        model_name=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )

    vector_store = vector_store_impl.get_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    logger.info(f"Loaded existing index ({vector_store_impl.document_count} documents)")
    return index


def ingest_codebase(
    repo_path: str,
    vector_store_impl: ChromaVectorStoreImpl,
    reset: bool = True,
) -> VectorStoreIndex:
    """
    Full ingestion pipeline: discover → chunk → embed → store.

    Args:
        repo_path: Local path to the codebase
        vector_store_impl: Vector store to write to
        reset: If True, clear existing vectors before ingesting

    Returns:
        VectorStoreIndex ready for querying
    """
    if reset:
        vector_store_impl.reset()

    files = discover_files(repo_path)
    if not files:
        raise ValueError(f"No ingestible files found in {repo_path}")

    nodes = load_and_chunk_files(files)
    index = build_index(nodes, vector_store_impl)
    return index
