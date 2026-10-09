"""
Codebase ingestion pipeline.

Handles: repo cloning → file discovery → AST-aware chunking → embedding
         → vector storage → graph construction (dependency + co-change)

The graph construction step runs after the vector index is built, using the
same file list and tree-sitter parse. It writes to a Kuzu embedded graph DB
(a directory on disk, no new services) alongside the ChromaDB volume.

Graph construction is opt-in via GRAPH_ENABLED env var (default: true when
kuzu is installed). It does not affect the vector index or query pipeline —
graph tools are additive to the existing tool registry.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import Optional

from git import Repo as GitRepo
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter

from config import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKING_STRATEGY,
    TOP_K,
)
from vector_store import ChromaVectorStoreImpl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graph config — resolved from environment
# ---------------------------------------------------------------------------

GRAPH_ENABLED = os.environ.get("GRAPH_ENABLED", "true").lower() == "true"
GRAPH_PATH = os.environ.get("GRAPH_PATH", "/data/graph")
GRAPH_CO_CHANGE_COMMITS = int(os.environ.get("GRAPH_CO_CHANGE_COMMITS", "100"))

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
    # Languages with no tree-sitter grammar here (Stata, SAS, SPSS, Julia): plain-text chunking.
    ".do", ".ado", ".mata", ".sas", ".sps", ".jl",
}

# Extensions that are never source: not reported as "skipped code" by skipped_source_files().
_NON_SOURCE_EXTENSIONS = {
    "", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".pdf", ".zip", ".gz", ".tar", ".tgz",
    ".lock", ".csv", ".tsv", ".dta", ".rds", ".rdata", ".xlsx", ".xls", ".docx", ".pptx",
    ".parquet", ".pkl", ".bin", ".pt", ".onnx", ".gguf", ".woff", ".woff2", ".ttf", ".eot",
    ".mp3", ".mp4", ".mov", ".gitignore", ".gitattributes", ".log", ".out", ".pyc", ".so", ".dll",
}

# Directories to skip during file discovery
SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".egg-info", ".eggs", "vendor", "target",
}


def clone_repo(repo_url: str, target_dir: Optional[str] = None, branch: Optional[str] = None) -> str:
    """
    Clone a git repository and return the local path.

    branch: optional branch/tag/ref to clone (e.g. "dev"). Without it,
    GitPython/git clones the repo's default branch (usually "master" or
    "main") — NOT whatever branch you happen to be viewing on github.com.
    A GitHub web URL like ".../tree/dev" is NOT a valid git clone target;
    use the plain repo URL and pass branch="dev" explicitly instead.
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="code-doc-")
    logger.info(f"Cloning {repo_url} (branch={branch or 'default'}) to {target_dir}")
    if branch:
        GitRepo.clone_from(repo_url, target_dir, depth=1, branch=branch)
    else:
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


def skipped_source_files(repo_path: str, min_files: int = 3) -> dict[str, int]:
    """
    Extensions present in the repo that discover_files() ignores, with their file counts.

    A repo whose real code is in a language we do not index would otherwise be embedded from its
    README alone and answer plausibly but ungrounded. Extensions with fewer than `min_files`
    files and obvious non-source types (images, data, archives) are left out.
    """
    counts: dict[str, int] = {}
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in filenames:
            ext = Path(fname).suffix.lower()
            if ext in LANGUAGE_MAP or ext in TEXT_EXTENSIONS or ext in _NON_SOURCE_EXTENSIONS:
                continue
            counts[ext] = counts.get(ext, 0) + 1
    return {e: n for e, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n >= min_files}


def load_and_chunk_files(files: list[dict]) -> list:
    """
    Load files and split into chunks.

    Code files: AST-aware chunking via tree-sitter (CodeSplitter)
    Text files: Sentence-based chunking (SentenceSplitter) as fallback
    """
    documents = []

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

    code_docs = [d for d in documents if d.metadata.get("file_type") == "code"]
    text_docs = [d for d in documents if d.metadata.get("file_type") == "text"]
    all_nodes = []

    if code_docs and CHUNKING_STRATEGY == "ast":
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
                logger.warning(f"  {language}: AST parsing failed ({e}), using text fallback")
                fallback = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                nodes = fallback.get_nodes_from_documents(lang_docs)
                all_nodes.extend(nodes)
                logger.info(f"  {language}: {len(lang_docs)} files → {len(nodes)} chunks (fallback)")
    elif code_docs:
        logger.info(f"  Using text-based chunking for code (strategy={CHUNKING_STRATEGY})")
        text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = text_splitter.get_nodes_from_documents(code_docs)
        all_nodes.extend(nodes)
        logger.info(f"  code: {len(code_docs)} files → {len(nodes)} chunks (text)")

    if text_docs:
        text_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = text_splitter.get_nodes_from_documents(text_docs)
        all_nodes.extend(nodes)
        logger.info(f"  text/config: {len(text_docs)} files → {len(nodes)} chunks")

    logger.info(f"Total chunks: {len(all_nodes)}")
    return dedupe_nodes(all_nodes)


def dedupe_nodes(nodes: list) -> list:
    """
    Drop chunks whose text is identical (ignoring whitespace differences) to an earlier one.

    Identical text embeds to an identical vector, so a repeat adds no retrieval value -- it only
    takes up a top-k slot at query time and costs embedding time at ingest. Large generated or
    boilerplate-heavy files produce many such repeats (one Icarus file alone yielded 730 chunks,
    and a third of that collection was exact repeats). The first occurrence is kept.
    """
    seen: set[int] = set()
    kept: list = []
    dropped_by_file: dict[str, int] = {}
    for node in nodes:
        text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
        key = hash(" ".join(text.split()))
        if key in seen:
            f = (getattr(node, "metadata", None) or {}).get("file_path", "?")
            dropped_by_file[f] = dropped_by_file.get(f, 0) + 1
            continue
        seen.add(key)
        kept.append(node)
    if dropped_by_file:
        top = sorted(dropped_by_file.items(), key=lambda kv: kv[1], reverse=True)[:3]
        logger.info(
            f"De-duplicated chunks: dropped {len(nodes) - len(kept)} exact repeats "
            f"(most in: {', '.join(f'{f} x{n}' for f, n in top)})"
        )
    return kept


def build_index(
    nodes: list,
    vector_store_impl: ChromaVectorStoreImpl,
) -> VectorStoreIndex:
    """Embed chunks and store in the vector database."""
    from embedding import get_embed_model  # shared with query-time retrieval; CPU by default
    embed_model = get_embed_model()
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


def load_existing_index(vector_store_impl: ChromaVectorStoreImpl) -> VectorStoreIndex:
    """Load an existing index from the vector store (no re-ingestion)."""
    from embedding import get_embed_model
    embed_model = get_embed_model()
    vector_store = vector_store_impl.get_vector_store()
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    logger.info(f"Loaded existing index ({vector_store_impl.document_count} documents)")
    return index


def build_graphs(files: list[dict], repo_path: str, reset: bool = True) -> None:
    """
    Build the dependency graph and co-change graph as a parallel output
    of the ingestion pipeline.

    Called after the vector index is built, using the same file list.
    No-op if GRAPH_ENABLED=false or kuzu is not installed.

    Args:
        files:     output of discover_files()
        repo_path: repo root (for git log and import resolution)
        reset:     if True, clears existing graph before building
    """
    if not GRAPH_ENABLED:
        logger.info("Graph build skipped (GRAPH_ENABLED=false)")
        return

    try:
        from graph_store import KuzuGraphStore, build_dependency_graph, build_co_change_graph
    except ImportError:
        logger.warning("graph_store module not found — skipping graph build")
        return

    graph_store = KuzuGraphStore(GRAPH_PATH)
    if not graph_store.available:
        return

    if reset:
        graph_store.reset()

    build_dependency_graph(files, graph_store, repo_path)
    build_co_change_graph(graph_store, repo_path, n_commits=GRAPH_CO_CHANGE_COMMITS)
    logger.info("Graph build complete")


def ingest_codebase(
    repo_path: str,
    vector_store_impl: ChromaVectorStoreImpl,
    reset: bool = True,
) -> VectorStoreIndex:
    """
    Full ingestion pipeline: discover → chunk → embed → store → build graphs.

    The graph build step is additive — it runs after the vector index is
    complete and does not affect it. If graph build fails, the vector index
    is still returned successfully.

    Args:
        repo_path:         Local path to the codebase
        vector_store_impl: Vector store to write to
        reset:             If True, clear existing vectors and graph before ingesting

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

    # Graph construction — parallel output, does not affect vector index
    try:
        build_graphs(files, repo_path, reset=reset)
    except Exception as e:
        logger.warning(f"Graph build failed (non-fatal): {e}")

    return index
