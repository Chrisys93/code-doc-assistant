"""
repo_index.py — multi-repo indexing support.

One embedding model, many collections: each repo (git URL or local path) is
ingested into its OWN ChromaDB collection, named deterministically from the ref.
This keeps repos isolated at retrieval time while sharing one embedding space,
so a query can target one repo, a group, or all of them.

The key entry point is `ensure_indexed(repos, chroma_host)` — a DETERMINISTIC
gate meant to run in app.py before a query is dispatched, guaranteeing every
repo the user entered is indexed (idempotent: already-indexed repos are skipped).
"""

from __future__ import annotations

import hashlib
import logging
import re

logger = logging.getLogger(__name__)

_MAX_SLUG = 40  # keep total collection name within Chroma's 63-char limit


def collection_for_repo(repo_ref: str) -> str:
    """
    Stable ChromaDB collection name for a repo URL or local path.

    Same ref -> same collection every time; different repos never collide
    (a short hash of the full ref disambiguates same-named repos from
    different owners/hosts). Result matches Chroma's naming rules:
    3-63 chars, starts/ends alphanumeric, only [a-z0-9_].
    """
    ref = (repo_ref or "").strip().rstrip("/")
    name = ref.split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower() or "repo"
    slug = slug[:_MAX_SLUG]
    digest = hashlib.sha1(ref.encode("utf-8")).hexdigest()[:8]
    return f"repo_{slug}_{digest}"


def _chroma_client(chroma_host: str):
    import chromadb
    host = chroma_host.replace("http://", "").replace("https://", "").split(":")[0]
    port = int(chroma_host.split(":")[-1]) if ":" in chroma_host.replace("http://", "") else 8000
    return chromadb.HttpClient(host=host, port=port)


def repo_indexed(repo_ref: str, chroma_host: str, min_docs: int = 1) -> tuple[bool, int]:
    """Return (is_indexed, doc_count) for a repo's collection. Missing collection -> (False, 0)."""
    coll = collection_for_repo(repo_ref)
    try:
        client = _chroma_client(chroma_host)
        n = client.get_collection(coll).count()
        return (n >= min_docs, n)
    except Exception:
        return (False, 0)


def _resolve_to_path(repo_ref: str) -> str:
    """Clone a URL to a writable temp dir; return local paths unchanged."""
    try:
        from src.ingest import clone_repo
    except ImportError:
        from ingest import clone_repo
    is_url = repo_ref.startswith(("http://", "https://", "git@")) or repo_ref.endswith(".git")
    return clone_repo(repo_ref) if is_url else repo_ref


def ensure_indexed(repos: list[str], chroma_host: str,
                   force: bool = False) -> dict[str, dict]:
    """
    Guarantee every repo in `repos` is indexed into its own collection BEFORE a
    query runs. Idempotent: an already-populated collection is left as-is unless
    force=True. Returns a per-repo status dict for UI display, e.g.:

        {"https://github.com/x/y": {"collection": "repo_y_1a2b3c4d",
                                    "docs": 1387, "status": "already_indexed"}}
    """
    try:
        from src.ingest import ingest_codebase
        from src.vector_store import ChromaVectorStoreImpl
    except ImportError:
        from ingest import ingest_codebase
        from vector_store import ChromaVectorStoreImpl

    results: dict[str, dict] = {}
    for ref in repos:
        ref = (ref or "").strip()
        if not ref:
            continue
        coll = collection_for_repo(ref)
        already, n = repo_indexed(ref, chroma_host)

        if already and not force:
            results[ref] = {"collection": coll, "docs": n, "status": "already_indexed"}
            continue

        try:
            path = _resolve_to_path(ref)
            store = ChromaVectorStoreImpl(host=chroma_host, collection_name=coll)
            # reset only when forcing a re-index; a fresh collection is already empty.
            ingest_codebase(path, store, reset=force)
            results[ref] = {"collection": coll, "docs": store.document_count,
                            "status": "reindexed" if force else "ingested"}
        except Exception as e:
            logger.exception("ensure_indexed failed for %s", ref)
            results[ref] = {"collection": coll, "docs": n, "status": "error", "error": str(e)}

    return results


def active_collections(repos: list[str]) -> list[str]:
    """Collection names for a set of repos — what a query should search."""
    return [collection_for_repo(r) for r in repos if (r or "").strip()]


def collections_doc_count(collection_names: list[str], chroma_host: str) -> int:
    """Total document count across a set of collections (missing ones count as 0).
    Used by node_tool_selection to judge readiness against the ACTIVE per-repo
    collections rather than the legacy single 'codebase' collection."""
    total = 0
    try:
        client = _chroma_client(chroma_host)
    except Exception:
        return 0
    for name in collection_names or []:
        try:
            total += client.get_collection(name).count()
        except Exception:
            continue
    return total
