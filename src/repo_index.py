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
import json
import logging
import os
import re
import urllib.request

logger = logging.getLogger(__name__)

_MAX_SLUG = 40  # keep total collection name within Chroma's 63-char limit


def _parse_ref(repo_ref: str) -> tuple[str, str | None]:
    """
    Split a repo reference into (url_or_path, branch). Branch syntax is
    `<url>#<branch>` (e.g. "https://github.com/x/y.git#dev") — a '#' fragment,
    not '@', because git@host:owner/repo.git SSH URLs already use '@' as part
    of the URL itself and splitting on it would break them.

    A bare GitHub web URL like ".../tree/dev" is explicitly NOT treated as a
    valid clone target — that's a browsing URL, not a git remote. If someone
    pastes one, ensure_indexed() will fail clone_repo() and report status=error
    with a message pointing at the #branch syntax instead.
    """
    ref = (repo_ref or "").strip()
    if "#" in ref:
        url, branch = ref.rsplit("#", 1)
        branch = branch.strip() or None
        return url.strip(), branch
    return ref, None


def collection_for_repo(repo_ref: str) -> str:
    """
    Stable ChromaDB collection name for a repo URL/path (+ optional #branch).

    Same ref -> same collection every time; different repos never collide
    (a short hash of the full ref, including branch, disambiguates same-named
    repos from different owners/hosts AND different branches of the same repo
    — "x.git" and "x.git#dev" hash to different collections). Result matches
    Chroma's naming rules: 3-63 chars, starts/ends alphanumeric, only [a-z0-9_].
    """
    ref = (repo_ref or "").strip().rstrip("/")
    url, branch = _parse_ref(ref)
    name = url.split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    if branch:
        name = f"{name}_{branch}"
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower() or "repo"
    slug = slug[:_MAX_SLUG]
    digest = hashlib.sha1(ref.encode("utf-8")).hexdigest()[:8]
    return f"repo_{slug}_{digest}"


def embedding_ready(timeout: float = 4.0) -> tuple[bool, str]:
    """Is the embedding model actually available in Ollama? (ok, reason).

    Checked BEFORE a repo is ingested: without the model every embedding call 404s, the collection
    is created but stays empty, and the failure used to show up only as a model answering
    "no specific information" from an empty context.
    """
    model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception as e:  # unreachable, DNS, 5xx ...
        return False, f"Ollama is not reachable at {host} ({e})"
    base = model.split(":")[0]
    if any(n.split(":")[0] == base for n in names):
        return True, "ok"
    have = ", ".join(names) or "no models at all"
    return False, f"embedding model '{model}' is not available in Ollama at {host} (it has: {have})"


def _hint(error: str) -> str:
    model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    if "not available in Ollama" in error or "try pulling" in error or "Ollama is not reachable" in error:
        return (f"Pull the embedding model into Ollama: `ollama pull {model}` inside the Ollama container "
                f"(Kubernetes: `kubectl -n <namespace> exec <ollama-pod> -- ollama pull {model}`; "
                f"Compose: `docker compose exec ollama ollama pull {model}`), then ask again.")
    return "See the app log for the full traceback."


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
    """Clone a URL (optionally #branch) to a writable temp dir; local paths unchanged."""
    url, branch = _parse_ref(repo_ref)
    is_url = url.startswith(("http://", "https://", "git@")) or url.endswith(".git")
    if not is_url:
        return url
    if "/tree/" in url or "/blob/" in url:
        raise ValueError(
            f"{url!r} looks like a GitHub web (browsing) URL, not a git remote. "
            f"Use the plain repo URL with #branch instead, e.g. "
            f"'https://github.com/owner/repo.git#dev'."
        )
    try:
        from src.ingest import clone_repo
    except ImportError:
        from ingest import clone_repo
    return clone_repo(url, branch=branch)


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

        ok, why = embedding_ready()
        if not ok:
            logger.error("ensure_indexed: not ingesting %s: %s", ref, why)
            results[ref] = {"collection": coll, "docs": n, "status": "error", "error": why, "hint": _hint(why)}
            continue

        try:
            path = _resolve_to_path(ref)
            store = ChromaVectorStoreImpl(host=chroma_host, collection_name=coll)
            # reset only when forcing a re-index; a fresh collection is already empty.
            ingest_codebase(path, store, reset=force)
            docs = store.document_count
            if docs == 0:
                msg = "ingestion finished but the collection is empty (0 chunks stored)"
                logger.error("ensure_indexed: %s: %s", ref, msg)
                results[ref] = {"collection": coll, "docs": 0, "status": "error", "error": msg,
                                "hint": "Check the app log around the ingestion step for embedding errors."}
            else:
                results[ref] = {"collection": coll, "docs": docs,
                                "status": "reindexed" if force else "ingested"}
        except Exception as e:
            logger.exception("ensure_indexed failed for %s", ref)
            results[ref] = {"collection": coll, "docs": n, "status": "error", "error": str(e), "hint": _hint(str(e))}

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
