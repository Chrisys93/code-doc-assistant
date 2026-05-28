"""
tools.py — Tool wrappers for the tool-selection agent.

Architecture: split registry
─────────────────────────────
Tools are divided into two categories:

  LOCAL_TOOL_REGISTRY  — plain Python callables running inside the container.
                         Always available regardless of LLM capability.
                         All existing tools (grep, cat, find, git_log, etc.)

  MCP_TOOL_REGISTRY    — MCP-backed tools accessed via MCPToolAdapter.
                         Require an MCP-capable LLM and active MCP server config.
                         Registered with type="mcp" and a server/tool_name pointer.

TOOL_REGISTRY          — unified view of both, used by node_tool_selection for
                         presenting available tools to the LLM, and by run_tool()
                         for dispatch.

Why a split rather than a single wrapper
─────────────────────────────────────────
Some LLMs (particularly lightweight/minimal tier quantised models) cannot reliably
produce the structured JSON required for MCP tool calls. Collapsing both tool types
into a single dispatcher would silently degrade for those models — the call would be
attempted, fail at serialisation, and return a confusing error. The split makes this
explicit: node_tool_selection filters the presented tool list based on whether the
current LLM is MCP-capable (resolved from INFERENCE_BACKEND + MODEL_TIER at runtime).
Local tools are always available. MCP tools are only offered when the LLM can use them.

Filesystem MCP access model
────────────────────────────
When the Filesystem MCP is active, file access is governed by an explicit permission
map rather than the bespoke _safe_path + subprocess allowlist used for local tools.
Three access tiers:

  read-only  → source files: *.py, *.ts, *.js, *.go, *.rs, *.java, *.cpp, *.c,
                              *.yaml, *.yml, *.toml, *.json, *.md, *.txt, *.env*
  read-write → generated artefacts: docs/, reports/, *.generated.*, *.docstring.*
  execute    → blocked entirely (no exec via MCP; shell tools use local registry)

This replaces the allowlist in _safe_path for MCP-routed file operations.
Local tools retain _safe_path for their own subprocess calls.

GitHub MCP scope (dev branch)
──────────────────────────────
The GitHub MCP replaces tool_github_fetch for external repo operations and extends
it with richer operations: list PRs, get commit diff, fetch issue body, list branches.
Commit/push operations are explicitly excluded from the dev branch — those belong to
a CI/CD integration, not the documentation agent.

Slack MCP scope (dev branch)
──────────────────────────────
Slack integration serves two purposes on dev:
  1. Developer workflow — query the assistant inline from Slack without opening the UI
  2. Self-documentation — the agent can post documentation summaries to relevant
     channels (#docs, #code-review) as a side-effect of a documentation run,
     making the assistant part of the team's ambient knowledge rather than a
     standalone tool

Security model (local tools)
──────────────────────────────
  - All shell tools run inside the container against the mounted repo volume only.
  - Command allowlist is enforced — no arbitrary shell execution.
  - File paths are validated to stay within REPO_PATH before any operation.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

# ---------------------------------------------------------------------------
# Path safety — all local tools enforce this
# ---------------------------------------------------------------------------

def _safe_path(repo_path: str, relative_or_absolute: str) -> str:
    """Resolve a path and assert it stays within repo_path. Raises ValueError otherwise."""
    repo = Path(repo_path).resolve()
    target = (repo / relative_or_absolute).resolve()
    if not str(target).startswith(str(repo)):
        raise ValueError(f"Path escape attempt: {relative_or_absolute!r} resolves outside repo")
    return str(target)


def _run(cmd: list[str], cwd: str, timeout: int = 30) -> tuple[str, bool]:
    """Run a subprocess, return (stdout+stderr, success)."""
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout + (f"\n[stderr]: {result.stderr}" if result.stderr.strip() else "")
        return output.strip(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return f"[timeout after {timeout}s]", False
    except Exception as e:
        return f"[error]: {e}", False


# ---------------------------------------------------------------------------
# Local tool implementations — shell / filesystem / AST / semantic
# ---------------------------------------------------------------------------

def tool_grep(repo_path: str, pattern: str, path: str = ".", flags: str = "-rn",
              include: str = "", max_lines: int = 100) -> dict[str, Any]:
    """
    Grep for a pattern within the repo.

    Args:
        repo_path:  Mounted repo root (e.g. /data/repos/myrepo)
        pattern:    Search pattern (regex supported)
        path:       Subdirectory to search (relative to repo_path, default ".")
        flags:      grep flags (restricted to safe subset)
        include:    File glob filter, e.g. "*.py"
        max_lines:  Truncate output after this many lines
    """
    # Allowlist flags — no --exec, no -z, no dangerous flags
    allowed_flags = {"-r", "-n", "-i", "-l", "-c", "-w", "-rn", "-ri", "-rin", "-rni"}
    if flags not in allowed_flags:
        return {"result": f"Disallowed flags: {flags}", "success": False}

    safe_path = _safe_path(repo_path, path)
    cmd = ["grep", flags, pattern, safe_path]
    if include:
        cmd += [f"--include={include}"]

    output, success = _run(cmd, cwd=repo_path)
    lines = output.splitlines()
    truncated = len(lines) > max_lines
    result = "\n".join(lines[:max_lines])
    if truncated:
        result += f"\n[... truncated at {max_lines} lines]"

    return {"result": result, "success": success, "truncated": truncated}


def tool_cat(repo_path: str, file_path: str, start_line: int = 1,
             end_line: int = -1) -> dict[str, Any]:
    """
    Read a file (or a line range) from the repo.

    Args:
        repo_path:  Mounted repo root
        file_path:  Path relative to repo_path
        start_line: First line to include (1-indexed)
        end_line:   Last line to include (-1 = end of file)
    """
    safe_file = _safe_path(repo_path, file_path)
    try:
        with open(safe_file) as f:
            all_lines = f.readlines()

        end = len(all_lines) if end_line == -1 else end_line
        selected = all_lines[start_line - 1:end]
        result = "".join(selected)
        return {"result": result, "success": True, "total_lines": len(all_lines)}
    except Exception as e:
        return {"result": str(e), "success": False}


def tool_find(repo_path: str, name_pattern: str = "", file_type: str = "f",
              path: str = ".") -> dict[str, Any]:
    """
    Find files matching a pattern within the repo.

    Args:
        repo_path:     Mounted repo root
        name_pattern:  Filename glob (e.g. "*.py", "config*")
        file_type:     "f" (files) | "d" (directories)
        path:          Subdirectory to search
    """
    safe_path = _safe_path(repo_path, path)
    cmd = ["find", safe_path, "-type", file_type]
    if name_pattern:
        cmd += ["-name", name_pattern]

    output, success = _run(cmd, cwd=repo_path)
    files = [line.replace(repo_path, "").lstrip("/") for line in output.splitlines() if line]
    return {"result": "\n".join(files), "files": files, "success": success}


def tool_git_log(repo_path: str, file_path: str = "", n: int = 10) -> dict[str, Any]:
    """
    Get recent git log, optionally scoped to a specific file.

    Args:
        repo_path:  Mounted repo root (must be a git repo)
        file_path:  Optional file to scope the log to
        n:          Number of commits to return
    """
    cmd = ["git", "log", f"-{n}", "--oneline", "--no-merges"]
    if file_path:
        safe = _safe_path(repo_path, file_path)
        cmd += ["--", safe]

    output, success = _run(cmd, cwd=repo_path)
    return {"result": output, "success": success}


def tool_git_blame(repo_path: str, file_path: str, start_line: int = 1,
                   end_line: int = 20) -> dict[str, Any]:
    """Git blame for a line range in a file."""
    safe = _safe_path(repo_path, file_path)
    cmd = ["git", "blame", f"-L{start_line},{end_line}", safe]
    output, success = _run(cmd, cwd=repo_path)
    return {"result": output, "success": success}


def tool_stat(repo_path: str, file_path: str) -> dict[str, Any]:
    """
    Get file metadata (size, modification time, permissions).
    Useful for checking staleness before embedding.
    """
    safe = _safe_path(repo_path, file_path)
    cmd = ["stat", "-c", "%n %s %y %A", safe]
    output, success = _run(cmd, cwd=repo_path)
    return {"result": output, "success": success}


def tool_vector_search(query: str, chroma_host: str, collection_name: str = "codebase",
                       top_k: int = 5, score_threshold: float = 0.3,
                       filter_file: str = "") -> dict[str, Any]:
    """
    Semantic vector search against ChromaDB.

    Args:
        query:            Natural language or code query
        chroma_host:      ChromaDB HTTP host (e.g. http://chromadb:8000)
        collection_name:  Collection to search
        top_k:            Number of results to return
        score_threshold:  Minimum similarity score (0–1)
        filter_file:      Optional: restrict to chunks from this file path
    """
    try:
        import chromadb
        client = chromadb.HttpClient(host=chroma_host.replace("http://", "").split(":")[0],
                                     port=int(chroma_host.split(":")[-1]))
        collection = client.get_collection(collection_name)

        where = {"source_file": {"$eq": filter_file}} if filter_file else None
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            score = 1 - dist  # ChromaDB returns L2 distance; invert for similarity
            if score >= score_threshold:
                chunks.append({
                    "content": doc,
                    "source_file": meta.get("source_file", "unknown"),
                    "start_line": meta.get("start_line"),
                    "end_line": meta.get("end_line"),
                    "chunk_type": meta.get("chunk_type", "text"),
                    "confidence": round(score, 4)
                })

        return {"chunks": chunks, "success": True, "count": len(chunks)}
    except Exception as e:
        return {"chunks": [], "success": False, "error": str(e)}


def tool_ast_parse(repo_path: str, file_path: str) -> dict[str, Any]:
    """
    Parse a source file with tree-sitter and return the symbol table:
    functions, classes, imports, and their line ranges.

    Falls back to a simple regex scan if tree-sitter parsing fails.
    """
    safe = _safe_path(repo_path, file_path)
    try:
        from tree_sitter_language_pack import get_parser
        import re

        ext = Path(safe).suffix.lstrip(".")
        lang_map = {"py": "python", "js": "javascript", "ts": "typescript",
                    "go": "go", "rs": "rust", "java": "java", "cpp": "cpp", "c": "c"}
        lang = lang_map.get(ext)

        with open(safe, "rb") as f:
            source = f.read()

        if lang:
            parser = get_parser(lang)
            tree = parser.parse(source)

            symbols = []
            def _walk(node):
                if node.type in ("function_definition", "class_definition",
                                 "function_declaration", "method_definition"):
                    name_node = node.child_by_field_name("name")
                    name = name_node.text.decode() if name_node else "<anonymous>"
                    symbols.append({
                        "type": node.type,
                        "name": name,
                        "start_line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                    })
                for child in node.children:
                    _walk(child)
            _walk(tree.root_node)

            result_str = "\n".join(
                f"{s['type']} `{s['name']}` (lines {s['start_line']}–{s['end_line']})"
                for s in symbols
            )
            return {"symbols": symbols, "result": result_str, "success": True,
                    "method": "tree-sitter"}

    except Exception:
        pass

    # Fallback: regex-based symbol extraction
    try:
        import re
        with open(safe) as f:
            lines = f.readlines()
        symbols = []
        for i, line in enumerate(lines, 1):
            m = re.match(r"^\s*(def|class|function|func)\s+(\w+)", line)
            if m:
                symbols.append({"type": m.group(1), "name": m.group(2), "start_line": i})
        result_str = "\n".join(f"{s['type']} `{s['name']}` (line {s['start_line']})" for s in symbols)
        return {"symbols": symbols, "result": result_str, "success": True, "method": "regex-fallback"}
    except Exception as e:
        return {"symbols": [], "result": str(e), "success": False}


def tool_graph_traverse(
    query_type: str,
    target: str,
    graph_path: str = "",
    depth: int = 1,
    min_weight: int = 2,
) -> dict[str, Any]:
    """
    Traverse the code dependency or co-change graph.

    Precise, enumerable answers to structural questions — complements
    vector_search (semantic/fuzzy) with graph traversal (exact/relational).

    query_type options:
      callees        → functions called by `target` function (up to `depth` hops)
      callers        → functions that call `target` function (up to `depth` hops)
      dependencies   → files imported by `target` file
      dependents     → files that import `target` file
      co_changed     → files frequently changed with `target` file (git history)
      symbols        → all functions and classes defined in `target` file
      cypher         → raw Cypher query passed as `target` string (advanced)

    Args:
        query_type:  one of the above options
        target:      function name, file path, or raw Cypher string
        graph_path:  path to Kuzu DB directory (default: GRAPH_PATH env var)
        depth:       traversal depth for callees/callers (default: 1)
        min_weight:  minimum co-change count for co_changed (default: 2)
    """
    try:
        from graph_store import KuzuGraphStore
    except ImportError:
        return {"result": "graph_store module not available", "success": False}

    gp = graph_path or os.environ.get("GRAPH_PATH", "/data/graph")
    gs = KuzuGraphStore(gp)

    if not gs.available:
        return {
            "result": "Graph store not available. "
                      "Install kuzu (pip install kuzu) and re-run ingestion.",
            "success": False,
        }

    try:
        if query_type == "callees":
            results = gs.callees(target, depth=depth)
            label = f"Functions called by `{target}` (depth={depth})"
        elif query_type == "callers":
            results = gs.callers(target, depth=depth)
            label = f"Functions that call `{target}` (depth={depth})"
        elif query_type == "dependencies":
            results = gs.dependencies(target)
            label = f"Files imported by `{target}`"
        elif query_type == "dependents":
            results = gs.dependents(target)
            label = f"Files that import `{target}`"
        elif query_type == "co_changed":
            rows = gs.co_changed_with(target, min_weight=min_weight)
            results = [f"{r['file']} (co-changes: {r['co_changes']})" for r in rows]
            label = f"Files frequently changed with `{target}` (min_weight={min_weight})"
        elif query_type == "symbols":
            rows = gs.symbols_in_file(target)
            results = [
                f"{r['kind']} `{r['name']}` (lines {r.get('start', '?')}–{r.get('end_line', '?')})"
                for r in rows
            ]
            label = f"Symbols defined in `{target}`"
        elif query_type == "cypher":
            rows = gs.query(target)
            result_str = "\n".join(str(r) for r in rows) if rows else "(no results)"
            return {"result": result_str, "rows": rows, "success": True}
        else:
            return {
                "result": f"Unknown query_type: {query_type!r}. "
                          "Valid: callees, callers, dependencies, dependents, "
                          "co_changed, symbols, cypher",
                "success": False,
            }

        result_str = f"{label}:\n" + (
            "\n".join(f"  - {r}" for r in results) if results else "  (none found)"
        )
        return {"result": result_str, "items": results, "success": True}

    except Exception as e:
        return {"result": f"Graph traversal error: {e}", "success": False}


def tool_github_fetch(owner: str, repo: str, file_path: str,
                      ref: str = "main", github_token: str = "") -> dict[str, Any]:
    """
    Fetch a file directly from the GitHub REST API (local fallback).
    Prefer the GitHub MCP tool when available — it handles OAuth, rate limits,
    and richer operations (PR diffs, issue body, branch listing).

    Args:
        owner:        GitHub org or username
        repo:         Repository name
        file_path:    Path within the repo (e.g. "src/app.py")
        ref:          Branch, tag, or commit SHA
        github_token: Optional PAT for private repos (read from env if not provided)
    """
    import urllib.request
    import base64, json

    token = github_token or os.environ.get("GITHUB_TOKEN", "")
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={ref}"
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return {"result": content, "success": True, "sha": data.get("sha")}
    except Exception as e:
        return {"result": str(e), "success": False}


# ---------------------------------------------------------------------------
# MCP tool adapter
# ---------------------------------------------------------------------------

ToolType = Literal["local", "mcp"]


@dataclass
class MCPToolAdapter:
    """
    Wraps an MCP server tool call in the same (args) → dict[str, Any] signature
    as local tools, so run_tool() dispatches both identically.

    The adapter is intentionally thin — it does not validate args or parse
    responses. That responsibility stays with the MCP server itself.

    mcp_server:  MCP server identifier (matches MCP_SERVER_CONFIG key)
    tool_name:   Tool name as published by the MCP server
    timeout:     Per-call timeout in seconds

    LLM compatibility note
    ──────────────────────
    MCP tool calls require the LLM to produce structured JSON conforming to
    the tool's input schema. Lightweight/minimal tier models may not do this
    reliably. node_tool_selection uses is_mcp_capable() to filter the presented
    tool list — MCP tools are only offered to capable LLMs. Local tools are
    always available regardless of model tier.
    """
    mcp_server: str
    tool_name: str
    timeout: int = 30

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the MCP tool call. Returns a normalised result dict."""
        config = MCP_SERVER_CONFIG.get(self.mcp_server)
        if not config:
            return {
                "result": f"MCP server '{self.mcp_server}' not configured or not enabled.",
                "success": False,
            }
        if not config.get("enabled", False):
            return {
                "result": f"MCP server '{self.mcp_server}' is disabled. "
                          f"Set MCP_{self.mcp_server.upper()}_ENABLED=true to enable.",
                "success": False,
            }
        try:
            from mcp import ClientSession
            import asyncio

            async def _call():
                async with ClientSession(config["url"]) as session:
                    result = await session.call_tool(self.tool_name, kwargs)
                    return result

            result = asyncio.run(_call())
            # Normalise: MCP returns content blocks; extract text content
            text_blocks = [b.text for b in result.content if hasattr(b, "text")]
            return {
                "result": "\n".join(text_blocks) if text_blocks else str(result),
                "success": not result.isError,
                "raw": result,
            }
        except ImportError:
            return {
                "result": "MCP client library not installed. Run: pip install mcp",
                "success": False,
            }
        except Exception as e:
            return {"result": str(e), "success": False}


def is_mcp_capable() -> bool:
    """
    Return True if the current LLM configuration can reliably produce
    MCP-compatible structured tool call JSON.

    full + balanced tiers are considered MCP-capable.
    lightweight + minimal are not — they are offered local tools only.
    """
    model_tier = os.environ.get("MODEL_TIER", "full").lower()
    return model_tier in ("full", "balanced")


# ---------------------------------------------------------------------------
# MCP server configuration
# Resolved from environment variables at import time.
# Add new MCP servers here; they are automatically available to MCP_TOOL_REGISTRY.
# ---------------------------------------------------------------------------

MCP_SERVER_CONFIG: dict[str, dict[str, Any]] = {
    "filesystem": {
        "url": os.environ.get("MCP_FILESYSTEM_URL", "http://localhost:3000"),
        "enabled": os.environ.get("MCP_FILESYSTEM_ENABLED", "false").lower() == "true",
        # Access tiers enforced by the MCP server config, not here:
        #   read-only  → source files (*.py, *.ts, *.yaml, *.md, etc.)
        #   read-write → generated artefacts (docs/, reports/, *.generated.*)
        #   execute    → blocked
    },
    "github": {
        "url": os.environ.get("MCP_GITHUB_URL", "http://localhost:3001"),
        "enabled": os.environ.get("MCP_GITHUB_ENABLED", "false").lower() == "true",
        # Scope: read operations + PR/issue/branch listing.
        # Commit/push excluded on dev branch — belongs to CI/CD, not the doc agent.
    },
    "slack": {
        "url": os.environ.get("MCP_SLACK_URL", "http://localhost:3002"),
        "enabled": os.environ.get("MCP_SLACK_ENABLED", "false").lower() == "true",
        # Scope: post messages to configured channels (#docs, #code-review).
        #        read channel history for context on a PR or feature.
        # Purpose: developer workflow (query inline) + self-documentation (post summaries).
    },
}


# ---------------------------------------------------------------------------
# Local tool registry
# ---------------------------------------------------------------------------

LOCAL_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "grep": {
        "type": "local",
        "fn": tool_grep,
        "description": "Search for patterns across the codebase. Best for: locating definitions, finding usages of a symbol, tracing imports.",
        "required_args": ["repo_path", "pattern"],
        "optional_args": ["path", "flags", "include", "max_lines"],
    },
    "cat": {
        "type": "local",
        "fn": tool_cat,
        "description": "Read a specific file or line range. Best for: fetching a known file, reading a specific function after grep located it.",
        "required_args": ["repo_path", "file_path"],
        "optional_args": ["start_line", "end_line"],
    },
    "find": {
        "type": "local",
        "fn": tool_find,
        "description": "Find files by name pattern. Best for: discovering all config files, finding test files, locating entry points.",
        "required_args": ["repo_path"],
        "optional_args": ["name_pattern", "file_type", "path"],
    },
    "git_log": {
        "type": "local",
        "fn": tool_git_log,
        "description": "Get recent commit history, optionally for a specific file. Best for: understanding recent changes, finding when something was introduced.",
        "required_args": ["repo_path"],
        "optional_args": ["file_path", "n"],
    },
    "git_blame": {
        "type": "local",
        "fn": tool_git_blame,
        "description": "Get authorship and commit info for a line range. Best for: understanding who changed what and when.",
        "required_args": ["repo_path", "file_path"],
        "optional_args": ["start_line", "end_line"],
    },
    "stat": {
        "type": "local",
        "fn": tool_stat,
        "description": "Get file metadata (size, modified time, permissions). Best for: checking if a file is recent before deciding to re-embed.",
        "required_args": ["repo_path", "file_path"],
        "optional_args": [],
    },
    "vector_search": {
        "type": "local",
        "fn": tool_vector_search,
        "description": "Semantic similarity search over embedded code chunks. Best for: conceptual questions ('how does caching work?'), cross-file relationships, architectural questions.",
        "required_args": ["query", "chroma_host"],
        "optional_args": ["collection_name", "top_k", "score_threshold", "filter_file"],
    },
    "ast_parse": {
        "type": "local",
        "fn": tool_ast_parse,
        "description": "Parse a file's AST to get a symbol table (functions, classes, line ranges). Best for: understanding a file's structure before fetching specific sections.",
        "required_args": ["repo_path", "file_path"],
        "optional_args": [],
    },
    "graph_traverse": {
        "type": "local",
        "fn": tool_graph_traverse,
        "description": (
            "Traverse the code dependency or co-change graph for precise structural answers. "
            "Complements vector_search (semantic) with exact graph traversal. "
            "query_type: callees (what does this function call?), "
            "callers (what calls this function?), "
            "dependencies (what does this file import?), "
            "dependents (what imports this file?), "
            "co_changed (what else changed with this file in git history?), "
            "symbols (all functions/classes in a file), "
            "cypher (raw Cypher for advanced queries). "
            "Requires GRAPH_ENABLED=true and kuzu installed."
        ),
        "required_args": ["query_type", "target"],
        "optional_args": ["graph_path", "depth", "min_weight"],
    },
    "github_fetch": {
        "type": "local",
        "fn": tool_github_fetch,
        "description": "Fetch a file from GitHub REST API (local fallback). Prefer github_mcp_* tools when MCP is available — they handle OAuth and richer operations.",
        "required_args": ["owner", "repo", "file_path"],
        "optional_args": ["ref", "github_token"],
    },
}


# ---------------------------------------------------------------------------
# MCP tool registry
# Tools are only offered to the LLM when is_mcp_capable() returns True
# and the corresponding MCP server is enabled.
# ---------------------------------------------------------------------------

MCP_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    # --- Filesystem MCP ---
    "fs_read": {
        "type": "mcp",
        "fn": MCPToolAdapter("filesystem", "read_file"),
        "description": "Read a file via Filesystem MCP. Enforces declarative access tiers (source=read-only, artefacts=read-write). Preferred over local cat for MCP-capable LLMs.",
        "required_args": ["path"],
        "optional_args": [],
        "mcp_server": "filesystem",
    },
    "fs_write": {
        "type": "mcp",
        "fn": MCPToolAdapter("filesystem", "write_file"),
        "description": "Write a file via Filesystem MCP. Restricted to artefact paths (docs/, reports/, *.generated.*). Source files are read-only.",
        "required_args": ["path", "content"],
        "optional_args": [],
        "mcp_server": "filesystem",
    },
    "fs_list": {
        "type": "mcp",
        "fn": MCPToolAdapter("filesystem", "list_directory"),
        "description": "List directory contents via Filesystem MCP.",
        "required_args": ["path"],
        "optional_args": [],
        "mcp_server": "filesystem",
    },
    # --- GitHub MCP ---
    "github_get_file": {
        "type": "mcp",
        "fn": MCPToolAdapter("github", "get_file_contents"),
        "description": "Fetch a file from GitHub via MCP. Handles OAuth and rate limits. Prefer over local github_fetch.",
        "required_args": ["owner", "repo", "path"],
        "optional_args": ["ref"],
        "mcp_server": "github",
    },
    "github_list_prs": {
        "type": "mcp",
        "fn": MCPToolAdapter("github", "list_pull_requests"),
        "description": "List open PRs for a repo. Useful for correlating code changes with PR descriptions.",
        "required_args": ["owner", "repo"],
        "optional_args": ["state", "head", "base"],
        "mcp_server": "github",
    },
    "github_get_pr_diff": {
        "type": "mcp",
        "fn": MCPToolAdapter("github", "get_pull_request_diff"),
        "description": "Get the diff for a specific PR. Best for: documenting what changed in a feature branch.",
        "required_args": ["owner", "repo", "pull_number"],
        "optional_args": [],
        "mcp_server": "github",
    },
    "github_get_issue": {
        "type": "mcp",
        "fn": MCPToolAdapter("github", "get_issue"),
        "description": "Fetch issue body and comments. Useful for understanding the intent behind a code change.",
        "required_args": ["owner", "repo", "issue_number"],
        "optional_args": [],
        "mcp_server": "github",
    },
    # --- Slack MCP ---
    "slack_post": {
        "type": "mcp",
        "fn": MCPToolAdapter("slack", "post_message"),
        "description": "Post a documentation summary to a Slack channel. Use for self-documentation side-effects (e.g. post to #docs after generating module documentation).",
        "required_args": ["channel", "text"],
        "optional_args": ["thread_ts"],
        "mcp_server": "slack",
    },
    "slack_get_channel_history": {
        "type": "mcp",
        "fn": MCPToolAdapter("slack", "get_channel_history"),
        "description": "Read recent messages from a Slack channel. Useful for context on a feature being documented (e.g. #code-review thread for a PR).",
        "required_args": ["channel"],
        "optional_args": ["limit"],
        "mcp_server": "slack",
    },
}


# ---------------------------------------------------------------------------
# Unified registry — used by node_tool_selection and run_tool()
# MCP tools are included only when is_mcp_capable() and server is enabled.
# ---------------------------------------------------------------------------

def build_tool_registry() -> dict[str, dict[str, Any]]:
    """
    Build the unified tool registry for the current runtime context.
    Called once at graph build time (build_graph() in agent_graph.py).

    Local tools are always included.
    MCP tools are included only when:
      1. is_mcp_capable() — the current LLM tier can produce structured JSON
      2. The tool's MCP server is enabled in MCP_SERVER_CONFIG
    """
    registry: dict[str, dict[str, Any]] = dict(LOCAL_TOOL_REGISTRY)

    if is_mcp_capable():
        for name, entry in MCP_TOOL_REGISTRY.items():
            server = entry.get("mcp_server", "")
            if MCP_SERVER_CONFIG.get(server, {}).get("enabled", False):
                registry[name] = entry

    return registry


# Eager-evaluated unified registry — used at module import time.
# node_tool_selection should call build_tool_registry() at graph build time
# for runtime-accurate filtering; this is the fallback for direct imports.
TOOL_REGISTRY: dict[str, dict[str, Any]] = build_tool_registry()


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a registered tool by name. Dispatches identically for local and MCP tools.
    Returns the tool's result dict, always with a 'latency_ms' key appended.
    """
    registry = build_tool_registry()
    if name not in registry:
        return {"result": f"Unknown tool: {name!r}. Available: {list(registry)}", "success": False}

    entry = registry[name]
    start = time.time()
    try:
        if entry["type"] == "local":
            result = entry["fn"](**args)
        else:
            # MCP: fn is an MCPToolAdapter instance, callable with kwargs
            result = entry["fn"](**args)
    except TypeError as e:
        result = {"result": f"Argument error calling {name!r}: {e}", "success": False}
    except Exception as e:
        result = {"result": f"Tool error in {name!r}: {e}", "success": False}

    result["latency_ms"] = round((time.time() - start) * 1000, 1)
    return result
