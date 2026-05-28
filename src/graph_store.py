"""
graph_store.py — Code dependency graph using Kuzu (embedded graph DB).

Built as a parallel output of the ingestion pipeline, alongside the ChromaDB
vector index. No new services — Kuzu is an embedded library (like SQLite),
the graph persists as a directory on disk.

Why a graph DB here, not the vector DB
───────────────────────────────────────
Vector DB:  answers "what is semantically similar to this query?"
            — fuzzy, approximate, meaning-based
Graph DB:   answers "what is structurally related to this entity?"
            — precise, enumerable, traversal-based

These are orthogonal retrieval mechanisms. The graph DB does not replace
the vector DB — it handles a class of queries the vector DB handles poorly:

  "What functions does node_supervisor call?"      → call graph traversal
  "What modules depend on vector_store.py?"        → import graph traversal
  "What changed together with tool_vector_search?" → co-change traversal
  "Find all callers of _safe_path transitively"    → multi-hop traversal

Graph structures built during ingestion
─────────────────────────────────────────
1. Code dependency graph
   Nodes: File, Function, Class
   Edges: IMPORTS (file→file), CALLS (function→function),
          DEFINES (file→function/class), INHERITS (class→class)
   Source: tree-sitter AST parse of each file

2. Co-change graph
   Nodes: File
   Edges: CHANGED_TOGETHER (file→file, weight = co-occurrence count)
   Source: git log --follow --name-only (last N commits)
   Use:   "if I change X, what else will likely need updating?"

3. Knowledge graph (documentation layer) — Phase 15+ / research
   Nodes: Concept, DesignDecision, Component
   Edges: IMPLEMENTS, DEPENDS_ON, DOCUMENTED_IN, REPLACED_BY
   Source: structured extraction from ARCHITECTURE.md + docstrings
   Note:  This is the research-level extension toward graph-aware RAG
          (GraphRAG pattern). Not built during standard ingestion — opt-in
          via GRAPH_BUILD_KNOWLEDGE=true. Relevant to orchestrated branch.

Kuzu vs Neo4j
──────────────
Kuzu is not derived from Neo4j. Both implement openCypher (an open standard),
but Kuzu is built from scratch at University of Waterloo (2022). It is
embedded (no server process, Python-native), written in C++. The right mental
model is "Neo4j : PostgreSQL :: Kuzu : SQLite" — same query language, completely
different deployment model. Zero new services, zero new infrastructure.

Schema
──────
Node tables:
  File(path STRING PRIMARY KEY, language STRING, line_count INT64)
  Function(id STRING PRIMARY KEY, name STRING, file_path STRING,
           start_line INT64, end_line INT64)
  Class(id STRING PRIMARY KEY, name STRING, file_path STRING,
        start_line INT64, end_line INT64)

Relationship tables:
  IMPORTS(File → File)
  DEFINES(File → Function)
  DEFINES_CLASS(File → Class)
  CALLS(Function → Function, call_count INT64)
  INHERITS(Class → Class)
  CHANGED_TOGETHER(File → File, weight INT64)

Usage in the agent
───────────────────
  from graph_store import KuzuGraphStore
  gs = KuzuGraphStore(graph_path)
  gs.query("MATCH (f:Function)-[:CALLS]->(g:Function) WHERE f.name = $name RETURN g.name",
           {"name": "node_supervisor"})
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

SCHEMA_DDL = """
CREATE NODE TABLE IF NOT EXISTS File(
    path     STRING,
    language STRING,
    line_count INT64,
    PRIMARY KEY(path)
);

CREATE NODE TABLE IF NOT EXISTS Function(
    id         STRING,
    name       STRING,
    file_path  STRING,
    start_line INT64,
    end_line   INT64,
    PRIMARY KEY(id)
);

CREATE NODE TABLE IF NOT EXISTS Class(
    id         STRING,
    name       STRING,
    file_path  STRING,
    start_line INT64,
    end_line   INT64,
    PRIMARY KEY(id)
);

CREATE REL TABLE IF NOT EXISTS IMPORTS(FROM File TO File);
CREATE REL TABLE IF NOT EXISTS DEFINES(FROM File TO Function);
CREATE REL TABLE IF NOT EXISTS DEFINES_CLASS(FROM File TO Class);
CREATE REL TABLE IF NOT EXISTS CALLS(FROM Function TO Function, call_count INT64);
CREATE REL TABLE IF NOT EXISTS INHERITS(FROM Class TO Class);
CREATE REL TABLE IF NOT EXISTS CHANGED_TOGETHER(FROM File TO File, weight INT64);
"""


# ---------------------------------------------------------------------------
# KuzuGraphStore
# ---------------------------------------------------------------------------

class KuzuGraphStore:
    """
    Embedded Kuzu graph store for code dependency and co-change graphs.

    graph_path: directory where Kuzu persists its data (e.g. /data/graph)
    """

    def __init__(self, graph_path: str):
        self.graph_path = graph_path
        self._db = None
        self._conn = None
        self._init_db()

    def _init_db(self) -> None:
        try:
            import kuzu
            # Kuzu creates the database directory itself — do NOT pre-create it.
            # mkdir the parent only, so the path is reachable.
            Path(self.graph_path).parent.mkdir(parents=True, exist_ok=True)
            self._db = kuzu.Database(self.graph_path)
            self._conn = kuzu.Connection(self._db)
            for stmt in SCHEMA_DDL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    self._conn.execute(stmt + ";")
            logger.info(f"Kuzu graph DB initialised at {self.graph_path}")
        except ImportError:
            logger.warning(
                "kuzu not installed — graph store disabled. "
                "Install with: pip install kuzu"
            )
        except Exception as e:
            logger.error(f"Kuzu init failed: {e}")

    @property
    def available(self) -> bool:
        return self._conn is not None

    def reset(self) -> None:
        """Delete and recreate the graph. Kuzu stores as a file, not a directory."""
        import shutil
        self._db = None
        self._conn = None
        path = Path(self.graph_path)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(self.graph_path)
            else:
                path.unlink()
        self._init_db()
        logger.info("Graph store reset")

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict]:
        """
        Execute a Cypher query and return results as a list of dicts.

        Example:
            gs.query(
                "MATCH (f:Function)-[:CALLS]->(g:Function) "
                "WHERE f.name = $name RETURN g.name AS callee",
                {"name": "node_supervisor"}
            )
        """
        if not self.available:
            return []
        try:
            result = self._conn.execute(cypher, parameters=params or {})
            rows = []
            while result.has_next():
                row = result.get_next()
                cols = result.get_column_names()
                rows.append(dict(zip(cols, row)))
            return rows
        except Exception as e:
            logger.error(f"Graph query failed: {e}\nCypher: {cypher}")
            return []

    # -----------------------------------------------------------------------
    # Graph construction helpers — called by build_graph()
    # -----------------------------------------------------------------------

    def upsert_file(self, path: str, language: str, line_count: int) -> None:
        if not self.available:
            return
        self._conn.execute(
            "MERGE (f:File {path: $path}) "
            "SET f.language = $lang, f.line_count = $lc",
            {"path": path, "lang": language, "lc": line_count},
        )

    def upsert_function(self, file_path: str, name: str,
                        start_line: int, end_line: int) -> str:
        if not self.available:
            return ""
        fid = f"{file_path}::{name}:{start_line}"
        self._conn.execute(
            "MERGE (fn:Function {id: $id}) "
            "SET fn.name = $name, fn.file_path = $fp, "
            "    fn.start_line = $sl, fn.end_line = $el",
            {"id": fid, "name": name, "fp": file_path, "sl": start_line, "el": end_line},
        )
        self._conn.execute(
            "MATCH (f:File {path: $fp}), (fn:Function {id: $id}) "
            "MERGE (f)-[:DEFINES]->(fn)",
            {"fp": file_path, "id": fid},
        )
        return fid

    def upsert_class(self, file_path: str, name: str,
                     start_line: int, end_line: int) -> str:
        if not self.available:
            return ""
        cid = f"{file_path}::{name}:{start_line}"
        self._conn.execute(
            "MERGE (c:Class {id: $id}) "
            "SET c.name = $name, c.file_path = $fp, "
            "    c.start_line = $sl, c.end_line = $el",
            {"id": cid, "name": name, "fp": file_path, "sl": start_line, "el": end_line},
        )
        self._conn.execute(
            "MATCH (f:File {path: $fp}), (c:Class {id: $id}) "
            "MERGE (f)-[:DEFINES_CLASS]->(c)",
            {"fp": file_path, "id": cid},
        )
        return cid

    def add_import(self, from_file: str, to_file: str) -> None:
        if not self.available:
            return
        self._conn.execute(
            "MATCH (a:File {path: $a}), (b:File {path: $b}) "
            "MERGE (a)-[:IMPORTS]->(b)",
            {"a": from_file, "b": to_file},
        )

    def add_call(self, caller_id: str, callee_id: str) -> None:
        if not self.available:
            return
        self._conn.execute(
            "MATCH (a:Function {id: $a}), (b:Function {id: $b}) "
            "MERGE (a)-[r:CALLS]->(b) "
            "ON MATCH SET r.call_count = r.call_count + 1 "
            "ON CREATE SET r.call_count = 1",
            {"a": caller_id, "b": callee_id},
        )

    def add_inheritance(self, child_id: str, parent_name: str, file_path: str) -> None:
        """Add INHERITS edge. Parent looked up by name within the same file first."""
        if not self.available:
            return
        self._conn.execute(
            "MATCH (child:Class {id: $child}), "
            "      (parent:Class {name: $pname}) "
            "WHERE parent.file_path = $fp OR parent.file_path <> '' "
            "MERGE (child)-[:INHERITS]->(parent)",
            {"child": child_id, "pname": parent_name, "fp": file_path},
        )

    def add_co_change(self, file_a: str, file_b: str) -> None:
        """Increment co-change weight between two files."""
        if not self.available:
            return
        self._conn.execute(
            "MATCH (a:File {path: $a}), (b:File {path: $b}) "
            "MERGE (a)-[r:CHANGED_TOGETHER]->(b) "
            "ON MATCH SET r.weight = r.weight + 1 "
            "ON CREATE SET r.weight = 1",
            {"a": file_a, "b": file_b},
        )

    # -----------------------------------------------------------------------
    # Convenience query methods — used by tool_graph_traverse
    # -----------------------------------------------------------------------

    def callees(self, function_name: str, depth: int = 1) -> list[str]:
        """Return functions called by function_name, up to depth hops."""
        rows = self.query(
            f"MATCH (f:Function)-[:CALLS*1..{depth}]->(g:Function) "
            "WHERE f.name = $name RETURN DISTINCT g.name AS callee, g.file_path AS fp",
            {"name": function_name},
        )
        return [f"{r['callee']} ({r['fp']})" for r in rows]

    def callers(self, function_name: str, depth: int = 1) -> list[str]:
        """Return functions that call function_name, up to depth hops."""
        rows = self.query(
            f"MATCH (f:Function)-[:CALLS*1..{depth}]->(g:Function) "
            "WHERE g.name = $name RETURN DISTINCT f.name AS caller, f.file_path AS fp",
            {"name": function_name},
        )
        return [f"{r['caller']} ({r['fp']})" for r in rows]

    def dependents(self, file_path: str) -> list[str]:
        """Return files that import file_path."""
        rows = self.query(
            "MATCH (a:File)-[:IMPORTS]->(b:File {path: $path}) RETURN a.path AS dep",
            {"path": file_path},
        )
        return [r["dep"] for r in rows]

    def dependencies(self, file_path: str) -> list[str]:
        """Return files that file_path imports."""
        rows = self.query(
            "MATCH (a:File {path: $path})-[:IMPORTS]->(b:File) RETURN b.path AS dep",
            {"path": file_path},
        )
        return [r["dep"] for r in rows]

    def co_changed_with(self, file_path: str, min_weight: int = 2) -> list[dict]:
        """Return files frequently changed together with file_path."""
        rows = self.query(
            "MATCH (a:File {path: $path})-[r:CHANGED_TOGETHER]->(b:File) "
            "WHERE r.weight >= $w "
            "RETURN b.path AS file, r.weight AS co_changes "
            "ORDER BY co_changes DESC",
            {"path": file_path, "w": min_weight},
        )
        return rows

    def symbols_in_file(self, file_path: str) -> list[dict]:
        """Return all functions and classes defined in a file."""
        fns = self.query(
            "MATCH (f:File {path: $path})-[:DEFINES]->(fn:Function) "
            "RETURN fn.name AS name, fn.start_line AS start, "
            "       fn.end_line AS end_line, 'function' AS kind",
            {"path": file_path},
        )
        cls = self.query(
            "MATCH (f:File {path: $path})-[:DEFINES_CLASS]->(c:Class) "
            "RETURN c.name AS name, c.start_line AS start, "
            "       c.end_line AS end_line, 'class' AS kind",
            {"path": file_path},
        )
        return sorted(fns + cls, key=lambda x: x.get("start", 0))


# ---------------------------------------------------------------------------
# Graph construction — called from ingest.py
# ---------------------------------------------------------------------------

def build_dependency_graph(
    files: list[dict],
    graph_store: KuzuGraphStore,
    repo_path: str,
) -> None:
    """
    Build the code dependency graph from already-discovered files.
    Runs the tree-sitter parse (same as chunking) to extract symbols and
    import relationships, then writes them to the Kuzu graph.

    files:       output of ingest.discover_files()
    graph_store: initialised KuzuGraphStore
    repo_path:   repo root (for resolving relative import paths)
    """
    if not graph_store.available:
        logger.warning("Graph store unavailable — skipping dependency graph build")
        return

    logger.info(f"Building dependency graph for {len(files)} files...")
    repo_root = Path(repo_path)
    # Index relative paths for import resolution
    all_rel_paths: set[str] = {f["relative_path"] for f in files}

    for file_info in files:
        rel = file_info["relative_path"]
        lang = file_info.get("language") or "text"

        try:
            with open(file_info["path"], "rb") as f:
                source_bytes = f.read()
            source_text = source_bytes.decode("utf-8", errors="replace")
            line_count = source_text.count("\n") + 1
        except Exception as e:
            logger.warning(f"Could not read {rel}: {e}")
            continue

        graph_store.upsert_file(rel, lang, line_count)

        if file_info["type"] != "code":
            continue

        # --- Parse symbols via tree-sitter ---
        try:
            from tree_sitter_language_pack import get_parser
            parser = get_parser(lang)
            tree = parser.parse(source_bytes)

            fn_ids: dict[str, str] = {}   # name → node id (for call resolution)
            cls_ids: dict[str, str] = {}  # name → node id

            def _walk(node, current_fn_id: str | None = None):
                if node.type in ("function_definition", "function_declaration",
                                 "method_definition"):
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        name = name_node.text.decode()
                        fid = graph_store.upsert_function(
                            rel, name,
                            node.start_point[0] + 1,
                            node.end_point[0] + 1,
                        )
                        fn_ids[name] = fid
                        # Recurse with this function as context for call detection
                        for child in node.children:
                            _walk(child, fid)
                        return

                elif node.type == "class_definition":
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        name = name_node.text.decode()
                        cid = graph_store.upsert_class(
                            rel, name,
                            node.start_point[0] + 1,
                            node.end_point[0] + 1,
                        )
                        cls_ids[name] = cid
                        # Check for base classes (Python: class Foo(Bar))
                        args = node.child_by_field_name("superclasses") or \
                               node.child_by_field_name("arguments")
                        if args:
                            for base in args.children:
                                if base.type == "identifier":
                                    graph_store.add_inheritance(cid, base.text.decode(), rel)

                elif node.type == "call" and current_fn_id:
                    # Extract callee name from call expression
                    fn_node = node.child_by_field_name("function")
                    if fn_node:
                        callee_name = fn_node.text.decode().split(".")[-1]
                        # We'll wire calls after all functions are registered (post-pass)
                        # Store as deferred edge
                        _deferred_calls.append((current_fn_id, callee_name))

                for child in node.children:
                    _walk(child, current_fn_id)

            _deferred_calls: list[tuple[str, str]] = []
            _walk(tree.root_node)

            # Resolve deferred call edges
            for caller_id, callee_name in _deferred_calls:
                if callee_name in fn_ids:
                    graph_store.add_call(caller_id, fn_ids[callee_name])

        except Exception as e:
            logger.debug(f"tree-sitter parse failed for {rel} ({lang}): {e}")

        # --- Extract imports ---
        _extract_imports(source_text, rel, lang, all_rel_paths, repo_root, graph_store)

    logger.info("Dependency graph build complete")


def _extract_imports(
    source: str,
    file_rel_path: str,
    language: str,
    all_rel_paths: set[str],
    repo_root: Path,
    graph_store: KuzuGraphStore,
) -> None:
    """
    Extract import statements and resolve to relative file paths within the repo.
    Only adds IMPORTS edges for files that exist in the repo (skips stdlib/third-party).
    """
    current_dir = Path(file_rel_path).parent

    if language == "python":
        # Match: import foo, from foo import bar, from .foo import bar
        for m in re.finditer(
            r"^(?:from\s+([\w.]+)\s+import|import\s+([\w.,\s]+))", source, re.MULTILINE
        ):
            module = (m.group(1) or m.group(2) or "").strip().split()[0]
            if not module:
                continue
            # Convert module path to file path candidates
            candidates = [
                module.replace(".", "/") + ".py",
                module.replace(".", "/") + "/__init__.py",
                str(current_dir / (module.lstrip(".").replace(".", "/") + ".py")),
            ]
            for candidate in candidates:
                # Normalise
                try:
                    norm = str(Path(candidate))
                except Exception:
                    continue
                if norm in all_rel_paths:
                    graph_store.add_import(file_rel_path, norm)
                    break

    elif language in ("javascript", "typescript"):
        for m in re.finditer(r'(?:import|require)\s*[(\'"](\.\.?/[^\'")]+)[\'")]', source):
            raw = m.group(1)
            for ext in (".ts", ".js", ".tsx", ".jsx", "/index.ts", "/index.js"):
                candidate = str((current_dir / raw).with_suffix("") ) + ext
                try:
                    norm = str(Path(candidate))
                except Exception:
                    continue
                if norm in all_rel_paths:
                    graph_store.add_import(file_rel_path, norm)
                    break


def build_co_change_graph(
    graph_store: KuzuGraphStore,
    repo_path: str,
    n_commits: int = 100,
) -> None:
    """
    Build the co-change graph from git history.
    For each pair of files changed in the same commit, increments their
    CHANGED_TOGETHER edge weight.

    n_commits: how many recent commits to analyse (default: 100)
    """
    if not graph_store.available:
        return

    logger.info(f"Building co-change graph from last {n_commits} commits...")
    try:
        result = subprocess.run(
            ["git", "log", f"-{n_commits}", "--name-only", "--pretty=format:COMMIT"],
            cwd=repo_path, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            logger.warning("git log failed — skipping co-change graph")
            return

        # Parse: group files by commit
        commits: list[list[str]] = []
        current: list[str] = []
        for line in result.stdout.splitlines():
            if line == "COMMIT":
                if current:
                    commits.append(current)
                current = []
            elif line.strip():
                current.append(line.strip())
        if current:
            commits.append(current)

        # For each commit, add co-change edges for all file pairs
        edge_count = 0
        for commit_files in commits:
            # Only files we know about (in the graph already)
            for i, fa in enumerate(commit_files):
                for fb in commit_files[i + 1:]:
                    if fa != fb:
                        graph_store.add_co_change(fa, fb)
                        graph_store.add_co_change(fb, fa)
                        edge_count += 1

        logger.info(f"Co-change graph: {len(commits)} commits, {edge_count} edge pairs")

    except Exception as e:
        logger.warning(f"Co-change graph build failed: {e}")
