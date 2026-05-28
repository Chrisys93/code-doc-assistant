"""
test_pipeline.py — End-to-end pipeline validation and unit tests.

Covers:
  1. Config resolution — all tiers, quantisation, backends, HNSW, deployment targets
  2. File discovery and chunking
  3. Vector store (ChromaDB in-process, HNSW metadata)
  4. Graph store (Kuzu, all query types)
  5. Tool registry — local tools, MCP capability gating, graph_traverse
  6. Agent state — dataclass integrity, SessionPreferences.update()
  7. Ingestion pipeline — end-to-end with local embeddings
  8. Retrieval quality — query → chunk hit testing
  9. Inference backend factory — _get_llm() returns correct type per backend

Services NOT required to run this test suite:
  - Ollama / vLLM / llama-server  (LLM calls are mocked)
  - ChromaDB server               (in-process chromadb.Client() used)
  - MLflow                        (not called in tests)
  - Kuzu                          (in-process, temp directory)

Usage:
  python test_pipeline.py                          # all tests
  python test_pipeline.py TestConfig               # one class
  MODEL_TIER=minimal python test_pipeline.py       # test minimal tier
  DEPLOYMENT_TARGET=cluster python test_pipeline.py
"""

from __future__ import annotations

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src is importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Local embedding (no Ollama dependency)
# ---------------------------------------------------------------------------

from llama_index.core.embeddings import BaseEmbedding
from pydantic import PrivateAttr


class LocalTestEmbedding(BaseEmbedding):
    """
    Character trigram hashing embedding for pipeline testing.
    Deterministic, vocabulary-based. NOT for production.
    """
    _dim: int = PrivateAttr(default=384)

    def __init__(self, dim: int = 384, **kwargs):
        super().__init__(**kwargs)
        self._dim = dim

    @classmethod
    def class_name(cls) -> str:
        return "LocalTestEmbedding"

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float64)
        text = text.lower()
        for i in range(len(text) - 2):
            idx = hash(text[i:i+3]) % self._dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)


# ============================================================
# 1. Config resolution tests
# ============================================================

class TestConfig(unittest.TestCase):
    """Test config.py resolves all tiers, backends, and knobs correctly."""

    def _reload_config(self, env: dict) -> object:
        """Reload config module with a patched environment."""
        import importlib
        with patch.dict(os.environ, env, clear=False):
            import config
            importlib.reload(config)
            return config

    def test_full_tier_default_quant(self):
        cfg = self._reload_config({"MODEL_TIER": "full", "QUANTISATION": "q4_K_M"})
        self.assertEqual(cfg._resolve_ollama_model(), "mistral-nemo:12b-instruct-q4_K_M")

    def test_balanced_tier_updated_model(self):
        cfg = self._reload_config({"MODEL_TIER": "balanced", "QUANTISATION": "q4_K_M"})
        self.assertIn("deepseek-coder-v2", cfg._resolve_ollama_model())
        self.assertIn("q4_K_M", cfg._resolve_ollama_model())

    def test_lightweight_no_quant_suffix(self):
        cfg = self._reload_config({"MODEL_TIER": "lightweight", "QUANTISATION": "q4_K_M"})
        self.assertEqual(cfg._resolve_ollama_model(), "phi3.5")

    def test_minimal_no_quant_suffix(self):
        cfg = self._reload_config({"MODEL_TIER": "minimal", "QUANTISATION": "q4_K_M"})
        self.assertEqual(cfg._resolve_ollama_model(), "qwen2.5-coder:3b-instruct")

    def test_full_fp16_no_suffix(self):
        cfg = self._reload_config({"MODEL_TIER": "full", "QUANTISATION": "fp16"})
        self.assertEqual(cfg._resolve_ollama_model(), "mistral-nemo:12b-instruct")

    def test_minimal_chunking_strategy(self):
        cfg = self._reload_config({"MODEL_TIER": "minimal"})
        self.assertEqual(cfg.CHUNKING_STRATEGY, "text")

    def test_full_chunking_strategy(self):
        cfg = self._reload_config({"MODEL_TIER": "full"})
        self.assertEqual(cfg.CHUNKING_STRATEGY, "ast")

    def test_local_hnsw_search_ef_reduced(self):
        cfg = self._reload_config({"DEPLOYMENT_TARGET": "local"})
        self.assertEqual(cfg.CHROMA_HNSW_SEARCH_EF, 20)

    def test_cluster_hnsw_search_ef_full(self):
        cfg = self._reload_config({"DEPLOYMENT_TARGET": "cluster"})
        self.assertEqual(cfg.CHROMA_HNSW_SEARCH_EF, 50)

    def test_llamacpp_backend_resolved(self):
        cfg = self._reload_config({"INFERENCE_BACKEND": "llamacpp"})
        self.assertEqual(cfg.INFERENCE_BACKEND, "llamacpp")
        self.assertIsNotNone(cfg.LLAMACPP_HOST)
        self.assertIsNotNone(cfg.LLAMACPP_MODEL)

    def test_embedding_dimension_resolution(self):
        cfg = self._reload_config({"EMBEDDING_MODEL": "all-minilm"})
        self.assertEqual(cfg.EMBEDDING_DIMENSION, 384)

        cfg = self._reload_config({"EMBEDDING_MODEL": "mxbai-embed-large"})
        self.assertEqual(cfg.EMBEDDING_DIMENSION, 1024)

    def test_mcp_flags_default_false(self):
        cfg = self._reload_config({})
        self.assertFalse(cfg.MCP_FILESYSTEM_ENABLED)
        self.assertFalse(cfg.MCP_GITHUB_ENABLED)
        self.assertFalse(cfg.MCP_SLACK_ENABLED)


# ============================================================
# 2. File discovery and chunking
# ============================================================

class TestDiscovery(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.repo_path = os.path.dirname(os.path.abspath(__file__))

    def test_discovers_python_files(self):
        from ingest import discover_files
        files = discover_files(self.repo_path)
        code_files = [f for f in files if f["type"] == "code"]
        self.assertGreater(len(code_files), 0)
        py_files = [f for f in code_files if f["language"] == "python"]
        self.assertGreater(len(py_files), 0)

    def test_discovers_text_files(self):
        from ingest import discover_files
        files = discover_files(self.repo_path)
        text_files = [f for f in files if f["type"] == "text"]
        self.assertGreater(len(text_files), 0)

    def test_skips_git_and_pycache(self):
        from ingest import discover_files
        files = discover_files(self.repo_path)
        paths = [f["relative_path"] for f in files]
        self.assertFalse(any(".git" in p for p in paths))
        self.assertFalse(any("__pycache__" in p for p in paths))

    def test_chunking_produces_nodes(self):
        from ingest import discover_files, load_and_chunk_files
        files = discover_files(self.repo_path)
        nodes = load_and_chunk_files(files)
        self.assertGreater(len(nodes), 0)

    def test_chunk_metadata_present(self):
        from ingest import discover_files, load_and_chunk_files
        files = discover_files(self.repo_path)
        nodes = load_and_chunk_files(files)
        for node in nodes[:5]:
            self.assertIn("file_path", node.metadata)
            self.assertIn("file_type", node.metadata)


# ============================================================
# 3. Vector store — HNSW metadata applied
# ============================================================

class TestVectorStore(unittest.TestCase):

    def test_hnsw_metadata_local(self):
        """HNSW metadata dict uses reduced searchEf for local deployment."""
        with patch.dict(os.environ, {"DEPLOYMENT_TARGET": "local"}, clear=False):
            import importlib
            import config as cfg_mod
            importlib.reload(cfg_mod)
            import vector_store as vs_mod
            importlib.reload(vs_mod)
            impl = vs_mod.ChromaVectorStoreImpl.__new__(vs_mod.ChromaVectorStoreImpl)
            impl._host = "http://localhost:8000"
            impl._collection_name = "test"
            meta = impl._hnsw_metadata()
            self.assertEqual(meta["hnsw:space"], "cosine")
            self.assertIn("hnsw:M", meta)
            self.assertIn("hnsw:construction_ef", meta)
            self.assertIn("hnsw:search_ef", meta)
            self.assertEqual(meta["hnsw:search_ef"], 20)

    def test_hnsw_metadata_cluster(self):
        """HNSW metadata dict uses full searchEf for cluster deployment."""
        with patch.dict(os.environ, {"DEPLOYMENT_TARGET": "cluster"}, clear=False):
            import importlib
            import config as cfg_mod
            importlib.reload(cfg_mod)
            import vector_store as vs_mod
            importlib.reload(vs_mod)
            impl = vs_mod.ChromaVectorStoreImpl.__new__(vs_mod.ChromaVectorStoreImpl)
            meta = impl._hnsw_metadata()
            self.assertEqual(meta["hnsw:search_ef"], 50)

    def test_in_process_chroma_roundtrip(self):
        """Full embed → store → retrieve roundtrip using in-process ChromaDB."""
        import chromadb
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from ingest import discover_files, load_and_chunk_files

        repo_path = os.path.dirname(os.path.abspath(__file__))
        files = discover_files(repo_path)
        nodes = load_and_chunk_files(files)

        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name="test_hnsw",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:M": 16,
                "hnsw:construction_ef": 100,
                "hnsw:search_ef": 20,
            },
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = LocalTestEmbedding(dim=384)

        index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=False,
        )
        self.assertGreater(collection.count(), 0)

        retriever = index.as_retriever(similarity_top_k=3)
        results = retriever.retrieve("model tier configuration")
        self.assertGreater(len(results), 0)


# ============================================================
# 4. Graph store (Kuzu)
# ============================================================

class TestGraphStore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            import kuzu
            cls.kuzu_available = True
        except ImportError:
            cls.kuzu_available = False

        cls.tmpdir = tempfile.mkdtemp(prefix="test_graph_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        if not self.kuzu_available:
            self.skipTest("kuzu not installed — skipping graph store tests")

    def _make_store(self):
        from graph_store import KuzuGraphStore
        # Each test gets a unique subdirectory — Kuzu cannot share a path across instances
        import uuid
        graph_path = os.path.join(self.tmpdir, f"g_{uuid.uuid4().hex[:8]}")
        return KuzuGraphStore(graph_path)

    def test_graph_store_initialises(self):
        gs = self._make_store()
        self.assertTrue(gs.available)

    def test_upsert_file(self):
        gs = self._make_store()
        gs.upsert_file("src/app.py", "python", 200)
        rows = gs.query("MATCH (f:File {path: 'src/app.py'}) RETURN f.language AS lang")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["lang"], "python")

    def test_upsert_function_and_defines_edge(self):
        gs = self._make_store()
        gs.upsert_file("src/tools.py", "python", 500)
        fid = gs.upsert_function("src/tools.py", "tool_grep", 10, 50)
        self.assertIn("tool_grep", fid)
        symbols = gs.symbols_in_file("src/tools.py")
        names = [s["name"] for s in symbols]
        self.assertIn("tool_grep", names)

    def test_import_edge(self):
        gs = self._make_store()
        gs.upsert_file("src/a.py", "python", 10)
        gs.upsert_file("src/b.py", "python", 10)
        gs.add_import("src/a.py", "src/b.py")
        deps = gs.dependencies("src/a.py")
        self.assertIn("src/b.py", deps)
        dependents = gs.dependents("src/b.py")
        self.assertIn("src/a.py", dependents)

    def test_call_edge(self):
        gs = self._make_store()
        gs.upsert_file("src/x.py", "python", 20)
        caller_id = gs.upsert_function("src/x.py", "caller_fn", 1, 10)
        callee_id = gs.upsert_function("src/x.py", "callee_fn", 12, 20)
        gs.add_call(caller_id, callee_id)
        callees = gs.callees("caller_fn")
        self.assertTrue(any("callee_fn" in c for c in callees))
        callers = gs.callers("callee_fn")
        self.assertTrue(any("caller_fn" in c for c in callers))

    def test_co_change_edge(self):
        gs = self._make_store()
        gs.upsert_file("src/p.py", "python", 10)
        gs.upsert_file("src/q.py", "python", 10)
        gs.add_co_change("src/p.py", "src/q.py")
        gs.add_co_change("src/p.py", "src/q.py")
        rows = gs.co_changed_with("src/p.py", min_weight=2)
        files = [r["file"] for r in rows]
        self.assertIn("src/q.py", files)

    def test_raw_cypher_query(self):
        gs = self._make_store()
        gs.upsert_file("src/z.py", "python", 5)
        rows = gs.query("MATCH (f:File) WHERE f.path = $p RETURN f.line_count AS lc",
                        {"p": "src/z.py"})
        self.assertEqual(rows[0]["lc"], 5)

    def test_reset_clears_data(self):
        gs = self._make_store()
        gs.upsert_file("src/tmp.py", "python", 1)
        gs.reset()
        rows = gs.query("MATCH (f:File) RETURN count(f) AS n")
        self.assertEqual(rows[0]["n"], 0)


# ============================================================
# 5. Tool registry — local tools, MCP gating, graph_traverse
# ============================================================

class TestToolRegistry(unittest.TestCase):

    def test_local_tools_always_present(self):
        from tools import LOCAL_TOOL_REGISTRY
        for expected in ["grep", "cat", "find", "git_log", "git_blame",
                         "stat", "vector_search", "ast_parse", "github_fetch",
                         "graph_traverse"]:
            self.assertIn(expected, LOCAL_TOOL_REGISTRY, f"Missing local tool: {expected}")

    def test_mcp_tools_in_mcp_registry(self):
        from tools import MCP_TOOL_REGISTRY
        for expected in ["fs_read", "fs_write", "fs_list",
                         "github_get_file", "github_list_prs",
                         "slack_post", "slack_get_channel_history"]:
            self.assertIn(expected, MCP_TOOL_REGISTRY, f"Missing MCP tool: {expected}")

    def test_is_mcp_capable_full(self):
        with patch.dict(os.environ, {"MODEL_TIER": "full"}):
            import importlib, tools
            importlib.reload(tools)
            self.assertTrue(tools.is_mcp_capable())

    def test_is_mcp_capable_balanced(self):
        with patch.dict(os.environ, {"MODEL_TIER": "balanced"}):
            import importlib, tools
            importlib.reload(tools)
            self.assertTrue(tools.is_mcp_capable())

    def test_is_mcp_capable_lightweight_false(self):
        with patch.dict(os.environ, {"MODEL_TIER": "lightweight"}):
            import importlib, tools
            importlib.reload(tools)
            self.assertFalse(tools.is_mcp_capable())

    def test_is_mcp_capable_minimal_false(self):
        with patch.dict(os.environ, {"MODEL_TIER": "minimal"}):
            import importlib, tools
            importlib.reload(tools)
            self.assertFalse(tools.is_mcp_capable())

    def test_build_registry_minimal_excludes_mcp(self):
        """minimal tier → build_tool_registry() returns only local tools."""
        with patch.dict(os.environ, {"MODEL_TIER": "minimal"}):
            import importlib, tools
            importlib.reload(tools)
            registry = tools.build_tool_registry()
            for tool_name in registry:
                self.assertEqual(registry[tool_name]["type"], "local",
                                 f"MCP tool {tool_name!r} appeared in minimal registry")

    def test_build_registry_full_includes_mcp_when_enabled(self):
        """full tier + MCP server enabled → MCP tool appears in registry."""
        with patch.dict(os.environ, {
            "MODEL_TIER": "full",
            "MCP_SLACK_ENABLED": "true",
        }):
            import importlib, tools
            importlib.reload(tools)
            registry = tools.build_tool_registry()
            mcp_tools = [n for n, e in registry.items() if e["type"] == "mcp"]
            self.assertTrue(len(mcp_tools) > 0, "No MCP tools in full-tier registry with server enabled")

    def test_graph_traverse_tool_runs_gracefully_without_kuzu(self):
        """graph_traverse returns a clean error when kuzu not available."""
        from tools import tool_graph_traverse
        with patch.dict(os.environ, {"GRAPH_PATH": "/tmp/nonexistent_graph_xyz"}):
            result = tool_graph_traverse("symbols", "src/app.py")
            # Should return success=False with a descriptive message, not raise
            self.assertFalse(result["success"])
            self.assertIsInstance(result["result"], str)

    def test_run_tool_unknown_name(self):
        from tools import run_tool
        result = run_tool("definitely_not_a_real_tool", {})
        self.assertFalse(result["success"])
        self.assertIn("Unknown tool", result["result"])

    def test_run_tool_latency_always_present(self):
        from tools import run_tool
        result = run_tool("find", {"repo_path": os.path.dirname(os.path.abspath(__file__))})
        self.assertIn("latency_ms", result)
        self.assertIsInstance(result["latency_ms"], float)


# ============================================================
# 6. Agent state — dataclass integrity
# ============================================================

class TestAgentState(unittest.TestCase):

    def test_session_preferences_update(self):
        from agent_state import SessionPreferences, PostGenerationFeedback
        prefs = SessionPreferences()
        fb1 = PostGenerationFeedback(
            response_shown="test",
            decision="accept",
            satisfaction_score=4,
            format_notes="NumPy docstrings",
            additional_files=["src/config.py"],
        )
        prefs.update(fb1)
        self.assertEqual(prefs.feedback_count, 1)
        self.assertAlmostEqual(prefs.avg_satisfaction, 4.0)
        self.assertEqual(prefs.preferred_format, "NumPy docstrings")
        self.assertIn("src/config.py", prefs.prioritised_files)

        fb2 = PostGenerationFeedback(
            response_shown="test2",
            decision="accept",
            satisfaction_score=2,
        )
        prefs.update(fb2)
        self.assertEqual(prefs.feedback_count, 2)
        self.assertAlmostEqual(prefs.avg_satisfaction, 3.0)
        # format_notes unchanged (fb2 had none)
        self.assertEqual(prefs.preferred_format, "NumPy docstrings")

    def test_tool_call_dataclass(self):
        from agent_state import ToolCall
        tc = ToolCall(tool_name="grep", args={"pattern": "def main"})
        self.assertIsNone(tc.result)
        self.assertFalse(tc.success)
        tc.result = "found it"
        tc.success = True
        self.assertEqual(tc.result, "found it")

    def test_hitl_checkpoint_dataclass(self):
        from agent_state import HITLCheckpoint, ToolCall
        tc = ToolCall(tool_name="cat", args={"file_path": "src/app.py", "repo_path": "."})
        checkpoint = HITLCheckpoint(proposed_tool_calls=[tc])
        self.assertIsNone(checkpoint.decision)
        checkpoint.decision = "approved"
        self.assertEqual(checkpoint.decision, "approved")

    def test_post_generation_feedback_defaults(self):
        from agent_state import PostGenerationFeedback
        fb = PostGenerationFeedback(response_shown="hello")
        self.assertEqual(fb.decision, "accept")
        self.assertEqual(fb.satisfaction_score, 5)
        self.assertEqual(fb.additional_files, [])


# ============================================================
# 7. Inference backend factory — _get_llm() returns correct type
# ============================================================

class TestInferenceBackend(unittest.TestCase):

    def _get_llm_for_backend(self, backend: str):
        """Reload agent_graph with the given backend and return _get_llm()."""
        with patch.dict(os.environ, {"INFERENCE_BACKEND": backend}):
            import importlib, agent_graph
            importlib.reload(agent_graph)
            return agent_graph._get_llm(temperature=0.1)

    def test_ollama_backend_returns_chat_ollama(self):
        from langchain_ollama import ChatOllama
        llm = self._get_llm_for_backend("ollama")
        self.assertIsInstance(llm, ChatOllama)

    def test_vllm_backend_returns_chat_openai(self):
        from langchain_openai import ChatOpenAI
        llm = self._get_llm_for_backend("vllm")
        self.assertIsInstance(llm, ChatOpenAI)
        self.assertIn("8080", llm.openai_api_base)

    def test_llamacpp_backend_returns_chat_openai(self):
        from langchain_openai import ChatOpenAI
        llm = self._get_llm_for_backend("llamacpp")
        self.assertIsInstance(llm, ChatOpenAI)
        self.assertIn("8081", llm.openai_api_base)

    def test_llamacpp_max_tokens_conservative(self):
        """llama-server context is GGUF-defined; default max_tokens should be <= 2048."""
        from langchain_openai import ChatOpenAI
        llm = self._get_llm_for_backend("llamacpp")
        self.assertLessEqual(llm.max_tokens, 2048)

    def test_vllm_and_llamacpp_different_hosts(self):
        """vLLM and llama-server must point at different ports."""
        from langchain_openai import ChatOpenAI
        vllm_llm = self._get_llm_for_backend("vllm")
        llamacpp_llm = self._get_llm_for_backend("llamacpp")
        self.assertNotEqual(vllm_llm.openai_api_base, llamacpp_llm.openai_api_base)


# ============================================================
# 8. End-to-end retrieval quality
# ============================================================

class TestRetrievalQuality(unittest.TestCase):
    """
    Full ingestion → retrieval roundtrip using in-process ChromaDB
    and local trigram embeddings (no external services required).
    """

    @classmethod
    def setUpClass(cls):
        import chromadb
        from llama_index.core import VectorStoreIndex, StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from ingest import discover_files, load_and_chunk_files

        repo_path = os.path.dirname(os.path.abspath(__file__))
        files = discover_files(repo_path)
        nodes = load_and_chunk_files(files)

        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name="retrieval_test",
            metadata={"hnsw:space": "cosine", "hnsw:M": 16,
                      "hnsw:construction_ef": 100, "hnsw:search_ef": 20},
        )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        embed_model = LocalTestEmbedding(dim=384)

        cls.index = VectorStoreIndex(
            nodes=nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=False,
        )
        cls.retriever = cls.index.as_retriever(similarity_top_k=5)

    def _assert_retrieves(self, query: str, expected_files: list[str]):
        results = self.retriever.retrieve(query)
        retrieved = [r.metadata.get("file_path", "") for r in results]
        hit = any(
            any(exp in rf for rf in retrieved)
            for exp in expected_files
        )
        self.assertTrue(
            hit,
            f"Query '{query}' did not retrieve any of {expected_files}. Got: {retrieved}"
        )

    def test_retrieves_config_for_model_tier(self):
        self._assert_retrieves("model tier configuration", ["config.py"])

    def test_retrieves_ingest_for_file_discovery(self):
        self._assert_retrieves("file discovery codebase", ["ingest.py"])

    def test_retrieves_vector_store_for_chromadb(self):
        self._assert_retrieves("chromadb collection hnsw", ["vector_store.py"])

    def test_retrieves_tools_for_grep(self):
        self._assert_retrieves("grep search pattern codebase", ["tools.py"])

    def test_retrieves_graph_store_for_kuzu(self):
        self._assert_retrieves("kuzu graph dependency", ["graph_store.py"])

    def test_retrieves_agent_graph_for_langgraph(self):
        self._assert_retrieves("langgraph supervisor node", ["agent_graph.py"])

    def test_retrieves_agent_state_for_hitl(self):
        self._assert_retrieves("HITL human review feedback", ["agent_state.py"])

    def test_at_least_half_queries_pass(self):
        """Regression guard: at least 50% of queries hit expected files."""
        queries = [
            ("model tier configuration", ["config.py"]),
            ("file discovery codebase", ["ingest.py"]),
            ("chromadb collection hnsw", ["vector_store.py"]),
            ("grep search pattern", ["tools.py"]),
            ("kuzu graph", ["graph_store.py"]),
            ("langgraph supervisor", ["agent_graph.py"]),
        ]
        passed = 0
        for query, expected in queries:
            results = self.retriever.retrieve(query)
            retrieved = [r.metadata.get("file_path", "") for r in results]
            if any(any(exp in rf for rf in retrieved) for exp in expected):
                passed += 1
        self.assertGreaterEqual(passed, len(queries) // 2,
                                f"Retrieval quality too low: {passed}/{len(queries)}")


# ============================================================
# 9. Minimal deployment — integration check
# ============================================================

class TestMinimalDeployment(unittest.TestCase):
    """
    Verify the minimal tier produces a consistent, internally coherent
    configuration across all components.
    """

    def setUp(self):
        self.env = {
            "MODEL_TIER": "minimal",
            "QUANTISATION": "q4_K_M",
            "DEPLOYMENT_TARGET": "local",
            "INFERENCE_BACKEND": "llamacpp",
        }

    def _reload(self):
        import importlib, config
        with patch.dict(os.environ, self.env):
            importlib.reload(config)
            return config

    def test_minimal_model_tag_no_suffix(self):
        cfg = self._reload()
        self.assertEqual(cfg._resolve_ollama_model(), "qwen2.5-coder:3b-instruct")

    def test_minimal_chunking_is_text(self):
        cfg = self._reload()
        self.assertEqual(cfg.CHUNKING_STRATEGY, "text")

    def test_minimal_hnsw_search_ef_reduced(self):
        cfg = self._reload()
        self.assertEqual(cfg.CHROMA_HNSW_SEARCH_EF, 20)

    def test_minimal_mcp_excluded_from_registry(self):
        with patch.dict(os.environ, self.env):
            import importlib, tools
            importlib.reload(tools)
            registry = tools.build_tool_registry()
            for tool_name, entry in registry.items():
                self.assertEqual(entry["type"], "local",
                                 f"MCP tool {tool_name!r} leaked into minimal registry")

    def test_minimal_graph_traverse_in_registry(self):
        """graph_traverse must be available even in minimal tier."""
        with patch.dict(os.environ, self.env):
            import importlib, tools
            importlib.reload(tools)
            registry = tools.build_tool_registry()
            self.assertIn("graph_traverse", registry)
            self.assertEqual(registry["graph_traverse"]["type"], "local")

    def test_minimal_llamacpp_backend(self):
        """minimal + llamacpp → _get_llm returns ChatOpenAI pointed at llama-server."""
        from langchain_openai import ChatOpenAI
        with patch.dict(os.environ, self.env):
            import importlib, agent_graph
            importlib.reload(agent_graph)
            llm = agent_graph._get_llm()
            self.assertIsInstance(llm, ChatOpenAI)
            self.assertIn("8081", llm.openai_api_base)


# ============================================================
# Runner
# ============================================================

def run_smoke_test():
    """
    Lightweight smoke test — prints a summary without unittest verbosity.
    Run with: python test_pipeline.py smoke
    """
    import importlib

    print("\n" + "=" * 60)
    print("CODE DOCUMENTATION ASSISTANT — Smoke Test")
    print("=" * 60)

    results = {}

    # Config
    try:
        import config
        importlib.reload(config)
        config.log_config()
        results["config"] = f"✅ MODEL_TIER={config.MODEL_TIER} BACKEND={config.INFERENCE_BACKEND}"
    except Exception as e:
        results["config"] = f"❌ {e}"

    # File discovery
    try:
        from ingest import discover_files
        files = discover_files(os.path.dirname(os.path.abspath(__file__)))
        code = sum(1 for f in files if f["type"] == "code")
        results["discovery"] = f"✅ {len(files)} files ({code} code)"
    except Exception as e:
        results["discovery"] = f"❌ {e}"

    # Graph store
    try:
        import kuzu
        tmpdir = tempfile.mkdtemp()
        from graph_store import KuzuGraphStore
        gs = KuzuGraphStore(tmpdir)
        results["graph_store"] = f"✅ Kuzu available, graph initialised"
        shutil.rmtree(tmpdir)
    except ImportError:
        results["graph_store"] = "⚠️  kuzu not installed — graph features disabled"
    except Exception as e:
        results["graph_store"] = f"❌ {e}"

    # Tool registry
    try:
        import tools
        importlib.reload(tools)
        reg = tools.build_tool_registry()
        mcp = sum(1 for e in reg.values() if e["type"] == "mcp")
        results["tools"] = (
            f"✅ {len(reg)} tools "
            f"({len(reg)-mcp} local, {mcp} MCP) "
            f"MCP-capable={tools.is_mcp_capable()}"
        )
    except Exception as e:
        results["tools"] = f"❌ {e}"

    print("\n--- Results ---")
    for component, status in results.items():
        print(f"  {component:15s}: {status}")

    failed = sum(1 for s in results.values() if s.startswith("❌"))
    print(f"\n{'✅ All checks passed' if failed == 0 else f'❌ {failed} check(s) failed'}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        sys.exit(run_smoke_test())
    else:
        unittest.main(verbosity=2)
