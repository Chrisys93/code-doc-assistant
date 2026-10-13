"""
test_tool_plan.py -- vector-store errors must not look like "no match", and tools that cannot run are not planned.

    python test_tool_plan.py        (from src/)

Needs no Chroma, Ollama or LLM: a fake `chromadb` and a fake embedder are injected.
"""
import os, sys, tempfile, types
os.environ.update(INFERENCE_BACKEND="llamacpp", LLAMACPP_CTX="4096", LLAMACPP_HOST="http://127.0.0.1:9")
sys.path.insert(0, ".")
sys.path.insert(0, os.path.join("..", "eval"))
sys.path.insert(0, os.path.join(".", "eval"))
for _mod, _cls in (("langchain_ollama", "ChatOllama"), ("langchain_openai", "ChatOpenAI")):
    try: __import__(_mod)
    except ImportError:
        _m = types.ModuleType(_mod); setattr(_m, _cls, object); sys.modules[_mod] = _m

try:
    import mlflow  # noqa: F401
except ImportError:      # the tests need no tracking server; a laptop without mlflow can still run them
    from unittest import mock
    sys.modules["mlflow"] = mock.MagicMock()
# fakes: the store and the embedder
chroma = types.ModuleType("chromadb"); sys.modules["chromadb"] = chroma
emb = types.ModuleType("embedding")
emb.get_embed_model = lambda: types.SimpleNamespace(get_query_embedding=lambda q: [0.0, 0.0, 0.0])
sys.modules["embedding"] = emb

class _Col:
    def count(self): return 2
    def query(self, **k):
        return {"documents": [["alpha beta", "gamma delta"]],
                "metadatas": [[{"file_path": "a.py"}, {"file_path": "b.py"}]], "distances": [[0.1, 0.2]]}

def store(behaviour: dict):
    class C:
        def __init__(self, *a, **k): pass
        def get_collection(self, name):
            b = behaviour[name]
            if b == "ok": return _Col()
            if b == "missing": raise ValueError(f"Collection {name} does not exist.")
            raise ConnectionError("connection refused")
    chroma.HttpClient = C

import tools
import agent_graph as ag
from agent_state import ToolCall

fails = []
def check(n, c, x=""):
    print(("PASS " if c else "FAIL ") + n + (f" [{x}]" if (x and not c) else ""))
    if not c: fails.append(n)

A, B = "repo_a_11111111", "repo_b_22222222"
def search(behaviour):
    store(behaviour)
    return tools.tool_vector_search("q", "http://chromadb:8000", collections=[A, B], top_k=5, score_threshold=0.0)

check("a missing collection is recognised", tools._is_missing_collection(ValueError("Collection x does not exist.")))
check("a connection error is not a missing collection", not tools._is_missing_collection(ConnectionError("connection refused")))

r = search({A: "down", B: "down"})
check("store unreachable for every collection -> FAILS (was: ok with 0 chunks)", r["success"] is False and "vector store error" in r["error"], str(r))
check("... and names each failed collection", len(r["collections_failed"]) == 2, str(r))

r = search({A: "missing", B: "ok"})
check("one repo not indexed, one fine -> success, hits from the fine one", r["success"] and r["count"] == 2 and r["collections_searched"] == [B], str(r))
check("... the missing one is reported as missing, not failed", r["collections_missing"] == [A] and r["collections_failed"] == [], str(r))

r = search({A: "down", B: "ok"})
check("one store error, one fine -> success with hits and the error reported", r["success"] and r["count"] == 2 and len(r["collections_failed"]) == 1, str(r))

r = search({A: "missing", B: "missing"})
check("nothing indexed -> still success with 0 chunks (an honest 'no match'), not an error", r["success"] and r["count"] == 0 and r["collections_failed"] == [], str(r))

# the graph surfaces it
store({A: "down", B: "down"})
tc = ToolCall(tool_name="vector_search", args={"query": "q", "chroma_host": "http://chromadb:8000"})
out = ag.node_tool_execution({"query": "q", "active_collections": [A, B], "approved_tool_calls": [tc],
                              "execution_trace": [], "retrieval_attempts": 0})
t = out["execution_trace"][-1]
check("trace says the store could not be asked", "VECTOR STORE ERRORS" in t["detail"] and t["status"] == "warn", str(t))
check("the failed call is a FAIL, not ok", "vector_search:FAIL" in t["detail"], str(t))

# tools that need a checkout are not planned when repo_path is a URL
URL = "https://github.com/x/y.git#dev"
calls = [ToolCall(tool_name="vector_search", args={"query": "q"}),
         ToolCall(tool_name="grep", args={"repo_path": URL, "pattern": "p"}),
         ToolCall(tool_name="cat", args={"repo_path": URL, "file_path": "a.py"}),
         ToolCall(tool_name="ast_parse", args={"repo_path": URL, "file_path": "a.py"})]
kept, dropped = ag._drop_unrunnable_tools(calls, "q")
check("URL repo_path: path tools dropped, vector_search kept", [c.tool_name for c in kept] == ["vector_search"] and dropped == ["grep", "cat", "ast_parse"], f"{kept} {dropped}")
with tempfile.TemporaryDirectory() as d:
    kept, dropped = ag._drop_unrunnable_tools([ToolCall(tool_name="grep", args={"repo_path": d, "pattern": "p"})], "q")
    check("a real directory keeps its path tools", [c.tool_name for c in kept] == ["grep"] and dropped == [])
kept, dropped = ag._drop_unrunnable_tools([ToolCall(tool_name="grep", args={"repo_path": URL, "pattern": "p"})], "my question")
check("nothing runnable left -> a plain vector_search on the question",
      len(kept) == 1 and kept[0].tool_name == "vector_search" and kept[0].args["query"] == "my question", str(kept))
ag.DROP_UNRUNNABLE_TOOLS = False
kept, dropped = ag._drop_unrunnable_tools([ToolCall(tool_name="grep", args={"repo_path": URL, "pattern": "p"})], "q")
check("DROP_UNRUNNABLE_TOOLS=off keeps the plan as the planner made it", [c.tool_name for c in kept] == ["grep"] and dropped == [])
ag.DROP_UNRUNNABLE_TOOLS = True

# a planner that proposes the same call twice (heavy-007 Q4) must not double the evidence
a1 = ToolCall(tool_name="vector_search", args={"query": "q", "chroma_host": "h", "collections": [A]})
a2 = ToolCall(tool_name="vector_search", args={"collections": [A], "chroma_host": "h", "query": "q"})   # same, keys reordered
b1 = ToolCall(tool_name="vector_search", args={"query": "other", "chroma_host": "h", "collections": [A]})
kept, n = ag._dedupe_tool_calls([a1, a2, b1])
check("identical calls collapse to the first, a different query stays", kept == [a1, b1] and n == 1, f"{kept} {n}")

store({A: "ok", B: "ok"})
out = ag.node_tool_execution({"query": "q", "active_collections": [A], "approved_tool_calls": [a1, a2],
                              "execution_trace": [], "retrieval_attempts": 0})
check("a repeated call is not run twice", len(out["executed_tool_calls"]) == 1 and len(out["retrieved_chunks"]) == 2,
      f"{len(out['executed_tool_calls'])} calls, {len(out['retrieved_chunks'])} chunks")
check("... and the trace says so", "1 repeated tool call(s) skipped" in out["execution_trace"][-1]["detail"], str(out["execution_trace"][-1]))

out2 = ag.node_tool_execution({"query": "q", "active_collections": [A], "approved_tool_calls": [
    ToolCall(tool_name="vector_search", args={"query": "x", "chroma_host": "h", "collections": [A]}),
    ToolCall(tool_name="vector_search", args={"query": "y", "chroma_host": "h", "collections": [A]})],
    "execution_trace": [], "retrieval_attempts": 0})
check("two different queries returning the same passages count them once",
      len(out2["executed_tool_calls"]) == 2 and len(out2["retrieved_chunks"]) == 2 and "2 duplicate chunk(s) removed" in out2["execution_trace"][-1]["detail"],
      f"{len(out2['retrieved_chunks'])} chunks; {out2['execution_trace'][-1]['detail']}")

# the sampler records paging pressure
import host_sampler
p = host_sampler.sample_pressure()
if os.path.exists("/proc/vmstat"):
    check("sampler records swap and paging counters", "host_swap_used_mb" in p and "host_pgmajfault" in p, str(p))
else:
    print("SKIP sampler paging counters (no /proc/vmstat)")

print("\nALL OK" if not fails else f"\n{len(fails)} FAILED: {fails}")
sys.exit(1 if fails else 0)
