"""
test_app_flow.py -- runs the REAL Streamlit app and the REAL LangGraph graph headless, with only
the LLM, Chroma retrieval and repo indexing faked (MLflow is the real library on a temp store). Checks the parts that used to break
silently in the UI: live per-node repainting of the Pipeline tab, both human-review pauses
(approve / modify / reject / regenerate), supervisor mode, and that what a reviewer sees is what runs.

    docker exec -w /app/src code-doc-app python test_app_flow.py        # exit code 0 = all passed

Needs no GPU, no Ollama, no Chroma. Takes a few seconds.
"""
import itertools
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("INFERENCE_BACKEND", "llamacpp")
os.chdir(HERE)

# --- fakes -----------------------------------------------------------------------------------
# MLflow: use the REAL library against a throw-away local store when it is installed (the app container
# has it), so the run bookkeeping is genuinely exercised; otherwise stub it and skip those checks.
import tempfile
_TMP = tempfile.mkdtemp(prefix="app-flow-")
try:
    import mlflow  # noqa: F401
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP}/mlflow.db"
    os.environ["MLFLOW_ARTIFACT_ROOT"] = _TMP
    REAL_MLFLOW = True
except ImportError:
    mlf = types.ModuleType("mlflow")
    class _Run:
        def __enter__(self): self.info = types.SimpleNamespace(run_id="run-1"); return self
        def __exit__(self, *a): return False
    mlf.start_run = lambda *a, **k: _Run()
    for _n in ("log_param", "log_metric", "log_text", "set_experiment", "set_tracking_uri"):
        setattr(mlf, _n, lambda *a, **k: None)
    sys.modules["mlflow"] = mlf
    REAL_MLFLOW = False

ri = types.ModuleType("repo_index")
ri.ensure_indexed = lambda repos, chroma_host, force=False: {r: {"status": "already_indexed", "docs": 10} for r in repos}
ri.active_collections = lambda repos: [f"repo_{i}" for i, _ in enumerate(repos)]
sys.modules["repo_index"] = ri

# Optional LLM client libs: in the app container they exist and are never called (the LLM is faked below);
# on a bare machine, stub them so the graph still imports.
for _mod, _cls in (("langchain_ollama", "ChatOllama"), ("langchain_openai", "ChatOpenAI")):
    try:
        __import__(_mod)
    except ImportError:
        _m = types.ModuleType(_mod); setattr(_m, _cls, object); sys.modules[_mod] = _m

import agent_graph as ag  # the real graph  (after mlflow is faked / configured)


class FakeLLM:
    model = "fake-model"
    calls = 0
    planner_prompts: list = []
    def invoke(self, messages):
        FakeLLM.calls += 1
        system = messages[0].content
        if "tool-selection agent" in system:       # planner: deliberately makes the usual mistakes
            FakeLLM.planner_prompts.append(messages[1].content)
            n = len(FakeLLM.planner_prompts)
            out = json.dumps([{"tool_name": "vector_search",
                               "args": {"query": f"q-plan-{n}", "chroma_host": "http://localhost:8000",
                                        "filter_file": "bogus"},
                               "reasoning": "conceptual"}])
        elif "evaluating a code documentation response" in system:
            out = json.dumps({"total": 9, "pass": True, "reason": "ok"})
        else:
            out = f"Answer #{FakeLLM.calls}: see `src/a.py` and `src/b.py`."
        return types.SimpleNamespace(content=out)

ag._get_llm = lambda **kw: FakeLLM()

RAN: list = []
ALLOW_FILTER = [False]
def fake_run_tool(name, args):
    assert name == "vector_search", name
    RAN.append(dict(args))
    if not ALLOW_FILTER[0]:
        assert "filter_file" not in args, "the planner's filter_file must not reach the tool"
    return {"success": True, "count": 3, "collections_searched": args.get("collections"),
            "chunks": [{"content": f"chunk{i}", "source_file": f"src/{'ab'[i % 2]}.py", "start_line": 1,
                        "end_line": 9, "chunk_type": "code", "confidence": 0.7, "repo": "repo_0"} for i in range(3)]}
ag.run_tool = fake_run_tool

# --- harness ---------------------------------------------------------------------------------
import streamlit as st
from streamlit.testing.v1 import AppTest

IMG: list = []
_orig_image = st.image
def _spy(*a, **k):
    IMG.append((len(st.session_state.last_trace), st.session_state.running_node))
    return _orig_image(*a, **k)
st.image = _spy

_ids = itertools.count(1)
def fresh(hitl, review):
    at = AppTest.from_file("app.py", default_timeout=120)
    at.session_state["thread_id"] = f"t-{next(_ids)}"
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    [t for t in at.toggle if t.label == "Tool plan HITL"][0].set_value(hitl)
    [s for s in at.selectbox if s.label == "Output review mode"][0].set_value(review)
    at.run()
    return at

def ask(at, q="Could A integrate B?"):
    IMG.clear()
    at.chat_input[0].set_value(q).run()
    assert not at.exception, [e.value for e in at.exception]

def press(at, key):
    [b for b in at.button if b.key == key][0].click().run()
    assert not at.exception, [e.value for e in at.exception]

def last(at): return at.session_state["messages"][-1]["content"]

# --- scenarios -------------------------------------------------------------------------------
print("1. Pipeline tab repaints once per node, with the running node marked")
at = fresh(False, "off"); ask(at)
seq = list(IMG)
print("   draws (trace entries, running node):", seq)
assert len(seq) >= 7 and [n for n, _ in seq] == sorted(n for n, _ in seq) and seq[-1][0] == 6, seq
assert {"tool_selection", "tool_execution", "generation"} <= {r for _, r in seq}, seq
assert "Answer" in last(at) and "Sources" in last(at)

print("2. HITL: pause for the tool plan, then for the output; reviewer sees the NORMALISED plan")
at = fresh(True, "human"); ask(at)
assert at.session_state["awaiting_hitl"] and not at.session_state["streaming_active"]
plan = at.session_state["pending_hitl"][0]["args"]
assert "filter_file" not in plan and plan["collections"] == ["repo_0"] and plan["chroma_host"] == ag.CHROMA_HOST, plan
assert "ignored planner args: filter_file='bogus'" in at.session_state["last_trace"][0]["detail"]
assert len(at.chat_input) == 0, "no new-question box while a review is pending"
press(at, "h1_approve")
assert at.session_state["awaiting_output_review"]
press(at, "h2_submit_1")
assert "Answer" in last(at) and not at.session_state["awaiting_output_review"]

print("3. Rejecting the tool plan asks what to do next; 'End here' ends cleanly with a clear message")
at = fresh(True, "human"); ask(at)
press(at, "h1_reject")
assert at.session_state["awaiting_hitl"], "rejecting must not end the run by itself"
assert [b for b in at.button if b.key == "h1_replan_fb"][0].disabled, "re-plan-with-feedback needs feedback text"
assert {"h1_replan_fb", "h1_replan", "h1_end"} <= {b.key for b in at.button}
press(at, "h1_end")
assert "rejected" in last(at).lower() and not at.session_state["awaiting_hitl"] and not at.session_state["streaming_active"]

print("3b. Reject -> re-plan WITH feedback: the planner is told what was rejected and why; new plan; approve")
FakeLLM.planner_prompts.clear()
at = fresh(True, "human"); ask(at)
first_plan = at.session_state["pending_hitl"][0]["args"]["query"]
press(at, "h1_reject")
at.text_area(key="h1_expect").set_value("search the helm chart, not the README").run()
assert not [b for b in at.button if b.key == "h1_replan_fb"][0].disabled
press(at, "h1_replan_fb")
assert at.session_state["awaiting_hitl"], "after a re-plan the reviewer must see the NEW plan"
new_plan = at.session_state["pending_hitl"][0]["args"]
assert new_plan["query"] != first_plan and "filter_file" not in new_plan, (first_plan, new_plan)
p2 = FakeLLM.planner_prompts[-1]
assert "REJECTED" in p2 and "search the helm chart, not the README" in p2 and first_plan in p2, p2
assert "re-plan 1" in [h for h in at.subheader][0].value
press(at, "h1_approve_r1")
assert at.session_state["awaiting_output_review"]
press(at, "h2_submit_1")
assert "Answer" in last(at)
assert any("human decision: replan" in t.get("detail", "") for t in at.session_state["last_trace"])
REPLAN_THREAD = at.session_state["thread_id"]

print("3c. Reject -> re-plan WITHOUT feedback still sends the plan back; the re-plan limit ends the loop")
ag.MAX_REPLANS = 1
at = fresh(True, "off"); ask(at)
press(at, "h1_reject"); press(at, "h1_replan")
assert at.session_state["awaiting_hitl"] and at.session_state["replan_round"] == 1
assert "No reason given" in FakeLLM.planner_prompts[-1]
press(at, "h1_reject_r1"); press(at, "h1_replan_r1")        # second re-plan exceeds the limit of 1
assert not at.session_state["awaiting_hitl"] and "rejected" in last(at).lower()
assert any("re-plan limit" in t.get("detail", "") for t in at.session_state["last_trace"])
ag.MAX_REPLANS = 3

print("4. HITL: 'Use modified', then Regenerate once, then accept")
at = fresh(True, "human"); ask(at); press(at, "h1_modify")
first = at.session_state["pending_output_review"]["response"]
at.radio(key="h2_dec_1").set_value("🔄 Regenerate").run()
press(at, "h2_submit_1")
second = at.session_state["pending_output_review"]
assert second["generation_attempts"] == 2 and second["response"] != first
press(at, "h2_submit_2")
assert "Answer" in last(at)

print("5. A human edit of collections / filter_file is honoured, not overridden")
at = fresh(True, "off"); ask(at)
edited = dict(at.session_state["pending_hitl"][0]["args"]); edited.update(filter_file="src/a.py", collections=["repo_9"])
ALLOW_FILTER[0] = True
at.text_area(key="h1_0").set_value(json.dumps(edited)).run()
press(at, "h1_modify")
ALLOW_FILTER[0] = False
assert RAN[-1]["filter_file"] == "src/a.py" and RAN[-1]["collections"] == ["repo_9"] and RAN[-1]["chroma_host"] == ag.CHROMA_HOST, RAN[-1]

print("6. Supervisor review mode runs straight through; the heavy tier is selectable; input re-enabled")
at = fresh(False, "supervisor"); ask(at)
assert not at.session_state["awaiting_hitl"] and not at.session_state["awaiting_output_review"] and "Answer" in last(at)
assert "heavy" in [s for s in at.selectbox if s.label == "Model tier"][0].options
assert len(at.chat_input) == 1 and not at.chat_input[0].disabled

print("7. MLflow: exactly ONE run per question, params/metrics land in it, outcome + status are right")
if not REAL_MLFLOW:
    print("   (mlflow not installed here - skipped)")
else:
    from mlflow import MlflowClient
    import tracking
    client = MlflowClient()
    exp_id = tracking._setup()

    def runs_for(thread):
        return client.search_runs([exp_id], filter_string=f"tags.thread_id = '{thread}'")

    # a question that was re-planned once and then answered
    rs = runs_for(REPLAN_THREAD)
    assert len(rs) == 1, f"expected 1 run for the whole question (re-plan included), got {len(rs)}"
    r = rs[0]
    assert r.info.status == "FINISHED" and r.data.tags["outcome"] == "answered", (r.info.status, r.data.tags)
    assert r.data.params["query"] == "Could A integrate B?" and r.data.params["rt.backend"] == ag._default_backend(), r.data.params
    assert r.data.params["output_review_mode"] == "human" and "repo_0" in r.data.params["collections"]
    assert r.data.metrics["tool_plan_replans"] == 1.0 and r.data.metrics["chunks_retrieved"] == 3.0, r.data.metrics
    assert r.data.metrics["unique_source_files"] == 2.0 and r.data.metrics["answered"] == 1.0
    assert r.data.metrics["planner_json_ok"] == 1.0 and r.data.metrics["planner_ignored_args"] >= 1.0, r.data.metrics
    assert "generation_latency_ms" in r.data.metrics, "node-level metrics must land in the question's run"
    assert r.data.tags["hitl.tool_plan_decision"] in ("approved", "modified")
    arts = [a.path for a in client.list_artifacts(r.info.run_id)]
    assert "result.json" in arts, arts
    res = json.load(open(client.download_artifacts(r.info.run_id, "result.json")))
    assert res["outcome"] == "answered" and res["sources"] and res["chunks"], res.keys()

    # a paused run stays RUNNING (not failed) while the human reviews, then 'End here' -> rejected
    at = fresh(True, "human"); ask(at)
    (paused,) = runs_for(at.session_state["thread_id"])
    assert paused.info.status == "RUNNING", f"run must stay RUNNING during a review pause, got {paused.info.status}"
    press(at, "h1_reject"); press(at, "h1_end")
    (ended,) = runs_for(at.session_state["thread_id"])
    assert ended.info.status == "FINISHED" and ended.data.tags["outcome"] == "rejected", (ended.info.status, ended.data.tags)

    # 'Clear conversation' mid-review abandons the run instead of leaving it RUNNING forever
    at = fresh(True, "human"); ask(at); thread = at.session_state["thread_id"]
    [b for b in at.button if "Clear" in b.label][0].click().run()
    (ab,) = runs_for(thread)
    assert ab.data.tags["outcome"] == "abandoned" and ab.info.status == "FINISHED", (ab.info.status, ab.data.tags)

    # no stray runs: nothing was logged outside the per-question runs
    stray = [r for r in client.search_runs([exp_id]) if "thread_id" not in r.data.tags]
    assert not stray, f"{len(stray)} stray run(s) outside a question"
    default = client.search_runs(["0"])
    assert not default, f"{len(default)} run(s) leaked into the Default experiment"

print("8. Indexing failure is a HARD STOP: clear error, no pipeline run, persistent notes, run closed as error")
_orig_ei = ri.ensure_indexed
ri.ensure_indexed = lambda repos, chroma_host, force=False: {r: {"status": "error", "docs": 0,
    "error": "embedding model 'nomic-embed-text' is not available in Ollama", "hint": "ollama pull nomic-embed-text"} for r in repos}
at = fresh(False, "off"); calls_before = FakeLLM.calls; ask(at)
msgs = at.session_state["messages"]
assert FakeLLM.calls == calls_before, "the LLM must not be called after an indexing failure"
assert any(m.get("kind") == "error" and "nomic-embed-text" in m["content"] and "ollama pull" in m["content"] for m in msgs), msgs
assert any(m.get("kind") == "index" and "error" in m["content"] for m in msgs), "indexing outcome must persist"
assert not at.session_state["streaming_active"] and at.session_state["current_run_id"] is None
assert [e for e in at.error], "an st.error must be on screen"
if REAL_MLFLOW:
    (er,) = runs_for(at.session_state["thread_id"])
    assert er.data.tags["outcome"] == "error", er.data.tags
ri.ensure_indexed = _orig_ei

print("9. Empty retrieval: generation is skipped, the answer says why, warn in trace, no sources, no review")
_orig_rt = ag.run_tool
ag.run_tool = lambda name, args: {"success": True, "count": 0, "chunks": [], "collections_searched": args.get("collections")}
at = fresh(False, "off"); calls_before = FakeLLM.calls; ask(at)
txt = last(at)
assert "nothing was retrieved" in txt.lower() and "Sources:" not in txt, txt
tr = at.session_state["messages"][-1]["trace"]
st_by = {t["node"]: t["status"] for t in tr}
assert st_by["tool_execution"] == "warn" and st_by["context_assembly"] == "warn" and st_by["generation"] == "warn", tr
assert not any("Answer #" in m["content"] for m in at.session_state["messages"])
assert any("warning" in e.label for e in at.expander), [e.label for e in at.expander]
if REAL_MLFLOW:
    (er,) = runs_for(at.session_state["thread_id"])
    assert er.data.tags["outcome"] == "no_retrieval", er.data.tags
ag.run_tool = _orig_rt

print("10. Normal answer keeps a persistent per-answer trace and the indexing note after it finishes")
at = fresh(False, "off"); ask(at)
assert at.session_state["messages"][-1].get("trace"), "answer must carry its trace"
assert any(m.get("kind") == "index" for m in at.session_state["messages"])
assert any("Pipeline trace" in e.label for e in at.expander)
assert "context_assembly" in [t["node"] for t in at.session_state["messages"][-1]["trace"]]

print("11. Heavy truncation by the context budget is a visible warning, but the answer is still generated")
_orig_rt = ag.run_tool
ag.run_tool = lambda name, args: {"success": True, "count": 10, "collections_searched": args.get("collections"),
    "chunks": [{"content": f"{i}" * 12000, "source_file": f"src/f{i}.py", "start_line": 1, "end_line": 9,
                "chunk_type": "code", "confidence": 0.7, "repo": "repo_0"} for i in range(10)]}
at = fresh(False, "off"); ask(at)
tr = {t["node"]: t for t in at.session_state["messages"][-1]["trace"]}
assert tr["context_assembly"]["status"] == "warn" and "dropped for budget" in tr["context_assembly"]["detail"], tr["context_assembly"]
assert "did not fit" in tr["context_assembly"]["detail"]
assert tr["generation"]["status"] == "ok" and "Answer #" in last(at), "an answer must still be generated"
ag.run_tool = _orig_rt

print("ALL APP-FLOW CHECKS PASSED")
