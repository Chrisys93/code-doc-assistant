"""
test_eval.py -- checks the evaluation tooling end to end with a fake LLM, fake retrieval and a throw-away MLflow
store (no GPU, no Ollama, no Chroma, no network). Takes a few seconds.

    python eval/test_eval.py          (or: docker exec -w /app/eval code-doc-app python test_eval.py)
"""
import csv
import glob
import json
import os
import sys
import tempfile
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.environ.get("EVAL_SRC") or os.path.join(HERE, "..", "src"))
os.environ["EVAL_SRC"] = SRC
sys.path.insert(0, HERE)
sys.path.insert(0, SRC)
os.chdir(SRC)
os.environ.setdefault("INFERENCE_BACKEND", "llamacpp")

TMP = tempfile.mkdtemp(prefix="eval-test-")
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{TMP}/mlflow.db"
os.environ["MLFLOW_ARTIFACT_ROOT"] = TMP

# --- fakes ---------------------------------------------------------------------------------
INDEXED: list = []
def fake_ensure(repos, chroma_host, force=False):
    INDEXED.append(list(repos))
    out = {}
    for r in repos:
        x = {"status": "ingested", "docs": 25 if "chapter3" in r else 10}
        if "chapter3" in r:
            x["warning"] = "not indexed (unsupported file types): .zig x3 - answers will not be grounded"
        out[r] = x
    return out
ri = types.ModuleType("repo_index")
ri.ensure_indexed = fake_ensure
ri.active_collections = lambda repos: [f"col_{r.rstrip('/').split('/')[-1].split('.')[0].lower()}" for r in repos]
sys.modules["repo_index"] = ri
for _mod, _cls in (("langchain_ollama", "ChatOllama"), ("langchain_openai", "ChatOpenAI")):
    try:
        __import__(_mod)
    except ImportError:
        _m = types.ModuleType(_mod); setattr(_m, _cls, object); sys.modules[_mod] = _m

import agent_graph as ag  # noqa: E402

ANSWERS = {
    "How are the hashes": "No relation. Hash-routing in Icarus maps names to nodes; embeddings are vectors. See `src/repo_index.py`.",
    "What are the files": "ukhls and iapt scripts produce ind_ghq_lfs and ghqevent. See `docs/script-index.md`.",
}
class FakeLLM:
    model = "fake-model"
    def invoke(self, messages):
        system = messages[0].content
        user = messages[-1].content
        if "tool-selection agent" in system:
            out = json.dumps([{"tool_name": "vector_search", "args": {"query": "q", "chroma_host": "x", "filter_file": "IcarusSEND"}, "reasoning": "r"}])
        elif "evaluating a code documentation response" in system:
            out = json.dumps({"total": 9, "pass": True, "reason": "ok"})
        else:
            out = next((v for k, v in ANSWERS.items() if k in user), "Generic answer about `src/config.py` and config, global settings, hidp.")
        return types.SimpleNamespace(content=out)
ag._get_llm = lambda **kw: FakeLLM()
SEARCHED: list = []
def fake_run_tool(name, args):
    SEARCHED.append(list(args.get("collections") or []))
    return {"success": True, "count": 2, "collections_searched": args.get("collections"),
            "chunks": [{"content": f"c{i}", "source_file": f, "start_line": 1, "end_line": 5, "chunk_type": "code",
                        "confidence": 0.8} for i, f in enumerate(["src/repo_index.py", "src/config.py"])]}
ag.run_tool = fake_run_tool

ag._llamacpp_served_model = lambda: "deepseek-coder-v2-lite-instruct-q4_k_m"     # what a real heavy server reports
import run_eval  # noqa: E402
import checks  # noqa: E402

# --- 1. checks -----------------------------------------------------------------------------
print("1. automatic checks")
q = {"cite_all": ["src/config.py"], "cite_any": ["x.py", "src/ingest.py"], "must_mention": [["routing"], ["vector"]], "must_not": ["definitely related"]}
r = checks.evaluate(q, "Hash routing, not vectors.", ["src/config.py", "src/ingest.py"])
assert r["cite_all_ok"] and r["cite_any_ok"] and r["mention_ok"] and r["must_not_ok"] and r["auto_pass"], r
r = checks.evaluate(q, "They are definitely related via routing and vector.", ["src/config.py"])
assert not r["cite_any_ok"] and not r["must_not_ok"] and not r["auto_pass"], r
assert checks.cites(["iapt/a b(1).do"], "iapt/a b(1).do") and checks.cites(["repo/src/config.py"], "src/config.py")
truth = ["iapt/0.clean data.do", "ukhls/U1.1_indresp_ghq_lfs.do", "ukhls/U1.7_master_IV_selection_tests.do"]
r = checks.evaluate({"ground_truth_files": truth}, "paths in `0.clean data.do` and ukhls/U1.1_indresp_ghq_lfs.do", [])
assert r["files_named_n"] == 2 and abs(r["file_recall"] - 2 / 3) < 1e-9, r
assert checks.evaluate({"grading": "ungraded"}, "x", [])["auto_pass"] is None

# --- 2. a full run -------------------------------------------------------------------------
print("2. full run: both stages, repeat, nested MLflow runs")
OUT = os.path.join(TMP, "results")
rc = run_eval.main(["--config", "llamacpp-heavy", "--stage", "all", "--out", OUT, "--group", "g1",
                    "--repeat", "2", "--configs-file", os.path.join(HERE, "configs.json"),
                    "--questions", os.path.join(HERE, "questions.json")])
assert rc == 0
d = os.path.join(OUT, "g1", "llamacpp-heavy")
files = {os.path.basename(p) for p in glob.glob(d + "/*.json")}
want = {f"Q{i}_r{k}.json" for i in range(1, 11) for k in (0, 1)} | {f"{p}_r{k}.json" for p in ("Q1_rerun", "Q4_rerun", "Q10_rerun") for k in (0, 1)} | {"summary.json", "WARMUP.json"}
assert files == want, sorted(want ^ files)
rec = json.load(open(d + "/Q1_r0.json"))
assert rec["outcome"] == "answered" and rec["sources"] and rec["node_ms"].get("generation") is not None, rec["node_ms"]
assert rec["checks"]["mention_ok"] is True
assert rec["mlflow_run_id"] and rec["t_end"] > rec["t_start"]
assert isinstance(rec["chunks"], list) and rec["chunks"] and {"rank", "file", "chars", "in_context"} <= set(rec["chunks"][0]), rec["chunks"][:1]
assert "context" in rec            # the text the model was shown is saved with every result
assert {"tool_selection", "tool_execution", "generation"} <= set(rec["node_ms"]), rec["node_ms"]
print("   node latencies:", rec["node_ms"])
q6 = json.load(open(d + "/Q6_r0.json"))
assert q6["collections"] == ["col_code-doc-assistant"], q6["collections"]      # scoped to one repo
q7 = json.load(open(d + "/Q7_r0.json"))
assert len(q7["collections"]) == 3 and q7["stage"] == "B"
assert json.load(open(d + "/Q4_rerun_r0.json"))["rerun_of"] == "Q4"
assert len(INDEXED) == 2 and len(INDEXED[0]) == 2 and len(INDEXED[1]) == 3, INDEXED
summary = json.load(open(d + "/summary.json"))
assert summary["stages"]["B"]["status"] and summary["questions_run"] == 26 and summary["errors"] == 0, summary["questions_run"]
rows = list(csv.DictReader(open(d + "/summary.csv")))
assert len(rows) == 27 and rows[0]["id"] == "WARMUP" and rows[1]["id"] == "Q1", [r["id"] for r in rows][:3]
cold = json.load(open(d + "/WARMUP.json"))
assert cold["kind"] == "warmup" and cold["wall_ms"] > 0 and cold["question"] == run_eval.WARMUP_QUESTION
assert summary["cold_start"]["wall_ms"] == cold["wall_ms"] and summary["questions_run"] == 26   # not in the statistics

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
cl = MlflowClient()
exp = cl.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT", "code-doc-assistant-dev"))
runs = cl.search_runs([exp.experiment_id], max_results=200)
parents = [r for r in runs if r.data.tags.get("run.kind") == "eval_parent"]
kids = [r for r in runs if r.data.tags.get("run.kind") == "eval_question"]
warm = [k for k in kids if k.data.tags.get("eval.warmup") == "true"]
kids = [k for k in kids if k.data.tags.get("eval.warmup") != "true"]
assert len(parents) == 1 and len(kids) == 26 and len(warm) == 1, (len(parents), len(kids), len(warm))
stray = [r for r in runs if not r.data.tags.get("run.kind")]
assert not stray, f"stray MLflow runs: {[(r.info.run_id, dict(r.data.metrics)) for r in stray]}"
assert "generation_latency_ms" in kids[0].data.metrics, "node-level metrics must land in the question's own run"
assert all(k.data.tags.get("mlflow.parentRunId") == parents[0].info.run_id for k in kids)
k0 = [k for k in kids if k.data.tags["eval.question"] == "Q1"][0]
assert "lat_generation_ms" in k0.data.metrics and "auto_pass" in k0.data.metrics and "wall_ms" in k0.data.metrics
assert k0.data.tags["eval.config"] == "llamacpp-heavy" and k0.data.tags["eval.t_start"]
assert parents[0].data.metrics["questions_run"] == 26 and "stage_B_index_ms" in parents[0].data.metrics
assert parents[0].data.metrics["cold_start_wall_ms"] > 0 and warm[0].data.tags["eval.question"] == "WARMUP"
assert parents[0].data.params["eval.config"] == "llamacpp-heavy"
print("   26 child runs nested under 1 parent; warm-up tagged and excluded; no stray runs")

print("3. tier guard, Ollama residency, --only / dry-run / unknown config")
ag._llamacpp_served_model = lambda: "mistral-nemo-instruct-2407-q4_k_m"          # server runs 'full', config wants 'heavy'
try:
    run_eval.main(["--config", "llamacpp-heavy", "--stage", "A", "--only", "Q1", "--out", OUT, "--group", "g2"]); raise AssertionError("should refuse")
except SystemExit as e:
    assert "serves" in str(e) and "modelTier=heavy" in str(e), e
assert run_eval.main(["--config", "llamacpp-heavy", "--stage", "A", "--only", "Q1", "--out", OUT, "--group", "g2", "--allow-tier-mismatch", "--no-warmup"]) == 0
import urllib.request as _ur
class _Resp:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def read(self, *a): return json.dumps({"models": [{"name": "mistral-nemo", "size": 8 * 1048576 * 1024, "size_vram": 0}]}).encode()
_orig_urlopen = _ur.urlopen
_ur.urlopen = lambda *a, **k: _Resp()
try:
    assert run_eval.main(["--config", "ollama-full", "--stage", "A", "--only", "Q1", "--out", OUT, "--group", "g3"]) == 0
finally:
    _ur.urlopen = _orig_urlopen
cold3 = json.load(open(os.path.join(OUT, "g3", "ollama-full", "WARMUP.json")))
assert cold3["residency"]["mistral-nemo"]["vram_fraction"] == 0.0, cold3["residency"]       # CPU-only is visible
ag._llamacpp_served_model = lambda: "deepseek-coder-v2-lite-instruct-q4_k_m"
assert run_eval.main(["--config", "llamacpp-heavy", "--dry-run", "--questions", os.path.join(HERE, "questions.json")]) == 0
sel = run_eval.select_questions(json.load(open(os.path.join(HERE, "questions.json"))), "B", "Q4_rerun,Q7")
assert [x["id"] for x in sel] == ["Q7", "Q4_rerun"], [x["id"] for x in sel]
try:
    run_eval.main(["--config", "nope"]); raise AssertionError("should exit")
except SystemExit as e:
    assert "unknown config" in str(e)

# --- 4. resources join ---------------------------------------------------------------------
print("4. host samples joined to question windows")
import join_resources  # noqa: E402
recs = [json.load(open(p)) for p in sorted(glob.glob(d + "/Q*_r0.json"))]
t0 = min(r["t_start"] for r in recs); t1 = max(r["t_end"] for r in recs)
sp = os.path.join(TMP, "samples.csv")
with open(sp, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["epoch", "gpu_mem_used_mb", "gpu_util_pct", "host_mem_used_mb"])
    t = t0 - 10
    while t < t1 + 1:
        busy = t >= t0
        w.writerow([round(t, 3), 6000 + (3000 if busy else 0), 80 if busy else 2, 20000])
        t += 0.05
assert join_resources.main(["--samples", sp, "--results", d, "--mlflow"]) == 0
rec = json.load(open(d + "/Q1_r0.json"))
assert rec["resources"]["gpu_mem_used_mb_peak"] == 9000 and rec["resources"]["gpu_mem_used_mb_peak_over_idle"] == 3000, rec["resources"]
assert os.path.exists(d + "/resources.csv")
k0 = cl.get_run(rec["mlflow_run_id"])
assert k0.data.metrics["res_gpu_mem_used_mb_peak"] == 9000

# --- 5. judge (fake Anthropic client) ------------------------------------------------------
print("5. judge is blind and records its model/prompt version")
SENT = []
class _Msgs:
    def create(self, **kw):
        SENT.append(kw)
        return types.SimpleNamespace(content=[types.SimpleNamespace(text='{"score": 2, "grounding": 1, "premise_corrected": true, "hallucinated": [], "justification": "ok"}')])
fake_anthropic = types.ModuleType("anthropic")
fake_anthropic.Anthropic = lambda: types.SimpleNamespace(messages=_Msgs())
sys.modules["anthropic"] = fake_anthropic
os.environ["ANTHROPIC_API_KEY"] = "test"
import judge  # noqa: E402
json.dump({**json.load(open(d + "/Q2_r0.json")), "manual_score": 2}, open(d + "/Q2_r0.json", "w"))
assert judge.judge_dir(d, calibrate=True) == 0
assert len(SENT) == 26
blob = json.dumps(SENT[0])
for secret in ("llamacpp-heavy", "llamacpp", "fake-model", "wall_ms"):
    assert secret not in blob, f"judge prompt leaks {secret}"
rec = json.load(open(d + "/Q2_r0.json"))
assert rec["judge"]["score"] == 2 and rec["judge"]["prompt_version"] == judge.PROMPT_VERSION and rec["judge"]["judge_model"]
assert cl.get_run(rec["mlflow_run_id"]).data.metrics["judge_score"] == 2

# --- 6. bench against a fake OpenAI-compatible server ---------------------------------------
print("6. serving benchmark: TTFT / prefill / decode from a streaming endpoint")
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers["Content-Length"]); body = json.loads(self.rfile.read(n))
        assert body["stream"] and self.path.endswith("/chat/completions")
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        import time as _t
        _t.sleep(0.05)
        for i in range(8):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": "tok "}}]}).encode() + b"\n\n"); self.wfile.flush(); _t.sleep(0.005)
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 8},
                         "timings": {"prompt_per_second": 1234.5, "predicted_per_second": 56.7}}).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
import bench  # noqa: E402
rc = bench.main(["--base-url", f"http://127.0.0.1:{srv.server_port}/v1", "--label", "fake", "--prompt-tokens", "64,256",
                 "--runs", "2", "--out", os.path.join(TMP, "bench")])
assert rc == 0
res = json.load(open(glob.glob(os.path.join(TMP, "bench", "*.json"))[0]))
assert len(res["rows"]) == 2 and res["rows"][0]["decode_tps_median"] == 56.7 and res["rows"][0]["prefill_tps_median"] == 1234.5
assert res["rows"][0]["ttft_s_median"] >= 0.05
srv.shutdown()
runs = cl.search_runs([exp.experiment_id], filter_string="tags.run.kind = 'bench'")
assert len(runs) == 1 and "p64_decode_tps_median" in runs[0].data.metrics

print("ALL EVAL CHECKS PASSED")
