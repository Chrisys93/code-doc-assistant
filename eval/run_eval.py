#!/usr/bin/env python3
"""
run_eval.py -- headless evaluation runner for the Code Documentation Assistant.

It drives the SAME LangGraph pipeline the UI uses (agent_graph.build_graph), question by question, with no
human pauses, and records everything to MLflow:

    parent run   one per invocation = one configuration evaluated (params, summary metrics, summary.json)
      child run  one per question   = the normal per-question run (tracking.start_query_run) plus latency
                                      per node, automatic check results and the time window (for resources)

Run it where the services are reachable -- normally inside the app container:

    kubectl exec -n code-doc deploy/<release>-code-doc-assistant-app -- \\
        python /app/eval/run_eval.py --config heavy-llamacpp --stage all

    --config NAME     entry of eval/configs.json (backend, tier, env overrides). ONE config per invocation:
                      most settings are read once at import, so a different config is a new process.
    --stage A|B|all   stage A = two original repos, stage B = + chapter3 (indexes it first and times it)
    --only Q1,Q4      run only these questions        --repeat N  ask each question N times (latency stats)
    --warmup          ask a fixed neutral question first (default on). It is the COLD START: logged on its own
                      (WARMUP.json, a row in summary.csv, cold_start_* metrics on the parent run, its own child run
                      tagged eval.warmup=true) and kept out of the latency statistics, so model load time is
                      reported, not hidden and not charged to Q1
    --judge           after the run, grade the answers with eval/judge.py (needs ANTHROPIC_API_KEY)
    --dry-run         print what would run; touch nothing

Outputs: eval/results/<group>/<config>/<question>.json, summary.json, summary.csv (+ the MLflow runs).
Resources (GPU/RAM) are not sampled here -- the pod cannot see the GPU. Run eval/host_sampler.py on the host
during the run and join afterwards with eval/join_resources.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_src() -> str:
    for cand in (os.environ.get("EVAL_SRC"), "/app/src", os.path.join(HERE, "..", "src")):
        if cand and os.path.isfile(os.path.join(cand, "agent_graph.py")):
            return os.path.abspath(cand)
    sys.exit("cannot find src/ (agent_graph.py); set EVAL_SRC")


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--configs-file", default=os.path.join(HERE, "configs.json"))
    ap.add_argument("--questions", default=os.path.join(HERE, "questions.json"))
    ap.add_argument("--stage", default="all", choices=["A", "B", "all"])
    ap.add_argument("--only", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--warmup", dest="warmup", action="store_true", default=True)
    ap.add_argument("--no-warmup", dest="warmup", action="store_false")
    ap.add_argument("--review-mode", default="off", choices=["off", "self", "supervisor"])
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--group", default=None, help="eval.group tag; default = timestamp")
    ap.add_argument("--allow-tier-mismatch", action="store_true",
                    help="run even if the llama-server's model does not belong to the config's tier")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


WARMUP_QUESTION = "Give a short overview of what this repository does."


def select_questions(qdoc: dict, stage: str, only: str) -> list[dict]:
    """Questions in run order: stage A, then stage B, then the stage-B dilution re-runs."""
    byid = {q["id"]: q for q in qdoc["questions"]}
    chosen: list[dict] = []
    for st in ("A", "B"):
        if stage not in ("all", st):
            continue
        chosen += [q for q in qdoc["questions"] if q["stage"] == st]
        if st == "B":
            for p in qdoc.get("dilution_probes", []):
                base = dict(byid[p["rerun_of"]])
                base.update(id=p["id"], stage="B", rerun_of=p["rerun_of"], probe_note=p.get("note"))
                chosen.append(base)
    if only:
        want = {x.strip() for x in only.split(",") if x.strip()}
        chosen = [q for q in chosen if q["id"] in want]
    return chosen


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main(argv=None) -> int:
    args = parse_args(argv)
    configs = _load(args.configs_file)["configs"]
    if args.config not in configs:
        sys.exit(f"unknown config {args.config!r}; known: {', '.join(configs)}")
    cfg = configs[args.config]
    qdoc = _load(args.questions)
    questions = select_questions(qdoc, args.stage, args.only)
    if not questions:
        sys.exit("no questions selected")

    if args.dry_run:
        print(f"config {args.config}: {json.dumps(cfg)}")
        for q in questions:
            print(f"  [{q['stage']}] {q['id']:<9} scope={q.get('scope', 'all active')} {q['question'][:80]}")
        return 0

    # Settings that agent_graph / config read at import time must be in place BEFORE it is imported.
    for k, v in (cfg.get("env") or {}).items():
        os.environ[str(k)] = str(v)
    group = args.group or time.strftime("%Y%m%d-%H%M%S")
    os.environ.setdefault("EVAL_GROUP", group)
    src = _find_src()
    sys.path.insert(0, HERE)
    sys.path.insert(0, src)
    os.chdir(src)

    import agent_graph as ag
    import repo_index
    import tracking
    from checks import evaluate

    backend, tier = cfg.get("backend"), cfg.get("tier")
    chroma = os.environ.get("CHROMA_HOST", "http://chromadb:8000")
    out_dir = os.path.join(args.out, group, args.config)
    os.makedirs(out_dir, exist_ok=True)

    runtime = ag.runtime_info(backend, tier)
    # llama-server serves ONLY the GGUF it was launched with; the tier in a config cannot change that. Evaluating
    # "full-llamacpp" against a server that is running DeepSeek would silently measure the wrong model.
    if backend == "llamacpp" and tier and not args.allow_tier_mismatch:
        import deployment
        served = deployment.tier_for_model(runtime.get("model"))
        if served and served != tier:
            sys.exit(f"config {args.config!r} wants tier {tier!r} but llama-server serves {runtime.get('model')!r} "
                     f"(tier {served!r}). Redeploy it: helm upgrade ... --set modelTier={tier}  "
                     f"(or pass --allow-tier-mismatch to evaluate what is running)")
    graph = ag.build_graph(checkpointer=ag.make_checkpointer(), output_review_mode=args.review_mode, hitl_enabled=False)
    ok, why = tracking.health()
    print(f"MLflow: {why}")
    if not ok:
        sys.exit("MLflow is not usable, and an evaluation without it records nothing (and the graph's nodes would "
                 "open stray runs). Fix MLFLOW_TRACKING_URI / the mlflow service first.")
    top_k = os.environ.get("TOP_K", "10")
    parent = tracking.start_group_run(
        f"eval {args.config} {group}",
        params={"eval.config": args.config, "eval.config_json": cfg, "eval.stages": args.stage,
                "eval.questions_version": qdoc.get("version"), "eval.review_mode": args.review_mode,
                "eval.repeat": args.repeat, "eval.warmup": args.warmup, "eval.note": args.note,
                "top_k": top_k, **{f"rt.{k}": v for k, v in runtime.items()}},
        tags={"eval.config": args.config, "eval.group": group})
    tracking.log_extra(parent, artifacts={"questions.json": qdoc})
    print(f"config {args.config} -> runtime {json.dumps(runtime, default=str)}")

    def residency() -> dict:
        """Where is the model actually running? Ollama reports how much of it is in VRAM (CPU-only = 0)."""
        if backend != "ollama":
            return {}
        try:
            import json as _j
            import urllib.request
            host = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
            with urllib.request.urlopen(host + "/api/ps", timeout=3) as r:
                models = _j.load(r).get("models") or []
            return {m.get("name"): {"size_mb": round((m.get("size") or 0) / 1048576),
                                    "vram_mb": round((m.get("size_vram") or 0) / 1048576),
                                    "vram_fraction": round((m.get("size_vram") or 0) / max(m.get("size") or 1, 1), 3)}
                    for m in models}
        except Exception:  # noqa: BLE001
            return {}

    repo_urls = qdoc["repos"]
    collections_by_repo: dict[str, str] = {}
    stage_info: dict[str, dict] = {}
    results: list[dict] = []
    warmup_rec: dict = {}
    settings_base = {
        "hitl_enabled": False, "output_review_mode": args.review_mode,
        "max_retrieval_attempts": ag.MAX_RETRIEVAL_ATTEMPTS, "max_generation_attempts": ag.MAX_GENERATION_ATTEMPTS,
        "confidence_threshold": ag.CONFIDENCE_THRESHOLD, "quality_gate": ag.QUALITY_GATE_THRESHOLD, "top_k": top_k,
    }

    def run_one(q: dict, stage_repos: list[str], rep: int, warmup: bool = False) -> dict:
        refs = [repo_urls[k] for k in (q.get("scope") or stage_repos)]
        cols = repo_index.active_collections(refs)
        init = {
            "query": q["question"], "repo_path": refs[0], "hitl_enabled": False,
            "output_review_mode": args.review_mode, "active_collections": cols,
            "proposed_tool_calls": [], "hitl_checkpoint": None, "approved_tool_calls": [], "executed_tool_calls": [],
            "retrieved_chunks": [], "confidence_scores": [], "retrieval_attempts": 0,
            "max_retrieval_attempts": ag.MAX_RETRIEVAL_ATTEMPTS, "supervisor_adjustments": [],
            "proceed_to_generation": False, "final_context": "", "response": "", "source_attribution": [],
            "post_generation_feedback": None, "generation_attempts": 0, "active_backend": backend,
            "active_model_tier": tier, "active_model": None, "execution_trace": [], "mlflow_run_id": None,
            "total_latency_ms": None,
        }
        # Always recorded, even the warm-up (tagged eval.warmup and left out of every statistic): the graph's nodes
        # call mlflow.log_* directly, and without an active run those calls would open stray runs.
        run_id = tracking.start_query_run(
            q["question"], runtime,
            settings={"repos": refs, "collections": cols, **settings_base},
            tags={"source": "eval", "run.kind": "eval_question", "mlflow.parentRunId": parent,
                  "eval.config": args.config, "eval.group": group, "eval.question": q["id"],
                  "eval.stage": q["stage"], "eval.repeat_index": rep, "eval.kind": q.get("kind"),
                  "eval.warmup": "true" if warmup else "false"})
        init["mlflow_run_id"] = run_id
        cfgd = {"configurable": {"thread_id": f"eval-{group}-{q['id']}-{rep}-{uuid.uuid4().hex[:6]}"}}
        node_ms: dict[str, float] = {}
        error = None
        t_start = time.time()
        t_prev = time.perf_counter()
        try:
            for step in graph.stream(init, config=cfgd, stream_mode="updates"):
                now = time.perf_counter()
                for node in step:
                    if node == "__interrupt__":
                        raise RuntimeError("the graph paused for a human; evaluation runs must not (review mode / HITL)")
                    node_ms[node] = node_ms.get(node, 0.0) + (now - t_prev) * 1000.0
                t_prev = now
        except Exception as e:  # noqa: BLE001 - one bad question must not end the evaluation
            error = f"{type(e).__name__}: {e}"
        t_end = time.time()
        state = dict(graph.get_state(cfgd).values or {})

        answer = state.get("response") or ""
        sources = list(state.get("source_attribution") or [])
        chunks = state.get("retrieved_chunks") or []
        files = sorted({getattr(c, "source_file", "") for c in chunks})
        chk = evaluate(q, answer, sources, files)
        trace = list(state.get("execution_trace") or [])
        warns = [t for t in trace if t.get("status") == "warn"]
        # What the model was actually shown, and which retrieved chunks made it in: without this a weak answer cannot be
        # told apart (chunk dropped for the budget / chunk present but ignored / retrieval missed it).
        ctx_text = state.get("final_context") or ""
        chunk_info = [{"rank": i + 1, "file": getattr(c, "source_file", ""),
                       "lines": f"{getattr(c, 'start_line', '')}-{getattr(c, 'end_line', '')}",
                       "chars": len(getattr(c, "content", "") or ""),
                       "confidence": getattr(c, "confidence", None),
                       "in_context": bool((getattr(c, "content", "") or "").strip()[:80]
                                          and (getattr(c, "content", "") or "").strip()[:80] in ctx_text)}
                      for i, c in enumerate(chunks)]
        outcome = ("error" if error else "no_retrieval" if state.get("retrieval_empty")
                   else "answered" if answer.strip() else "no_response")
        wall_ms = (t_end - t_start) * 1000.0
        gen_ms = float(state.get("total_latency_ms") or 0.0)
        rec = {
            "id": q["id"], "stage": q["stage"], "kind": q.get("kind"), "rerun_of": q.get("rerun_of"),
            "repeat_index": rep, "question": q["question"], "scope": q.get("scope"), "collections": cols,
            "config": args.config, "group": group, "outcome": outcome, "error": error,
            "answer": answer, "sources": sources, "retrieved_files": files, "chunks_n": len(chunks),
            "model": state.get("active_model"), "backend": state.get("active_backend") or backend,
            "t_start": t_start, "t_end": t_end, "wall_ms": wall_ms, "generation_ms": gen_ms,
            "node_ms": {k: round(v, 1) for k, v in node_ms.items()},
            "warnings": [{"node": w.get("node"), "detail": w.get("detail")} for w in warns],
            "trace": trace, "checks": chk, "mlflow_run_id": run_id,
            "context": ctx_text, "chunks": chunk_info,
            "expected": q.get("expected"), "manual_checks": q.get("manual_checks"), "rubric": q.get("rubric"),
            "manual_score": None, "judge": None,
        }
        metrics = {f"lat_{k}_ms": v for k, v in node_ms.items()}
        metrics.update({"gen_ms": gen_ms, "eval_wall_ms": wall_ms, "warnings_n": len(warns),
                        "auto_cite_all_ok": chk["cite_all_ok"], "auto_cite_any_ok": chk["cite_any_ok"],
                        "auto_mention_ok": chk["mention_ok"], "auto_must_not_ok": chk["must_not_ok"]})
        if chk["auto_pass"] is not None:
            metrics["auto_pass"] = chk["auto_pass"]
        if "file_recall" in chk:
            metrics["file_recall"] = chk["file_recall"]
        tracking.log_extra(run_id, metrics=metrics,
                           tags={"eval.t_start": f"{t_start:.3f}", "eval.t_end": f"{t_end:.3f}",
                                 "eval.outcome": outcome},
                           artifacts={"eval_checks.json": chk, "context_chunks.json": chunk_info,
                                      "context.json": {"final_context": ctx_text}})
        tracking.finish_run(run_id, state, error=error)
        rec["mlflow_url"] = tracking.run_url(run_id)
        return rec

    def save(rec: dict, name: str | None = None) -> None:
        name = name or (f"{rec['id']}" + (f"_r{rec['repeat_index']}" if args.repeat > 1 else ""))
        with open(os.path.join(out_dir, name + ".json"), "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False, default=str)

    err = None
    try:
        warmed = False
        for st in ("A", "B"):
            qs = [q for q in questions if q["stage"] == st]
            if not qs:
                continue
            stage_repos = qdoc["stages"][st]["repos"]
            refs = [repo_urls[k] for k in stage_repos]
            t0 = time.perf_counter()
            status = repo_index.ensure_indexed(refs, chroma)
            idx_ms = (time.perf_counter() - t0) * 1000.0
            bad = {r: x for r, x in status.items() if x.get("status") == "error" or not x.get("docs")}
            stage_info[st] = {"repos": refs, "index_ms": round(idx_ms, 1), "status": status}
            print(f"stage {st}: indexed/checked {len(refs)} repo(s) in {idx_ms/1000:.1f}s -> "
                  + ", ".join(f"{r.split('/')[-1]}={x.get('status')}({x.get('docs')})" for r, x in status.items()))
            for r, x in status.items():
                if x.get("warning"):
                    print(f"  WARNING {r.split('/')[-1]}: {x['warning']}")
            tracking.log_extra(parent, metrics={f"stage_{st}_index_ms": idx_ms,
                                                f"stage_{st}_docs": sum(x.get("docs") or 0 for x in status.values())},
                               artifacts={f"stage_{st}_indexing.json": status})
            if bad:
                print(f"  stage {st} skipped: indexing failed for {list(bad)}")
                continue
            if args.warmup and not warmed:
                print("warm-up / cold start (logged separately, not in the statistics) ...")
                wq = {"id": "WARMUP", "stage": st, "kind": "warmup", "question": WARMUP_QUESTION,
                      "grading": "ungraded"}
                cold = run_one(wq, stage_repos, rep=0, warmup=True)
                cold["residency"] = residency()
                warmup_rec.update(cold)
                save(cold, name="WARMUP")
                print(f"  cold start: {cold['wall_ms']/1000:.1f}s (generation {cold['generation_ms']/1000:.1f}s)"
                      + (f"  residency: {cold['residency']}" if cold["residency"] else ""))
                warmed = True
            for q in qs:
                for rep in range(args.repeat):
                    rec = run_one(q, stage_repos, rep=rep)
                    save(rec)
                    results.append(rec)
                    c = rec["checks"]
                    print(f"  {rec['id']:<9} {rec['outcome']:<12} {rec['wall_ms']/1000:6.1f}s  "
                          f"auto={c['auto_pass']}  sources={len(rec['sources'])}"
                          + (f"  ERROR {rec['error']}" if rec["error"] else ""))
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        print("evaluation aborted:", err)

    # ---- summary -------------------------------------------------------------------------
    walls = [r["wall_ms"] for r in results]
    gens = [r["generation_ms"] for r in results if r["generation_ms"]]
    graded = [r for r in results if r["checks"]["auto_pass"] is not None]
    summary = {
        "group": group, "config": args.config, "config_json": cfg, "runtime": runtime, "note": args.note,
        "questions_run": len(results), "answered": sum(r["outcome"] == "answered" for r in results),
        "errors": sum(r["outcome"] == "error" for r in results),
        "auto_pass_rate": (sum(bool(r["checks"]["auto_pass"]) for r in graded) / len(graded)) if graded else None,
        "wall_ms": {"mean": statistics.fmean(walls) if walls else 0, "p50": _pct(walls, .5), "p95": _pct(walls, .95),
                    "max": max(walls) if walls else 0},
        "generation_ms": {"mean": statistics.fmean(gens) if gens else 0, "p50": _pct(gens, .5), "p95": _pct(gens, .95)},
        "stages": stage_info, "mlflow_parent_run": parent, "error": err,
        "cold_start": ({"wall_ms": warmup_rec["wall_ms"], "generation_ms": warmup_rec["generation_ms"],
                        "node_ms": warmup_rec["node_ms"], "outcome": warmup_rec["outcome"],
                        "residency": warmup_rec.get("residency"), "mlflow_run_id": warmup_rec.get("mlflow_run_id")}
                       if warmup_rec else None),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    cols = ["id", "stage", "repeat_index", "outcome", "wall_ms", "generation_ms", "chunks_n", "auto_pass",
            "cite_all_ok", "cite_any_ok", "mention_ok", "file_recall", "warnings_n", "manual_score", "judge_score"]
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ([dict(warmup_rec, id="WARMUP", repeat_index=0)] if warmup_rec else []) + results:
            c = r["checks"]
            w.writerow({"id": r["id"], "stage": r["stage"], "repeat_index": r["repeat_index"], "outcome": r["outcome"],
                        "wall_ms": round(r["wall_ms"], 1), "generation_ms": round(r["generation_ms"], 1),
                        "chunks_n": r["chunks_n"], "auto_pass": c["auto_pass"], "cite_all_ok": c["cite_all_ok"],
                        "cite_any_ok": c["cite_any_ok"], "mention_ok": c["mention_ok"],
                        "file_recall": c.get("file_recall"), "warnings_n": len(r["warnings"]),
                        "manual_score": r["manual_score"], "judge_score": (r["judge"] or {}).get("score")})
    cold_metrics = {}
    if warmup_rec:
        cold_metrics = {"cold_start_wall_ms": warmup_rec["wall_ms"], "cold_start_generation_ms": warmup_rec["generation_ms"]}
        for m, v in (warmup_rec.get("residency") or {}).items():
            cold_metrics["ollama_vram_fraction"] = v["vram_fraction"]
            cold_metrics["ollama_vram_mb"] = v["vram_mb"]
    tracking.log_extra(parent, metrics={**cold_metrics,
        "questions_run": summary["questions_run"], "answered": summary["answered"], "errors": summary["errors"],
        "auto_pass_rate": summary["auto_pass_rate"] if summary["auto_pass_rate"] is not None else 0.0,
        "wall_ms_mean": summary["wall_ms"]["mean"], "wall_ms_p50": summary["wall_ms"]["p50"],
        "wall_ms_p95": summary["wall_ms"]["p95"], "generation_ms_mean": summary["generation_ms"]["mean"]},
        artifacts={"summary.json": summary})
    tracking.end_group_run(parent, error=err)
    print(f"\n{summary['answered']}/{summary['questions_run']} answered, auto-pass {summary['auto_pass_rate']}, "
          f"wall p50 {summary['wall_ms']['p50']/1000:.1f}s p95 {summary['wall_ms']['p95']/1000:.1f}s")
    print(f"results: {out_dir}")
    if parent:
        print(f"MLflow parent run: {tracking.run_url(parent)}")

    if args.judge and results:
        import judge
        judge.judge_dir(out_dir)
    return 1 if err else 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    raise SystemExit(main())
