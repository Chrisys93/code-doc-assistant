#!/usr/bin/env python3
"""
judge.py -- OPTIONAL LLM-as-judge over saved evaluation results (never part of the answering pipeline).

    ANTHROPIC_API_KEY=... python eval/judge.py eval/results/<group>/<config> [--model ID] [--mlflow]

For each <question>.json it sends the question, the reference notes from questions.json, the answer and the
retrieved source files to a Claude model and records a 0-2 score, a grounding score, hallucinated items and a one-
paragraph justification ("judge" in the JSON, `judge_score` in summary.csv, MLflow metrics on the question run).

Design rules
  * BLIND: the judge never sees the backend, model, config or latency, so it cannot favour a setup.
  * The reference notes are guidance, not an answer key: answers that are right in different words score well.
  * Everything is reproducible: the judge model id, prompt version and rubric go into every record.
  * Calibrate before trusting it: fill `manual_score` (0-2) in a sample of the question JSONs and run with
    --calibrate to see agreement. A category the judge disagrees on stays manual.
  * Off by default; it sends answers and retrieved file names to an external API - only for repos you may share.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROMPT_VERSION = "judge-v1"
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5")

SYSTEM = """You are a strict, fair grader of answers produced by a code-documentation assistant.
You are given a QUESTION, REFERENCE NOTES written by the evaluation author, optional CHECKS the answer should satisfy,
the ANSWER, and the list of FILES the assistant retrieved. You do not know which model produced the answer.

Score 0-2:
  0 = wrong, invented, or accepts a false premise in the question
  1 = partly right, or right but not grounded in the retrieved files, or misses important parts of the reference
  2 = right, grounded in the retrieved files, and complete enough for the question
Also give grounding 0-2 (are the claims supported by the retrieved FILES? the answer's own citations count only if the
file was retrieved) and list hallucinated items (files, functions, settings or facts that do not exist or are not supported).
Reference notes are guidance: accept correct answers phrased differently, and penalise confident claims the notes contradict.
For open-ended questions with a RUBRIC, score against the rubric and say so.
Reply with ONLY a JSON object:
{"score": 0|1|2, "grounding": 0|1|2, "premise_corrected": true|false|null, "hallucinated": ["..."], "justification": "..."}"""


def build_user(rec: dict) -> str:
    parts = [f"QUESTION:\n{rec['question']}",
             f"REFERENCE NOTES:\n{rec.get('expected') or '(none)'}"]
    if rec.get("manual_checks"):
        parts.append("CHECKS:\n- " + "\n- ".join(rec["manual_checks"]))
    if rec.get("rubric"):
        parts.append("RUBRIC:\n" + json.dumps(rec["rubric"], indent=1))
    parts.append("FILES RETRIEVED:\n" + (", ".join(rec.get("retrieved_files") or []) or "(none)"))
    parts.append("ANSWER:\n" + (rec.get("answer") or "(empty)")[:12000])
    return "\n\n".join(parts)


def _client():
    try:
        import anthropic
    except ImportError:
        sys.exit("the 'anthropic' package is not installed (pip install anthropic)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic()


def judge_one(client, model: str, rec: dict) -> dict:
    kw = dict(model=model, max_tokens=800, system=SYSTEM, messages=[{"role": "user", "content": build_user(rec)}])
    try:
        msg = client.messages.create(temperature=0, **kw)
    except Exception as e:  # noqa: BLE001 - some model versions reject sampling parameters
        if "temperature" not in str(e).lower():
            raise
        msg = client.messages.create(**kw)
    text = "".join(getattr(b, "text", "") for b in msg.content).strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        j = json.loads(text[start:end + 1])
    except ValueError:
        j = {"score": None, "grounding": None, "justification": f"unparseable judge reply: {text[:300]}"}
    j.update(judge_model=model, prompt_version=PROMPT_VERSION)
    return j


def judge_dir(path: str, model: str | None = None, to_mlflow: bool = True, calibrate: bool = False) -> int:
    model = model or DEFAULT_MODEL
    client = _client()
    files = sorted(p for p in glob.glob(os.path.join(path, "*.json")) if not p.endswith("summary.json"))
    scored = []
    for p in files:
        rec = json.load(open(p, encoding="utf-8"))
        if "question" not in rec or rec.get("kind") == "warmup":
            continue
        if rec.get("outcome") == "error":
            rec["judge"] = {"score": 0, "grounding": 0, "justification": "no answer (run error)",
                            "judge_model": model, "prompt_version": PROMPT_VERSION}
        else:
            rec["judge"] = judge_one(client, model, rec)
        json.dump(rec, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
        sc = rec["judge"].get("score")
        scored.append((rec, sc))
        print(f"  {rec['id']:<9} judge={sc}  {str(rec['judge'].get('justification'))[:100]}")
        if to_mlflow and rec.get("mlflow_run_id") and sc is not None:
            try:
                src = os.environ.get("EVAL_SRC") or next((c for c in ("/app/src", os.path.join(os.path.dirname(__file__), "..", "src"))
                                                          if os.path.isfile(os.path.join(c, "tracking.py"))), None)
                if src:
                    sys.path.insert(0, os.path.abspath(src))
                import tracking
                tracking.log_extra(rec["mlflow_run_id"], metrics={"judge_score": sc,
                                   "judge_grounding": rec["judge"].get("grounding") or 0},
                                   tags={"judge.model": model, "judge.prompt": PROMPT_VERSION},
                                   artifacts={"judge.json": rec["judge"]})
            except Exception as e:  # noqa: BLE001
                print("    MLflow logging skipped:", e)
    vals = [s for _, s in scored if s is not None]
    if vals:
        print(f"mean judge score {sum(vals)/len(vals):.2f} over {len(vals)} answers")
    if calibrate:
        pairs = [(r["manual_score"], s) for r, s in scored if r.get("manual_score") is not None and s is not None]
        if pairs:
            exact = sum(a == b for a, b in pairs) / len(pairs)
            mad = sum(abs(a - b) for a, b in pairs) / len(pairs)
            print(f"calibration vs manual ({len(pairs)} answers): exact agreement {exact:.0%}, mean abs diff {mad:.2f}")
        else:
            print("calibration: no answers with both manual_score and a judge score yet")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args(argv)
    return judge_dir(a.results_dir, a.model, to_mlflow=not a.no_mlflow, calibrate=a.calibrate)


if __name__ == "__main__":
    raise SystemExit(main())
