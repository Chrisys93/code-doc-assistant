#!/usr/bin/env python3
"""
bench.py -- raw serving benchmark for any OpenAI-compatible endpoint (llama-server, Ollama /v1, vLLM, Strata).

It measures the model server, not the pipeline: time to first token (TTFT), prefill speed, decode speed and total
time, for several prompt sizes, with a fixed prompt so results are comparable across backends and models.

    python eval/bench.py --base-url http://llamacpp:8080/v1 --label heavy-llamacpp \\
        --prompt-tokens 256,1024,3000 --max-tokens 128 --runs 3

Why synthetic prompts: the pipeline's own latency mixes retrieval, planning and generation; this isolates the
server, so "backend A vs backend B" is a statement about the backend. The prompt-size sweep also exposes the
context-size / speed trade-off (long prompts are prefill-bound).

Numbers: llama-server reports exact `timings` (prompt_per_second, predicted_per_second) in the final chunk when
available and those are preferred; otherwise speeds come from the client-side clock and the reported usage.
Records to MLflow (own run, tag run.kind=bench) when MLFLOW_TRACKING_URI is reachable, and always to JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FILLER = ("The cache placement strategy assigns each content item to a node by hashing its name, so that every "
          "request for the same item is routed to the same cache. ")


def make_prompt(approx_tokens: int) -> str:
    chars = max(40, approx_tokens * 4)
    body = (FILLER * (chars // len(FILLER) + 1))[:chars]
    return body + "\n\nIn one short paragraph, summarise the text above."


def stream_once(base_url: str, model: str, prompt: str, max_tokens: int, api_key: str = "none",
                timeout: float = 600.0) -> dict:
    import requests
    body = {"model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = {}
    timings = {}
    with requests.post(base_url.rstrip("/") + "/chat/completions", json=body, stream=True, timeout=timeout,
                       headers={"Authorization": f"Bearer {api_key}"}) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith(b"data:"):
                continue
            data = raw[5:].strip()
            if data == b"[DONE]":
                break
            try:
                j = json.loads(data)
            except ValueError:
                continue
            if j.get("usage"):
                usage = j["usage"]
            if j.get("timings"):
                timings = j["timings"]
            for ch in j.get("choices") or []:
                delta = (ch.get("delta") or {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
    total = time.perf_counter() - t0
    ptok = usage.get("prompt_tokens") or timings.get("prompt_n")
    ctok = usage.get("completion_tokens") or timings.get("predicted_n") or chunks
    decode_s = max(total - (ttft or 0.0), 1e-9)
    out = {
        "ttft_s": ttft, "total_s": total, "prompt_tokens": ptok, "completion_tokens": ctok,
        "decode_tps": timings.get("predicted_per_second") or ((ctok - 1) / decode_s if ctok and ctok > 1 else None),
        "prefill_tps": timings.get("prompt_per_second") or ((ptok / ttft) if ptok and ttft else None),
        "server_timings": bool(timings),
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base, e.g. http://llamacpp:8080/v1")
    ap.add_argument("--model", default="default", help="model id sent in the request (llama-server ignores it)")
    ap.add_argument("--label", required=True, help="name of this backend/model/settings combination")
    ap.add_argument("--prompt-tokens", default="256,1024,3000")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "bench"))
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args(argv)

    sizes = [int(x) for x in args.prompt_tokens.split(",") if x.strip()]
    api_key = os.environ.get("OPENAI_API_KEY", "none")
    if not args.no_warmup:
        print("warm-up ...")
        stream_once(args.base_url, args.model, make_prompt(64), 8, api_key)
    rows = []
    for n in sizes:
        prompt = make_prompt(n)
        runs = []
        for i in range(args.runs):
            r = stream_once(args.base_url, args.model, prompt, args.max_tokens, api_key)
            runs.append(r)
            print(f"  ~{n:>5} tok  run {i+1}: ttft {r['ttft_s'] or 0:6.2f}s  prefill {r['prefill_tps'] or 0:7.1f} t/s  "
                  f"decode {r['decode_tps'] or 0:6.1f} t/s  ({r['completion_tokens']} out)")
        def med(key):
            v = [x[key] for x in runs if x.get(key) is not None]
            return statistics.median(v) if v else None
        rows.append({"target_prompt_tokens": n, "prompt_tokens": med("prompt_tokens"), "runs": runs,
                     "ttft_s_median": med("ttft_s"), "prefill_tps_median": med("prefill_tps"),
                     "decode_tps_median": med("decode_tps"), "total_s_median": med("total_s")})
    result = {"label": args.label, "base_url": args.base_url, "model": args.model, "max_tokens": args.max_tokens,
              "runs": args.runs, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{time.strftime('%Y%m%d-%H%M%S')}-{args.label}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("saved", path)

    if not args.no_mlflow and os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            for cand in (os.environ.get("EVAL_SRC"), "/app/src", os.path.join(HERE, "..", "src")):
                if cand and os.path.isfile(os.path.join(cand, "tracking.py")):
                    sys.path.insert(0, os.path.abspath(cand))
                    break
            import tracking
            rid = tracking.start_group_run(f"bench {args.label}", params={
                "bench.label": args.label, "bench.base_url": args.base_url, "bench.max_tokens": args.max_tokens,
                "bench.runs": args.runs, "bench.sizes": sizes}, tags={"run.kind": "bench", "bench.label": args.label})
            m = {}
            for row in rows:
                k = row["target_prompt_tokens"]
                for name in ("ttft_s_median", "prefill_tps_median", "decode_tps_median", "total_s_median"):
                    if row[name] is not None:
                        m[f"p{k}_{name}"] = row[name]
            tracking.log_extra(rid, metrics=m, artifacts={"bench.json": result})
            tracking.end_group_run(rid)
            print("MLflow:", tracking.run_url(rid))
        except Exception as e:  # noqa: BLE001
            print("MLflow logging skipped:", e)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
