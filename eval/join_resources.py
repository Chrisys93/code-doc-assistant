#!/usr/bin/env python3
"""
join_resources.py -- attach host resource samples (eval/host_sampler.py) to each question's time window.

    python eval/join_resources.py --samples eval/results/samples.csv --results eval/results/<group>/<config> [--mlflow]

For every <question>.json it takes the samples between t_start and t_end and computes peak/mean GPU memory,
GPU utilisation, power, host memory and per-pod k8s memory/CPU, relative to an idle baseline (median of the
samples before the first question). Writes them into the JSON ("resources"), to resources.csv, and with --mlflow
logs them as metrics on each question's run (needs MLFLOW_TRACKING_URI reachable from the host, e.g. via the
socat proxy: MLFLOW_TRACKING_URI=http://localhost:5000).

The pod clocks and the host clock are the same machine's clock (docker/minikube on WSL2), so epochs line up.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys


def load_samples(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", None) else None) for k, v in r.items()})
    return rows


def _vals(rows, key):
    return [r[key] for r in rows if r.get(key) is not None]


def summarise(window: list[dict], base: dict) -> dict:
    out: dict[str, float] = {"samples_n": len(window)}
    if not window:
        return out
    keys = sorted({k for r in window for k in r if k != "epoch"})
    for k in keys:
        v = _vals(window, k)
        if not v:
            continue
        out[f"{k}_peak"] = max(v)
        out[f"{k}_mean"] = statistics.fmean(v)
        if k in base and k.endswith(("_mb", "_mi")):
            out[f"{k}_peak_over_idle"] = max(v) - base[k]
    return out


def _log_parent(client, args, table, base) -> None:
    """Attach the raw samples + per-question table to the PARENT run, with run-level peaks/means, so the whole
    measurement can be re-checked from MLflow alone."""
    try:
        summ = json.load(open(os.path.join(args.results, "summary.json"), encoding="utf-8"))
    except Exception:  # noqa: BLE001
        summ = {}
    parent = summ.get("mlflow_parent_run")
    if not parent:
        print("no mlflow_parent_run in summary.json - parent run not updated")
        return
    try:
        from mlflow.entities import Metric
        import time
        now = int(time.time() * 1000)
        m = {}
        def agg(key, fn, name):
            v = [r[key] for r in table if r.get(key) is not None and r["id"] != "WARMUP" and r[key] == r[key]]
            if v:
                m[name] = float(fn(v))
        agg("gpu_mem_used_mb_peak", max, "res_gpu_mem_used_mb_peak")
        agg("gpu_util_pct_mean", statistics.mean, "res_gpu_util_pct_mean")
        agg("gpu_power_w_mean", statistics.mean, "res_gpu_power_w_mean")
        agg("host_mem_used_mb_peak", max, "res_host_mem_used_mb_peak")
        for k, v in base.items():
            m[f"res_idle_{k}"] = float(v)
        client.log_batch(parent, metrics=[Metric(k, v, now, 0) for k, v in m.items()])
        client.log_artifact(parent, args.samples, "resources")
        client.log_artifact(parent, os.path.join(args.results, "resources.csv"), "resources")
        print(f"parent run {parent}: logged {len(m)} resource metrics + samples/resources.csv artifacts")
    except Exception as e:  # noqa: BLE001
        print(f"could not update the parent run: {e}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--results", required=True, help="results/<group>/<config> directory")
    ap.add_argument("--mlflow", action="store_true")
    args = ap.parse_args(argv)

    samples = load_samples(args.samples)
    files = sorted(p for p in glob.glob(os.path.join(args.results, "*.json")) if not p.endswith("summary.json"))
    recs = [(p, json.load(open(p, encoding="utf-8"))) for p in files]
    recs = [(p, r) for p, r in recs if r.get("t_start")]
    if not recs or not samples:
        print("nothing to join (no question files with t_start, or no samples)")
        return 1
    first = min(r["t_start"] for _, r in recs)
    before = [s for s in samples if s["epoch"] < first]
    base = {}
    if before:
        for k in {k for s in before for k in s if k != "epoch"}:
            v = _vals(before, k)
            if v:
                base[k] = statistics.median(v)
    else:
        print("WARNING: no samples before the first question - no idle baseline (start the sampler earlier)")

    client = None
    if args.mlflow:
        try:
            from mlflow import MlflowClient
            client = MlflowClient()
        except Exception as e:  # noqa: BLE001
            print("MLflow unavailable:", e)

    table = []
    for p, r in recs:
        win = [s for s in samples if r["t_start"] <= s["epoch"] <= r["t_end"]]
        res = summarise(win, base)
        r["resources"] = res
        r["resources_idle_baseline"] = base
        json.dump(r, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=str)
        table.append({"id": r["id"], "repeat_index": r.get("repeat_index", 0), "wall_ms": r["wall_ms"], **res})
        if client and r.get("mlflow_run_id"):
            try:
                from mlflow.entities import Metric
                import time
                now = int(time.time() * 1000)
                client.log_batch(r["mlflow_run_id"], metrics=[Metric(f"res_{k}", float(v), now, 0)
                                                              for k, v in res.items() if v is not None])
            except Exception as e:  # noqa: BLE001
                print(f"  {r['id']}: could not log to MLflow: {e}")

    cols = sorted({k for row in table for k in row}, key=lambda k: (k not in ("id", "repeat_index", "wall_ms"), k))
    with open(os.path.join(args.results, "resources.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    if client:
        _log_parent(client, args, table, base)  # after resources.csv exists (it is attached as an artifact)
    for row in table:
        print(f"  {row['id']:<9} samples={row['samples_n']:<4} "
              f"vram_peak={row.get('gpu_mem_used_mb_peak', float('nan')):7.0f}MB "
              f"gpu_util_mean={row.get('gpu_util_pct_mean', float('nan')):5.1f}% "
              f"power_mean={row.get('gpu_power_w_mean', float('nan')):5.1f}W "
              f"host_mem_peak={row.get('host_mem_used_mb_peak', float('nan')):7.0f}MB")
    print("wrote", os.path.join(args.results, "resources.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
