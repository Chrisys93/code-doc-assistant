#!/usr/bin/env python3
"""
host_sampler.py -- samples GPU and memory on the machine that runs the model servers, once per interval, to CSV.

Run it on the HOST (WSL2 shell, where nvidia-smi works) for the whole evaluation, started ~30 s before the first
question so an idle baseline is captured:

    python eval/host_sampler.py --out eval/results/samples.csv --interval 1 --k8s-namespace code-doc

Columns: epoch, gpu_mem_used_mb, gpu_mem_total_mb, gpu_util_pct, gpu_power_w, gpu_temp_c, gpu_sm_clock_mhz,
host_mem_used_mb, host_mem_avail_mb, host_cpu_pct, k8s_<pod>_cpu_m / k8s_<pod>_mem_mi (when --k8s-namespace is
given and metrics-server works: `minikube addons enable metrics-server`).

Join to question time windows with eval/join_resources.py. Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import csv
import signal
import shutil
import subprocess
import time

GPU_FIELDS = ["memory.used", "memory.total", "utilization.gpu", "power.draw", "temperature.gpu", "clocks.sm"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def sample_gpu() -> dict:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if not out:
            return {}
        v = [_num(x.strip()) for x in out[0].split(",")]
        keys = ["gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_util_pct", "gpu_power_w", "gpu_temp_c", "gpu_sm_clock_mhz"]
        return dict(zip(keys, v))
    except Exception:  # noqa: BLE001
        return {}


def sample_host() -> dict:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return {"host_mem_used_mb": (vm.total - vm.available) / 1048576, "host_mem_avail_mb": vm.available / 1048576,
                "host_cpu_pct": psutil.cpu_percent(interval=None)}
    except ImportError:
        info = {}
        try:
            for line in open("/proc/meminfo"):
                k, v = line.split(":")
                info[k] = float(v.split()[0]) / 1024
            return {"host_mem_used_mb": info["MemTotal"] - info["MemAvailable"], "host_mem_avail_mb": info["MemAvailable"]}
        except Exception:  # noqa: BLE001
            return {}


def sample_k8s(ns: str) -> dict:
    if not shutil.which("kubectl"):
        return {}
    try:
        out = subprocess.run(["kubectl", "top", "pod", "-n", ns, "--no-headers"], capture_output=True, text=True,
                             timeout=8).stdout.strip().splitlines()
    except Exception:  # noqa: BLE001
        return {}
    row = {}
    for line in out:
        p = line.split()
        if len(p) >= 3:
            name = p[0]
            row[f"k8s_{name}_cpu_m"] = _num(p[1].rstrip("m"))
            row[f"k8s_{name}_mem_mi"] = _num(p[2].rstrip("Mi"))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--k8s-namespace", default="")
    ap.add_argument("--k8s-every", type=int, default=5, help="sample kubectl top every N intervals (it is slow)")
    args = ap.parse_args()
    if not shutil.which("nvidia-smi"):
        print("WARNING: nvidia-smi not found - GPU columns will be empty (run this in the WSL2 shell, not in a pod)")
    # A background job in a non-interactive shell ignores SIGINT, and a script stops it with SIGTERM: handle both,
    # so the CSV is always flushed.
    def _stop(signum, frame):  # noqa: ARG001
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    rows, fields, n = [], ["epoch"], 0
    k8s = {}
    print(f"sampling every {args.interval}s to {args.out}  (Ctrl+C to stop)")
    try:
        while True:
            t = time.time()
            if args.k8s_namespace and n % args.k8s_every == 0:
                k8s = sample_k8s(args.k8s_namespace)
            row = {"epoch": round(t, 3), **sample_gpu(), **sample_host(), **k8s}
            for k in row:
                if k not in fields:
                    fields.append(k)
            rows.append(row)
            n += 1
            if n % 30 == 0:
                with open(args.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(rows)
            time.sleep(max(0.0, args.interval - (time.time() - t)))
    except KeyboardInterrupt:
        pass
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
