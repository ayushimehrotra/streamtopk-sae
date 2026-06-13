"""
Runtime benchmark: measures median/p10/p90/min/max latency per (impl, B, d, F, k, dtype) cell.

Usage:
  python -m bench.run_runtime --impl cpu,cuda_exact,cuda_approx,eager,compiled,triton \
      --warmup 10 --iters 100 --out results/runtime.csv
  python -m bench.run_runtime --dry-run
"""

import argparse
import csv
import os
import time
import random
import itertools
import numpy as np
import torch

from bench.grid import BENCH_GRID, BENCH_GRID_CPU, iter_grid, dry_run_cell
from streamtopk_sae.utils import make_inputs, str_to_dtype
from streamtopk_sae.reference import reference_topk_sae
from streamtopk_sae.baselines import baseline_eager, baseline_compiled, baseline_triton
from streamtopk_sae.ops import topk_sae_cpu, topk_sae_cuda_exact, topk_sae_cuda_approx


IMPL_FNS = {
    "eager":       lambda X, W, b, k: baseline_eager(X, W, b, k),
    "compiled":    lambda X, W, b, k: baseline_compiled(X, W, b, k),
    "triton":      lambda X, W, b, k: baseline_triton(X, W, b, k),
    "cpu":         lambda X, W, b, k: topk_sae_cpu(X, W, b, k),
    "cuda_exact":  lambda X, W, b, k: topk_sae_cuda_exact(X, W, b, k),
    "cuda_approx": lambda X, W, b, k: topk_sae_cuda_approx(X, W, b, k, max(16, k)),
}

DEFAULT_IMPLS = list(IMPL_FNS.keys())


def time_cuda_impl(fn, X, W, b, k, warmup, iters):
    for _ in range(warmup):
        fn(X, W, b, k)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(X, W, b, k)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us
    return times


def time_cpu_impl(fn, X, W, b, k, warmup, iters):
    for _ in range(warmup):
        fn(X, W, b, k)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(X, W, b, k)
        times.append((time.perf_counter() - t0) * 1e6)
    return times


def bench_cell(impl_name, fn, cell, warmup, iters):
    B, d, F, k = cell["B"], cell["d"], cell["F"], cell["k"]
    dtype_str   = cell["dtype"]
    dtype        = str_to_dtype(dtype_str)

    is_cpu = impl_name == "cpu"
    device = "cpu" if is_cpu else "cuda"

    try:
        X, W_enc, b_enc = make_inputs(B, d, F, dtype=dtype, device=device)

        if is_cpu:
            times = time_cpu_impl(fn, X, W_enc, b_enc, k, warmup, iters)
        else:
            times = time_cuda_impl(fn, X, W_enc, b_enc, k, warmup, iters)

        return {
            "impl":   impl_name,
            "B": B, "d": d, "F": F, "k": k,
            "dtype":  dtype_str,
            "device": torch.cuda.get_device_name(0) if not is_cpu else "cpu",
            "median_us": float(np.median(times)),
            "p10_us":    float(np.percentile(times, 10)),
            "p90_us":    float(np.percentile(times, 90)),
            "min_us":    float(np.min(times)),
            "max_us":    float(np.max(times)),
            "status":    "ok",
        }
    except torch.cuda.OutOfMemoryError:
        return {"impl": impl_name, "B": B, "d": d, "F": F, "k": k,
                "dtype": dtype_str, "status": "OOM"}
    except Exception as e:
        return {"impl": impl_name, "B": B, "d": d, "F": F, "k": k,
                "dtype": dtype_str, "status": f"ERR:{e}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl",    default=",".join(DEFAULT_IMPLS),
                        help="Comma-separated list of impls to benchmark")
    parser.add_argument("--warmup",  type=int, default=10)
    parser.add_argument("--iters",   type=int, default=100)
    parser.add_argument("--out",     default="results/runtime.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run one cell per impl as smoke test")
    args = parser.parse_args()

    impls = [x.strip() for x in args.impl.split(",")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows = []
    for impl_name in impls:
        fn = IMPL_FNS[impl_name]
        grid = BENCH_GRID_CPU if impl_name == "cpu" else BENCH_GRID

        cells = [dry_run_cell(grid)] if args.dry_run else list(iter_grid(grid))

        # Shuffle for randomized order
        if not args.dry_run:
            random.shuffle(cells)

        warmup = 2 if args.dry_run else args.warmup
        iters  = 3 if args.dry_run else args.iters

        for cell in cells:
            if impl_name == "cpu" and cell.get("dtype", "fp32") != "fp32":
                continue
            row = bench_cell(impl_name, fn, cell, warmup, iters)
            rows.append(row)
            print(row)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.dry_run:
        print(f"\nDry-run complete ({len(rows)} row(s)) — results NOT written to disk")
        return

    fieldnames = ["impl", "B", "d", "F", "k", "dtype", "device",
                  "median_us", "p10_us", "p90_us", "min_us", "max_us", "status"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
