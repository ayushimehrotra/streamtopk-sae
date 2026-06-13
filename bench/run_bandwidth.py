"""
Achieved bandwidth and arithmetic intensity benchmark.

For each impl on representative cells:
  bytes_moved = bytes_X + bytes_W + bytes_bias + bytes_output
  bandwidth   = bytes_moved / runtime
  flops       = 2 * B * d * F  (matmul)
  AI          = flops / bytes_moved

Usage:
  python -m bench.run_bandwidth --out results/bandwidth.csv
  python -m bench.run_bandwidth --dry-run
"""

import argparse
import csv
import os
import time
import numpy as np
import torch

from streamtopk_sae.utils import make_inputs, str_to_dtype
from streamtopk_sae.baselines import baseline_eager, baseline_compiled, baseline_triton
from streamtopk_sae.ops import topk_sae_cpu, topk_sae_cuda_exact, topk_sae_cuda_approx

REPRESENTATIVE_CELLS = [
    dict(B=128,  d=768,  F=2**16, k=32, dtype="bf16"),
    dict(B=128,  d=2048, F=2**18, k=64, dtype="bf16"),
    dict(B=512,  d=2048, F=2**16, k=32, dtype="bf16"),
    dict(B=32,   d=4096, F=2**14, k=16, dtype="bf16"),
]

IMPL_FNS = {
    "eager":       lambda X, W, b, k: baseline_eager(X, W, b, k),
    "triton":      lambda X, W, b, k: baseline_triton(X, W, b, k),
    "cuda_exact":  lambda X, W, b, k: topk_sae_cuda_exact(X, W, b, k),
    "cuda_approx": lambda X, W, b, k: topk_sae_cuda_approx(X, W, b, k, max(16, k)),
}


def time_impl(fn, X, W, b, k, warmup=5, iters=20):
    for _ in range(warmup):
        fn(X, W, b, k)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn(X, W, b, k)
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1e-3)  # seconds
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default="results/bandwidth.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cells = [REPRESENTATIVE_CELLS[0]] if args.dry_run else REPRESENTATIVE_CELLS

    rows = []
    for cell in cells:
        B, d, F, k = cell["B"], cell["d"], cell["F"], cell["k"]
        dtype_str   = cell["dtype"]
        dtype        = str_to_dtype(dtype_str)
        scalar_bytes = 2 if dtype in (torch.float16, torch.bfloat16) else 4

        bytes_X    = B * d * scalar_bytes
        bytes_W    = F * d * scalar_bytes
        bytes_bias = F * 4
        bytes_out  = B * k * (4 + 4)
        bytes_moved = bytes_X + bytes_W + bytes_bias + bytes_out
        flops = 2 * B * d * F  # matmul dominates

        for impl_name, fn in IMPL_FNS.items():
            try:
                X, W_enc, b_enc = make_inputs(B, d, F, dtype=dtype, device="cuda")
                t = time_impl(fn, X, W_enc, b_enc, k,
                               warmup=2 if args.dry_run else 5,
                               iters=3  if args.dry_run else 20)
                bw_gb_s = bytes_moved / t / 1e9
                ai      = flops / bytes_moved
                row = {
                    "impl": impl_name, "B": B, "d": d, "F": F, "k": k,
                    "dtype": dtype_str,
                    "runtime_s":    t,
                    "bytes_moved":  bytes_moved,
                    "flops":        flops,
                    "bandwidth_GBs": bw_gb_s,
                    "arithmetic_intensity": ai,
                    "status": "ok",
                }
            except torch.cuda.OutOfMemoryError:
                row = {"impl": impl_name, "B": B, "d": d, "F": F, "k": k,
                       "dtype": dtype_str, "status": "OOM"}
            except Exception as e:
                row = {"impl": impl_name, "B": B, "d": d, "F": F, "k": k,
                       "dtype": dtype_str, "status": f"ERR:{e}"}
            rows.append(row)
            print(row)
            torch.cuda.empty_cache()

    if args.dry_run:
        print(f"\nDry-run complete ({len(rows)} row(s)) — results NOT written to disk")
        return

    fieldnames = ["impl", "B", "d", "F", "k", "dtype",
                  "runtime_s", "bytes_moved", "flops",
                  "bandwidth_GBs", "arithmetic_intensity", "status"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
