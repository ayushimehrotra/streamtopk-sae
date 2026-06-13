"""
Peak memory benchmark.

Each GPU measurement is isolated in a subprocess so the CUDA caching allocator
starts clean for every cell — prevents inflated baseline_mem from prior large
allocations contaminating subsequent measurements.

CPU: resource.getrusage before/after in-process (already cheap enough).

Usage:
  python -m bench.run_memory --impl cuda_exact,eager,cuda_approx --out results/memory.csv
  python -m bench.run_memory --dry-run
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import torch

from bench.grid import BENCH_GRID, BENCH_GRID_CPU, iter_grid, dry_run_cell
from streamtopk_sae.utils import make_inputs, str_to_dtype
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


def bench_gpu_memory_single(impl_name, cell):
    """Measure one cell in-process. Only call this from a fresh subprocess."""
    fn = IMPL_FNS[impl_name]
    B, d, F, k = cell["B"], cell["d"], cell["F"], cell["k"]
    dtype = str_to_dtype(cell["dtype"])

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        X, W_enc, b_enc = make_inputs(B, d, F, dtype=dtype, device="cuda")
        baseline_mem = torch.cuda.max_memory_allocated()

        torch.cuda.reset_peak_memory_stats()
        fn(X, W_enc, b_enc, k)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()

        output_size = B * k * (4 + 4)  # values fp32 + indices int32
        working_mem = peak_mem - baseline_mem - output_size

        return {
            "impl": impl_name, "B": B, "d": d, "F": F, "k": k,
            "dtype": cell["dtype"],
            "baseline_mem_mb": baseline_mem / 1e6,
            "peak_mem_mb":     peak_mem / 1e6,
            "working_mem_mb":  working_mem / 1e6,
            "status": "ok",
        }
    except torch.cuda.OutOfMemoryError:
        return {"impl": impl_name, "B": B, "d": d, "F": F, "k": k,
                "dtype": cell["dtype"], "status": "OOM"}
    except Exception as e:
        return {"impl": impl_name, "B": B, "d": d, "F": F, "k": k,
                "dtype": cell["dtype"], "status": f"ERR:{e}"}


def bench_gpu_memory(impl_name, cell):
    """Spawn a subprocess to measure one GPU cell with a clean CUDA context."""
    cmd = [
        sys.executable, "-m", "bench.run_memory", "--_measure",
        impl_name,
        str(cell["B"]), str(cell["d"]), str(cell["F"]),
        str(cell["k"]), cell["dtype"],
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if result.returncode != 0:
            return {"impl": impl_name, **cell,
                    "status": f"ERR:{result.stderr[-300:].strip()}"}
        return json.loads(result.stdout.strip())
    except subprocess.TimeoutExpired:
        return {"impl": impl_name, **cell, "status": "TIMEOUT"}
    except Exception as e:
        return {"impl": impl_name, **cell, "status": f"ERR:{e}"}


def bench_cpu_memory(cell):
    import resource
    B, d, F, k = cell["B"], cell["d"], cell["F"], cell["k"]
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    X, W_enc, b_enc = make_inputs(B, d, F, dtype=torch.float32, device="cpu")
    topk_sae_cpu(X, W_enc, b_enc, k)
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "impl": "cpu", "B": B, "d": d, "F": F, "k": k, "dtype": "fp32",
        "working_mem_mb": (after - before) / 1024,
        "status": "ok",
    }


def _parse_ints(s):
    return [int(x) for x in s.split(",")]


def _parse_strs(s):
    return [x.strip() for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl",    default="cuda_exact,eager,cuda_approx")
    parser.add_argument("--out",     default="results/memory.csv")
    parser.add_argument("--dry-run", action="store_true")
    # Grid filters: comma-separated values restrict which cells are run
    parser.add_argument("--B",     type=_parse_ints, default=None, metavar="B,...",
                        help="Restrict batch sizes (e.g. --B 32,128)")
    parser.add_argument("--d",     type=_parse_ints, default=None, metavar="d,...",
                        help="Restrict hidden dims (e.g. --d 768)")
    parser.add_argument("--F",     type=_parse_ints, default=None, metavar="F,...",
                        help="Restrict latent counts (e.g. --F 16384,65536)")
    parser.add_argument("--k",     type=_parse_ints, default=None, metavar="k,...",
                        help="Restrict top-k values (e.g. --k 32)")
    parser.add_argument("--dtype", type=_parse_strs, default=None, metavar="dtype,...",
                        help="Restrict dtypes (e.g. --dtype bf16)")
    # Hidden: called by bench_gpu_memory() to run a single cell in a fresh process
    parser.add_argument("--_measure", nargs=6,
                        metavar=("IMPL", "B", "d", "F", "k", "DTYPE"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Subprocess single-cell mode: measure and print JSON, then exit
    if args._measure:
        impl_name, B, d, F, k, dtype_str = args._measure
        cell = {"B": int(B), "d": int(d), "F": int(F), "k": int(k), "dtype": dtype_str}
        row = bench_gpu_memory_single(impl_name, cell)
        print(json.dumps(row))
        return

    impls = [x.strip() for x in args.impl.split(",")]

    def apply_filters(cells):
        for cell in cells:
            if args.B     and cell["B"]     not in args.B:     continue
            if args.d     and cell["d"]     not in args.d:     continue
            if args.F     and cell["F"]     not in args.F:     continue
            if args.k     and cell["k"]     not in args.k:     continue
            if args.dtype and cell["dtype"] not in args.dtype: continue
            yield cell

    rows = []
    for impl_name in impls:
        grid = BENCH_GRID_CPU if impl_name == "cpu" else BENCH_GRID
        if args.dry_run:
            cells = [dry_run_cell(grid)]
        else:
            cells = list(apply_filters(iter_grid(grid)))

        for cell in cells:
            if impl_name == "cpu":
                row = bench_cpu_memory(cell)
            else:
                row = bench_gpu_memory(impl_name, cell)
            rows.append(row)
            print(row)

    if args.dry_run:
        print(f"\nDry-run complete ({len(rows)} row(s)) — results NOT written to disk")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = ["impl", "B", "d", "F", "k", "dtype",
                  "baseline_mem_mb", "peak_mem_mb", "working_mem_mb", "status"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
