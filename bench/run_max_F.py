"""
Max feasible F benchmark: binary search for largest F that fits in memory.

Usage:
  python -m bench.run_max_F --out results/max_F.csv
  python -m bench.run_max_F --dry-run
"""

import argparse
import csv
import os
import torch

from streamtopk_sae.utils import make_inputs, str_to_dtype
from streamtopk_sae.baselines import baseline_eager
from streamtopk_sae.ops import topk_sae_cuda_exact, topk_sae_cuda_approx

CONFIGS = [
    dict(B=128,  d=768,  k=32,  dtype="bf16"),
    dict(B=2048, d=768,  k=32,  dtype="bf16"),  # large B forces eager score-matrix OOM at F>2M
]

IMPL_FNS = {
    "eager":       lambda X, W, b, k: baseline_eager(X, W, b, k),
    "cuda_exact":  lambda X, W, b, k: topk_sae_cuda_exact(X, W, b, k),
    "cuda_approx": lambda X, W, b, k: topk_sae_cuda_approx(X, W, b, k, max(16, k)),
}

F_MIN_POW = 14  # 2^14
F_MAX_POW = 22  # 2^22 = 4M; capped to avoid system OOM on shared 24GB GPU


def try_forward(impl_name, fn, B, d, F, k, dtype):
    try:
        torch.cuda.empty_cache()
        X, W_enc, b_enc = make_inputs(B, d, F, dtype=dtype, device="cuda")
        fn(X, W_enc, b_enc, k)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    except Exception:
        return False
    finally:
        torch.cuda.empty_cache()


def binary_search_max_F(impl_name, fn, B, d, k, dtype):
    lo, hi = F_MIN_POW, F_MAX_POW
    result = None
    while lo <= hi:
        mid = (lo + hi) // 2
        F   = 2 ** mid
        ok  = try_forward(impl_name, fn, B, d, F, k, dtype)
        if ok:
            result = F
            lo = mid + 1
        else:
            hi = mid - 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default="results/max_F.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    configs = [CONFIGS[0]] if args.dry_run else CONFIGS

    rows = []
    for cfg in configs:
        B, d, k = cfg["B"], cfg["d"], cfg["k"]
        dtype_str = cfg["dtype"]
        dtype      = str_to_dtype(dtype_str)

        for impl_name, fn in IMPL_FNS.items():
            if args.dry_run:
                # Just test one F
                F = 2 ** F_MIN_POW
                ok = try_forward(impl_name, fn, B, d, F, k, dtype)
                max_F = F if ok else None
            else:
                max_F = binary_search_max_F(impl_name, fn, B, d, k, dtype)

            row = {
                "impl": impl_name, "B": B, "d": d, "k": k, "dtype": dtype_str,
                "max_F": max_F if max_F is not None else "OOM",
            }
            rows.append(row)
            print(row)

    if args.dry_run:
        print(f"\nDry-run complete ({len(rows)} row(s)) — results NOT written to disk")
        return

    fieldnames = ["impl", "B", "d", "k", "dtype", "max_F"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
