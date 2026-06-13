"""
Recall benchmark for the approximate kernel.

Usage:
  python -m bench.run_recall --out results/recall.csv
  python -m bench.run_recall --dry-run
"""

import argparse
import csv
import os
import numpy as np
import torch

from bench.grid import RECALL_GRID, iter_grid, dry_run_cell
from streamtopk_sae.utils import make_inputs, str_to_dtype, recall_at_k
from streamtopk_sae.reference import reference_topk_sae
from streamtopk_sae.ops import topk_sae_cuda_approx

SUPPORTED_C = [16, 32, 64, 128]


def _parse_ints(s):
    return [int(x) for x in s.split(",")]


def _parse_strs(s):
    return [x.strip() for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default="results/recall.csv")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--B",     type=_parse_ints, default=None, metavar="B,...")
    parser.add_argument("--d",     type=_parse_ints, default=None, metavar="d,...")
    parser.add_argument("--F",     type=_parse_ints, default=None, metavar="F,...")
    parser.add_argument("--k",     type=_parse_ints, default=None, metavar="k,...")
    parser.add_argument("--dtype", type=_parse_strs, default=None, metavar="dtype,...")
    args = parser.parse_args()

    def apply_filters(cells):
        for cell in cells:
            if args.B     and cell["B"]     not in args.B:     continue
            if args.d     and cell["d"]     not in args.d:     continue
            if args.F     and cell["F"]     not in args.F:     continue
            if args.k     and cell["k"]     not in args.k:     continue
            if args.dtype and cell["dtype"] not in args.dtype: continue
            yield cell

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    grid  = RECALL_GRID
    if args.dry_run:
        cells = [dry_run_cell(grid)]
    else:
        cells = list(apply_filters(iter_grid(grid)))

    rows = []
    for cell in cells:
        B, d, F, k = cell["B"], cell["d"], cell["F"], cell["k"]
        c_mult      = cell["c_multiplier"]
        dtype_str   = cell["dtype"]
        dtype        = str_to_dtype(dtype_str)

        c = k * c_mult
        # Snap to supported
        c = min(SUPPORTED_C, key=lambda s: (abs(s - c), s))
        if c < k:
            c = min(s for s in SUPPORTED_C if s >= k)

        try:
            X, W_enc, b_enc = make_inputs(B, d, F, dtype=dtype, device="cuda")
            ref_v, ref_i = reference_topk_sae(X, W_enc, b_enc, k)
            pred_v, pred_i = topk_sae_cuda_approx(X, W_enc, b_enc, k, c)

            recalls = []
            for i in range(B):
                true_set = set(ref_i[i].tolist())
                pred_set = set(pred_i[i].tolist())
                recalls.append(len(true_set & pred_set) / k)

            row = {
                "B": B, "d": d, "F": F, "k": k,
                "c": c, "c_multiplier": c_mult, "dtype": dtype_str,
                "recall_mean": float(np.mean(recalls)),
                "recall_std":  float(np.std(recalls)),
                "recall_min":  float(np.min(recalls)),
                "recall_p10":  float(np.percentile(recalls, 10)),
                "recall_p50":  float(np.median(recalls)),
                "recall_p90":  float(np.percentile(recalls, 90)),
                "status": "ok",
            }
        except torch.cuda.OutOfMemoryError:
            row = {"B": B, "d": d, "F": F, "k": k, "c": c, "c_multiplier": c_mult,
                   "dtype": dtype_str, "status": "OOM"}
        except Exception as e:
            row = {"B": B, "d": d, "F": F, "k": k, "c": c, "c_multiplier": c_mult,
                   "dtype": dtype_str, "status": f"ERR:{e}"}

        rows.append(row)
        print(row)
        torch.cuda.empty_cache()

    if args.dry_run:
        print(f"\nDry-run complete ({len(rows)} row(s)) — results NOT written to disk")
        return

    fieldnames = ["B", "d", "F", "k", "c", "c_multiplier", "dtype",
                  "recall_mean", "recall_std", "recall_min",
                  "recall_p10", "recall_p50", "recall_p90", "status"]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
