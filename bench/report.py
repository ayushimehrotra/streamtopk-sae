"""
Report generator: reads CSVs from results/ and produces tables and plots.

Usage:
  python -m bench.report
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = "results"
REPORT_DIR  = os.path.join(RESULTS_DIR, "report")

IMPL_LABELS = {
    "eager":       "Eager (cuBLAS)",
    "cuda_exact":  "cuda_exact",
    "cuda_approx": "cuda_approx",
    "triton":      "Triton",
    "cpu":         "CPU (OpenMP)",
}
IMPL_COLORS = {
    "eager":       "#1f77b4",
    "cuda_exact":  "#ff7f0e",
    "cuda_approx": "#2ca02c",
    "triton":      "#9467bd",
    "cpu":         "#8c564b",
}

# NVIDIA RTX A5000 specs
PEAK_BF16_GFLOPS   = 222_200   # 222.2 TFLOPS bf16 tensor core
PEAK_HBM_GBS       = 768       # GB/s


def load_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping.")
        return None
    df = pd.read_csv(path)
    for col in ["median_us", "bandwidth_GBs", "arithmetic_intensity",
                "flops", "runtime_s", "working_mem_mb", "recall_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["status"] == "ok"] if "status" in df.columns else df


def shape_label(row):
    return f"B={row['B']}\nF={int(row['F'])//1024}K\nk={row['k']}"


# ---------------------------------------------------------------------------
# Plot 1: Proper roofline (GFLOPS vs FLOP/byte, log-log)
# ---------------------------------------------------------------------------
def make_roofline_plot(bw_df):
    if bw_df is None:
        return

    bw_df = bw_df.copy()
    bw_df["achieved_GFLOPS"] = bw_df["flops"] / bw_df["runtime_s"] / 1e9

    fig, ax = plt.subplots(figsize=(8, 5))

    # Roofline ceiling
    ai_range = np.logspace(-1, 4, 400)
    memory_roof  = PEAK_HBM_GBS * ai_range          # GFLOPS
    compute_roof = np.full_like(ai_range, PEAK_BF16_GFLOPS)
    roofline     = np.minimum(memory_roof, compute_roof)
    ridge_point  = PEAK_BF16_GFLOPS / PEAK_HBM_GBS  # ~289 FLOP/byte

    ax.plot(ai_range, roofline, "k-", linewidth=2, label="Roofline (A5000 bf16)", zorder=5)
    ax.axvline(ridge_point, color="k", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(ridge_point * 1.1, PEAK_BF16_GFLOPS * 0.6,
            f"Ridge\n{ridge_point:.0f} FLOP/B", fontsize=8, alpha=0.7)

    # Data points
    for impl in bw_df["impl"].unique():
        sub = bw_df[bw_df["impl"] == impl]
        color = IMPL_COLORS.get(impl, "gray")
        label = IMPL_LABELS.get(impl, impl)
        ax.scatter(sub["arithmetic_intensity"], sub["achieved_GFLOPS"],
                   label=label, color=color, s=80, zorder=10, edgecolors="white", linewidth=0.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
    ax.set_ylabel("Achieved Performance (GFLOPS)", fontsize=11)
    ax.set_title("Roofline Analysis – StreamTopK-SAE (NVIDIA RTX A5000)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

    for ext in ("pdf", "png"):
        out = os.path.join(REPORT_DIR, f"roofline.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Wrote results/report/roofline.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Plot 2: Runtime bar chart — latency per shape per implementation
# ---------------------------------------------------------------------------
def make_runtime_bar_chart(rt_df):
    if rt_df is None:
        return

    # Focus on bf16 shapes, exclude cpu (too slow, compresses the y-axis)
    df = rt_df[(rt_df["dtype"].isin(["bf16"])) &
               (rt_df["impl"].isin(["eager", "cuda_exact", "cuda_approx"]))].copy()
    df["median_ms"] = df["median_us"] / 1000

    # Build a label per (B, F, k) combination
    shape_keys = ["B", "F", "k"]
    shapes = df[shape_keys].drop_duplicates().sort_values(["F", "B", "k"])
    impls  = ["eager", "cuda_exact", "cuda_approx"]

    x      = np.arange(len(shapes))
    width  = 0.25
    offsets = np.array([-width, 0, width])

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, impl in enumerate(impls):
        vals = []
        for _, s in shapes.iterrows():
            row = df[(df["impl"] == impl) &
                     (df["B"] == s["B"]) & (df["F"] == s["F"]) & (df["k"] == s["k"])]
            vals.append(row["median_ms"].values[0] if len(row) else 0)
        color = IMPL_COLORS.get(impl, "gray")
        label = IMPL_LABELS.get(impl, impl)
        bars = ax.bar(x + offsets[i], vals, width, label=label, color=color, alpha=0.85)

    xlabels = [f"B={r.B}, F={int(r.F)//1024}K\nk={r.k}" for _, r in shapes.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms, log scale)", fontsize=11)
    ax.set_title("Latency by Implementation – bf16, RTX A5000", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.2)
    ax.set_ylim(bottom=0.1)

    for ext in ("pdf", "png"):
        out = os.path.join(REPORT_DIR, f"runtime_bar.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Wrote results/report/runtime_bar.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Plot 3: Memory savings — working memory vs F
# ---------------------------------------------------------------------------
def make_memory_plot(mem_df):
    if mem_df is None:
        return

    df = mem_df[mem_df["dtype"].isin(["bf16", "fp16"])].copy()
    impls = ["eager", "cuda_exact", "cuda_approx"]
    df_F  = df.groupby(["impl", "F"])["working_mem_mb"].median().reset_index()
    Fs    = sorted(df_F["F"].unique())

    fig, ax = plt.subplots(figsize=(7, 4))
    for impl in impls:
        sub = df_F[df_F["impl"] == impl].sort_values("F")
        if sub.empty:
            continue
        color = IMPL_COLORS.get(impl, "gray")
        label = IMPL_LABELS.get(impl, impl)
        ax.plot(sub["F"], sub["working_mem_mb"], marker="o", color=color, label=label, linewidth=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("F (number of latents)", fontsize=11)
    ax.set_ylabel("Working Memory (MB)", fontsize=11)
    ax.set_title("Peak Working Memory vs F – B=32, d=768, bf16", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

    for ext in ("pdf", "png"):
        out = os.path.join(REPORT_DIR, f"memory.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Wrote results/report/memory.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Plot 4: Recall vs c for approximate kernel
# ---------------------------------------------------------------------------
def make_recall_plot(rc_df):
    if rc_df is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10.colors

    groups = rc_df.groupby(["F", "k"])
    for i, ((F, k), sub) in enumerate(groups):
        sub_sorted = sub.sort_values("c")
        ax.plot(sub_sorted["c"], sub_sorted["recall_mean"],
                marker="o", color=colors[i % len(colors)],
                label=f"F={F:,}, k={k}", linewidth=2)
        ax.fill_between(sub_sorted["c"],
                        sub_sorted["recall_mean"] - sub_sorted["recall_std"],
                        sub_sorted["recall_mean"] + sub_sorted["recall_std"],
                        color=colors[i % len(colors)], alpha=0.15)

    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.5, label="Perfect recall")
    ax.set_xlabel("Candidates per tile (c)", fontsize=11)
    ax.set_ylabel("Recall@k", fontsize=11)
    ax.set_title("Approximate Kernel: Recall vs Candidates per Tile", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    for ext in ("pdf", "png"):
        out = os.path.join(REPORT_DIR, f"recall.{ext}")
        fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Wrote results/report/recall.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def make_runtime_table(df):
    if df is None:
        return
    idx  = df.groupby(["B", "d", "F", "k", "dtype"])["median_us"].idxmin()
    best = df.loc[idx, ["B", "d", "F", "k", "dtype", "impl", "median_us"]]
    eager = df[df["impl"] == "eager"][["B", "d", "F", "k", "dtype", "median_us"]]
    eager = eager.rename(columns={"median_us": "eager_us"})
    merged = best.merge(eager, on=["B", "d", "F", "k", "dtype"], how="left")
    merged["speedup_vs_eager"] = merged["eager_us"] / merged["median_us"]
    merged.to_csv(os.path.join(REPORT_DIR, "runtime_table.csv"), index=False)
    merged.to_markdown(os.path.join(REPORT_DIR, "runtime_table.md"), index=False)
    print(f"  Wrote results/report/runtime_table.{{csv,md}}")


def make_memory_table(df):
    if df is None:
        return
    pivot = df.groupby(["impl", "F", "dtype"])["working_mem_mb"].median().reset_index()
    pivot.to_csv(os.path.join(REPORT_DIR, "memory_table.csv"), index=False)
    pivot.to_markdown(os.path.join(REPORT_DIR, "memory_table.md"), index=False)
    print(f"  Wrote results/report/memory_table.{{csv,md}}")


def make_max_F_table(df):
    if df is None:
        return
    df.to_csv(os.path.join(REPORT_DIR, "max_F_table.csv"), index=False)
    df.to_markdown(os.path.join(REPORT_DIR, "max_F_table.md"), index=False)
    print(f"  Wrote results/report/max_F_table.{{csv,md}}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    print("Loading CSVs...")

    rt_df   = load_csv("runtime.csv")
    mem_df  = load_csv("memory.csv")
    bw_df   = load_csv("bandwidth.csv")
    rc_df   = load_csv("recall.csv")
    maxf_df = load_csv("max_F.csv")

    print("\nGenerating tables and plots...")
    make_runtime_table(rt_df)
    make_memory_table(mem_df)
    make_max_F_table(maxf_df)
    make_roofline_plot(bw_df)
    make_runtime_bar_chart(rt_df)
    make_memory_plot(mem_df)
    make_recall_plot(rc_df)

    print(f"\nAll outputs written to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
