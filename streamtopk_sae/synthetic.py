"""
Synthetic preactivation generator for TopK SAE benchmarks.

Produces X, W_enc, b_enc such that Z = X @ W_enc.T + b_enc has statistics
resembling real TopK SAE preactivations:
  1. Heavy-tailed positive tail per row.
  2. Long-tail per-feature firing rate distribution.
  3. Within-row correlation between high-scoring features.
  4. Bias structure with a small fraction of frequently-firing features.

CLI: python -m streamtopk_sae.synthetic --diagnostics
"""

import sys
import math
import torch
import numpy as np


def generate_realistic_preacts(
    B: int,
    F: int,
    d: int,
    n_themes: int = None,  # defaults to min(d, F//4) — at most d for orthogonality
    k_effective: int = 32,
    heavy_tail_alpha: float = 2.5,
    seed: int = 0,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Returns X (B, d), W_enc (F, d), b_enc (F,).
    Z = X @ W_enc.T + b_enc approximates real TopK SAE preactivation statistics.

    Key design choices:
    - n_themes <= d so themes can be made orthogonal via QR. Orthogonal themes
      ensure inactive features score near-zero (critical for boundary separation).
    - Zipf-weighted theme activation creates long-tail firing rate distribution.
    - Pareto-weighted X theme coefficients create heavy-tailed positive scores.
    - Small bias on a subset of features maintains bias structure from real SAEs.
    - Bias kept << in-cluster scores to preserve boundary separation.
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    np_rng = np.random.default_rng(seed)

    # n_themes must be <= d for orthogonality. features_per_theme * n_active ≈ k_effective.
    features_per_theme = max(2, min(16, k_effective // 2))
    if n_themes is None:
        n_themes = min(d, F // features_per_theme)
    n_themes = min(n_themes, d)  # enforce orthogonality constraint
    features_per_theme = F // n_themes  # recalculate to fill F exactly

    # 1. Orthogonal theme vectors via QR decomposition (n_themes <= d).
    raw = torch.randn(d, n_themes, generator=rng, dtype=torch.float32)
    Q, _ = torch.linalg.qr(raw)          # Q: (d, n_themes), columns orthonormal
    themes = Q.T.contiguous()             # themes: (n_themes, d), rows orthonormal

    # 2. Assign each feature to exactly one theme (round-robin, then shuffle).
    feature_themes = np.tile(np.arange(n_themes), features_per_theme + 1)[:F]
    np_rng.shuffle(feature_themes)

    # W_enc: theme direction + tiny noise (eps=0.01). Normalized to unit vectors.
    # Tiny noise differentiates features within the same cluster (needed for realistic
    # within-cluster score spread) while keeping inactive feature scores near-zero.
    eps = 0.01
    W_enc_f32 = torch.stack([themes[feature_themes[fi]] for fi in range(F)])
    W_enc_f32 = W_enc_f32 + eps * torch.randn(F, d, generator=rng, dtype=torch.float32)
    W_enc_f32 = W_enc_f32 / W_enc_f32.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # 3. For each row, activate n_active_themes theme groups with Zipf-weighted probability.
    #    Zipf weighting creates long-tail firing rate distribution (popular themes >> rare).
    n_active_themes = max(1, k_effective // features_per_theme)
    # Zipf probabilities for theme selection
    theme_pop   = 1.0 / (np.arange(1, n_themes + 1) ** 1.5)
    theme_probs = theme_pop / theme_pop.sum()

    X_f32 = torch.zeros(B, d, dtype=torch.float32)
    for bi in range(B):
        active = np_rng.choice(n_themes, size=n_active_themes,
                                replace=False, p=theme_probs)
        # Pareto weights: creates heavy-tailed positive preactivations
        w_np  = (np_rng.pareto(heavy_tail_alpha, size=n_active_themes) + 1.0).astype(np.float32)
        X_f32[bi] = (themes[active] * torch.from_numpy(w_np).unsqueeze(1)).sum(0)
    X_f32 = X_f32 / X_f32.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # 4. b_enc: small bias for 5% of features. Kept small (max ≈ 0.05) so that biased
    #    out-cluster features cannot outscore the weakest in-cluster feature.
    n_biased   = max(1, F // 20)
    biased_idx = np_rng.choice(F, size=n_biased, replace=False)
    b_enc_f32  = torch.zeros(F, dtype=torch.float32)
    # Bias scale 0.03: small enough that even max-biased out-cluster << min in-cluster
    b_enc_f32[biased_idx] = torch.from_numpy(
        (np_rng.pareto(2.0, size=n_biased) * 0.03).astype(np.float32)
    )

    X     = X_f32.to(dtype)
    W_enc = W_enc_f32.to(dtype)
    b_enc = b_enc_f32  # always fp32

    return X, W_enc, b_enc


def _run_diagnostics():
    """Print statistical properties of the synthetic generator output."""
    import scipy.stats

    B, F, d, k = 512, 4096, 256, 32

    print(f"Generating synthetic preactivations: B={B}, F={F}, d={d}, k={k}")
    X, W_enc, b_enc = generate_realistic_preacts(B, F, d, seed=0)

    # Compute Z on CPU in fp32
    Z = X.float() @ W_enc.float().T + b_enc

    # 1. Top-k positivity: fraction of rows where all top-k preacts are positive
    topk_vals, _ = torch.topk(Z, k, dim=1)
    kth_val = topk_vals[:, -1]  # smallest top-k value per row
    frac_positive = (kth_val > 0).float().mean().item()
    print(f"\n[1] Top-k positivity (>95% expected): {frac_positive:.3f}")
    assert frac_positive > 0.95, f"FAIL: top-k positivity={frac_positive:.3f} < 0.95"
    print("    PASS")

    # 2. Per-feature firing rate long-tail
    _, all_idxs = torch.topk(Z, k, dim=1)
    fires = torch.zeros(F)
    for fi in all_idxs.view(-1).tolist():
        fires[fi] += 1
    fires = fires / B
    fires_sorted = fires.sort(descending=True).values
    top1pct = max(1, F // 100)
    median_rate = fires_sorted[F // 2].item()
    top1pct_rate = fires_sorted[:top1pct].mean().item()
    ratio = top1pct_rate / max(median_rate, 1e-8)
    print(f"\n[2] Firing rate long-tail (top 1% / median > 10x expected): {ratio:.1f}x")
    assert ratio > 10, f"FAIL: firing rate ratio={ratio:.1f}x < 10"
    print("    PASS")

    # 3. Index non-uniformity (entropy < 0.9 * log F)
    probs = fires / fires.sum().clamp(min=1e-8)
    probs_np = probs.numpy()
    probs_np = probs_np[probs_np > 0]
    entropy = -np.sum(probs_np * np.log(probs_np))
    max_entropy = math.log(F)
    ratio_e = entropy / max_entropy
    print(f"\n[3] Index non-uniformity (entropy/logF < 0.9 expected): {ratio_e:.3f}")
    assert ratio_e < 0.9, f"FAIL: entropy ratio={ratio_e:.3f} >= 0.9"
    print("    PASS")

    # 4. k-th/k+1-th separation: fraction of rows where kth > k+1th by a margin
    if k < F:
        topkp1_vals, _ = torch.topk(Z, k + 1, dim=1)
        kth   = topkp1_vals[:, k - 1]
        kp1th = topkp1_vals[:, k]
        gap   = kth - kp1th
        # "well-separated" = gap > 5% of kth magnitude
        sep_frac = (gap > 0.05 * kth.abs().clamp(min=1e-8)).float().mean().item()
        print(f"\n[4] Boundary separation (>80% expected): {sep_frac:.3f}")
        assert sep_frac > 0.80, f"FAIL: separation fraction={sep_frac:.3f} < 0.80"
        print("    PASS")
    else:
        print("\n[4] Boundary separation: skipped (k == F)")

    print("\nAll diagnostics passed.")


if __name__ == "__main__":
    if "--diagnostics" in sys.argv:
        _run_diagnostics()
    else:
        print("Usage: python -m streamtopk_sae.synthetic --diagnostics")
        sys.exit(1)
