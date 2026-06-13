"""
Tests for statistical properties of the synthetic preactivation generator.
"""

import math
import numpy as np
import torch
import pytest
from streamtopk_sae.synthetic import generate_realistic_preacts


B, F, d, k = 512, 4096, 256, 32


@pytest.fixture(scope="module")
def preacts():
    X, W_enc, b_enc = generate_realistic_preacts(B, F, d, seed=0)
    Z = X.float() @ W_enc.float().T + b_enc
    return Z


def test_topk_positivity(preacts):
    """Top-k preactivations per row positive in >95% of rows."""
    Z = preacts
    topk_vals, _ = torch.topk(Z, k, dim=1)
    kth_val = topk_vals[:, -1]
    frac = (kth_val > 0).float().mean().item()
    assert frac > 0.95, f"top-k positivity={frac:.3f} < 0.95"


def test_firing_rate_long_tail(preacts):
    """Top 1% of features fire on >10x the median rate."""
    Z = preacts
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
    assert ratio > 10, f"firing rate ratio={ratio:.1f}x < 10"


def test_index_nonuniformity(preacts):
    """Top-k indices NOT uniform (entropy < 0.9 * log F)."""
    Z = preacts
    _, all_idxs = torch.topk(Z, k, dim=1)
    fires = torch.zeros(F)
    for fi in all_idxs.view(-1).tolist():
        fires[fi] += 1
    probs = fires / fires.sum().clamp(min=1e-8)
    probs_np = probs.numpy()
    probs_np = probs_np[probs_np > 0]
    entropy = -np.sum(probs_np * np.log(probs_np))
    max_entropy = math.log(F)
    ratio = entropy / max_entropy
    assert ratio < 0.9, f"entropy ratio={ratio:.3f} >= 0.9"


def test_boundary_separation(preacts):
    """k-th-largest preact well-separated from (k+1)-th in >80% of rows."""
    Z = preacts
    topkp1_vals, _ = torch.topk(Z, k + 1, dim=1)
    kth   = topkp1_vals[:, k - 1]
    kp1th = topkp1_vals[:, k]
    gap   = kth - kp1th
    sep_frac = (gap > 0.05 * kth.abs().clamp(min=1e-8)).float().mean().item()
    assert sep_frac > 0.80, f"boundary separation fraction={sep_frac:.3f} < 0.80"
