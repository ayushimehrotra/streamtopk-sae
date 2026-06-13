"""
Utility helpers: dtype mapping, recall computation, timing.
"""

import os
import time
import torch


DTYPE_MAP = {
    "fp16":  torch.float16,
    "bf16":  torch.bfloat16,
    "fp32":  torch.float32,
}


def str_to_dtype(s):
    if s not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{s}'. Choose from: {list(DTYPE_MAP)}")
    return DTYPE_MAP[s]


def recall_at_k(pred_idxs, true_idxs):
    """
    Compute Recall@k per row.
    pred_idxs: (B, k) int32/int64
    true_idxs: (B, k) int32/int64
    Returns: scalar mean recall over batch.
    """
    B, k = true_idxs.shape
    recalls = []
    for i in range(B):
        true_set = set(true_idxs[i].tolist())
        pred_set = set(pred_idxs[i].tolist())
        recalls.append(len(true_set & pred_set) / k)
    return sum(recalls) / B


def use_synthetic():
    """Return True if synthetic generator should be used (default); False for gaussian."""
    return os.environ.get("STREAMTOPK_DATA", "synthetic").lower() != "gaussian"


def make_inputs(B, d, F, dtype=torch.bfloat16, seed=0, device="cuda"):
    """Generate inputs using synthetic generator or gaussian depending on env var."""
    from streamtopk_sae.synthetic import generate_realistic_preacts
    if use_synthetic():
        X, W_enc, b_enc = generate_realistic_preacts(B, F, d, seed=seed, dtype=dtype)
    else:
        rng = torch.Generator()
        rng.manual_seed(seed)
        X     = torch.randn(B, d, dtype=dtype, generator=rng)
        W_enc = torch.randn(F, d, dtype=dtype, generator=rng)
        b_enc = torch.zeros(F, dtype=torch.float32)
    return X.to(device), W_enc.to(device), b_enc.to(device)
