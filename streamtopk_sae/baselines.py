"""
Baseline implementations for TopK SAE forward pass:
  1. baseline_eager: dense matmul + fp32 topk on GPU, weak baseline.
  2. baseline_compiled: same wrapped in torch.compile(max-autotune).
  3. baseline_triton: see triton_kernels.py.
"""

import torch
import functools
from streamtopk_sae.triton_kernels import triton_topk_sae


def baseline_eager(X, W_enc, b_enc, k):
    """Dense matmul in input dtype, topk in fp32. Weak GPU baseline."""
    preacts = (X @ W_enc.T + b_enc).float()
    values, indices = torch.topk(preacts, k, dim=-1)
    return values, indices.to(torch.int32)


# Module-level cache for compiled function
_compiled_fn = None
_compiled_key = None


def _make_compiled():
    @torch.compile(mode="max-autotune", fullgraph=True)
    def _inner(X, W_enc, b_enc, k):
        # Cast bias to input dtype before matmul to avoid mixed-dtype errors in inductor
        preacts = (X @ W_enc.T + b_enc.to(X.dtype)).float()
        values, indices = torch.topk(preacts, k, dim=-1)
        return values, indices.to(torch.int32)
    return _inner


def baseline_compiled(X, W_enc, b_enc, k):
    """Dense matmul + topk wrapped in torch.compile(max-autotune)."""
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = _make_compiled()
    return _compiled_fn(X, W_enc, b_enc, k)


def warmup_compiled(B, d, F, k, dtype):
    """Warm up the compiled baseline with dummy inputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    X     = torch.randn(B, d, device=device, dtype=dt)
    W_enc = torch.randn(F, d, device=device, dtype=dt)
    b_enc = torch.zeros(F, device=device, dtype=torch.float32)
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = _make_compiled()
    for _ in range(3):
        baseline_compiled(X, W_enc, b_enc, k)
    torch.cuda.synchronize()


def baseline_triton(X, W_enc, b_enc, k):
    """Triton matmul + row-wise topk (not fused). See triton_kernels.py."""
    return triton_topk_sae(X, W_enc, b_enc, k)
