"""
Python-level wrappers for native C++/CUDA extensions.
Handles dtype casting, shape validation, and CUDA availability checks.
"""

import os
import torch

try:
    import streamtopk_sae_native as _native
    _NATIVE_AVAILABLE = True
except ImportError:
    _NATIVE_AVAILABLE = False


def _check_native():
    if not _NATIVE_AVAILABLE:
        raise RuntimeError(
            "streamtopk_sae_native extension not built. Run: pip install -e ."
        )


def topk_sae_cpu(X, W_enc, b_enc, k):
    """
    CPU streaming top-k SAE forward.
    Inputs fp16/bf16 are upcast to fp32 (no native fp16 arithmetic on CPU).
    Returns: values (B, k) fp32, indices (B, k) int32.
    """
    _check_native()
    if X.dtype != torch.float32:
        X     = X.float()
        W_enc = W_enc.float()
    if b_enc.dtype != torch.float32:
        b_enc = b_enc.float()

    X     = X.contiguous()
    W_enc = W_enc.contiguous()
    b_enc = b_enc.contiguous()

    return _native.streamtopk_cpu_forward(X, W_enc, b_enc, k)


def topk_sae_cuda_exact(X, W_enc, b_enc, k):
    """
    CUDA exact streaming top-k SAE forward.
    Supports fp16, bf16, fp32. b_enc always fp32.
    Returns: values (B, k) fp32, indices (B, k) int32.
    """
    _check_native()
    if b_enc.dtype != torch.float32:
        b_enc = b_enc.float()

    X     = X.contiguous()
    W_enc = W_enc.contiguous()
    b_enc = b_enc.contiguous()

    return _native.streamtopk_cuda_exact_forward(X, W_enc, b_enc, k)


def topk_sae_cuda_approx(X, W_enc, b_enc, k, c, candidate_buffer=None):
    """
    CUDA approximate streaming top-k SAE forward.
    Uses block-candidate selection: top-c per tile, final top-k via torch.topk.
    c must be in {16, 32, 64, 128} and >= k.
    Returns: values (B, k) fp32, indices (B, k) int32.
    """
    _check_native()
    if b_enc.dtype != torch.float32:
        b_enc = b_enc.float()

    X     = X.contiguous()
    W_enc = W_enc.contiguous()
    b_enc = b_enc.contiguous()

    flat_vals, flat_idxs, _, _ = _native.streamtopk_cuda_approx_forward(
        X, W_enc, b_enc, k, c, candidate_buffer
    )

    # Final top-k over candidate set (Python-side, v1)
    top_vals, top_pos = torch.topk(flat_vals, k, dim=-1)
    top_idxs = flat_idxs.gather(1, top_pos).to(torch.int32)

    return top_vals, top_idxs
