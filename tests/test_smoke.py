"""
Smoke tests: import and minimal forward pass for each impl.
"""

import torch
import pytest


def test_import():
    import streamtopk_sae
    assert hasattr(streamtopk_sae, "reference_topk_sae")


def test_reference():
    from streamtopk_sae.reference import reference_topk_sae
    X     = torch.randn(4, 64)
    W_enc = torch.randn(128, 64)
    b_enc = torch.zeros(128)
    v, i  = reference_topk_sae(X, W_enc, b_enc, 8)
    assert v.shape == (4, 8)
    assert i.shape == (4, 8)
    assert v.dtype == torch.float32
    assert i.dtype == torch.int32


def test_synthetic_generate():
    from streamtopk_sae.synthetic import generate_realistic_preacts
    X, W, b = generate_realistic_preacts(8, 64, 32, seed=0)
    assert X.shape  == (8, 32)
    assert W.shape  == (64, 32)
    assert b.shape  == (64,)


def test_baseline_eager():
    from streamtopk_sae.baselines import baseline_eager
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X     = torch.randn(4, 64, device=device, dtype=torch.bfloat16)
    W_enc = torch.randn(128, 64, device=device, dtype=torch.bfloat16)
    b_enc = torch.zeros(128, device=device, dtype=torch.float32)
    v, i  = baseline_eager(X, W_enc, b_enc, 8)
    assert v.shape == (4, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_op():
    from streamtopk_sae.ops import topk_sae_cpu
    X     = torch.randn(4, 64, dtype=torch.float32)
    W_enc = torch.randn(128, 64, dtype=torch.float32)
    b_enc = torch.zeros(128, dtype=torch.float32)
    v, i  = topk_sae_cpu(X, W_enc, b_enc, 8)
    assert v.shape == (4, 8)
    assert i.shape == (4, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_exact_op():
    from streamtopk_sae.ops import topk_sae_cuda_exact
    X     = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    W_enc = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    b_enc = torch.zeros(128, dtype=torch.float32, device="cuda")
    v, i  = topk_sae_cuda_exact(X, W_enc, b_enc, 8)
    assert v.shape == (4, 8)
    assert i.shape == (4, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_approx_op():
    from streamtopk_sae.ops import topk_sae_cuda_approx
    X     = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
    W_enc = torch.randn(128, 64, dtype=torch.bfloat16, device="cuda")
    b_enc = torch.zeros(128, dtype=torch.float32, device="cuda")
    v, i  = topk_sae_cuda_approx(X, W_enc, b_enc, 8, 16)
    assert v.shape == (4, 8)
    assert i.shape == (4, 8)
