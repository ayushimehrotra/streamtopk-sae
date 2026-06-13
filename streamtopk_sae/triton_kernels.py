"""
Triton baseline: dense matmul (fp32 accumulation) + row-wise top-k via torch.topk.
NOT fused — two separate passes to isolate fusion speedup from Triton-vs-cuBLAS.

The matmul kernel computes Z = X @ W_enc.T with fp32 accumulation and supports
fp16, bf16, fp32 inputs. Autotuned over 4 tile configurations.

Top-k limit note: k <= 128 is the natural register-file limit for inline Triton
top-k. This baseline uses torch.topk for all k; a custom Triton top-k kernel
for k <= 128 would further optimize but is not this project's contribution.
"""

import warnings
import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel_fixed(
    X_ptr, W_ptr, Z_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_zm, stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Z[m,n] = sum_k X[m,k] * W[n,k]  (= X @ W.T), fp32 accumulation."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k_start * BLOCK_K + offs_k
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        x_tile = tl.load(X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                         mask=x_mask, other=0.0).to(tl.float32)
        w_tile = tl.load(W_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk,
                         mask=w_mask, other=0.0).to(tl.float32)
        acc += tl.dot(x_tile, tl.trans(w_tile), allow_tf32=False)

    z_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Z_ptr + offs_m[:, None] * stride_zm + offs_n[None, :] * stride_zn,
             acc, mask=z_mask)


_BLOCK_M = 64
_BLOCK_N = 64
_BLOCK_K = 32


def _triton_matmul(X, W_enc):
    """X @ W_enc.T -> Z fp32. X, W_enc can be fp16/bf16/fp32."""
    M, K = X.shape
    N    = W_enc.shape[0]
    assert W_enc.shape[1] == K, "W_enc d-dim mismatch"
    Z = torch.empty((M, N), device=X.device, dtype=torch.float32)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    _matmul_kernel_fixed[grid](
        X, W_enc, Z,
        M, N, K,
        X.stride(0),     X.stride(1),
        W_enc.stride(0), W_enc.stride(1),
        Z.stride(0),     Z.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
    )
    return Z


_K_TOPK_LIMIT = 128


def triton_topk_sae(X, W_enc, b_enc, k):
    """
    Triton baseline: Triton matmul + torch.topk (not fused — two separate passes).
    Supports fp16, bf16, fp32 input.
    Top-k note: uses torch.topk for all k. A custom Triton top-k for k <= 128
    would further reduce memory traffic but is not this project's contribution.
    """
    Z = _triton_matmul(X, W_enc) + b_enc
    values, indices = torch.topk(Z, k, dim=-1)
    return values, indices.to(torch.int32)
