"""
Ground-truth reference implementation for TopK SAE forward pass.
Dense matmul + topk, fp32 accumulation.
"""

import torch


def reference_topk_sae(X, W_enc, b_enc, k):
    """
    X:     (B, d)
    W_enc: (F, d)
    b_enc: (F,)
    Returns: values (B, k) fp32, indices (B, k) int32
    Output values are NOT sorted within each row.
    """
    preacts = X @ W_enc.T + b_enc
    values, indices = torch.topk(preacts.float(), k, dim=-1)
    return values, indices.to(torch.int32)
