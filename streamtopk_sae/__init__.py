"""
StreamTopK-SAE: streaming fused top-k SAE encoder.

Provides CPU, CUDA exact, and CUDA approximate implementations of the TopK
SAE forward pass (matmul + bias + row-wise top-k) without materializing the
full (B, F) score matrix.
"""

from streamtopk_sae.reference import reference_topk_sae
from streamtopk_sae.ops import topk_sae_cpu, topk_sae_cuda_exact, topk_sae_cuda_approx
from streamtopk_sae.baselines import baseline_eager, baseline_compiled, baseline_triton
from streamtopk_sae.utils import recall_at_k, make_inputs

__all__ = [
    "reference_topk_sae",
    "topk_sae_cpu",
    "topk_sae_cuda_exact",
    "topk_sae_cuda_approx",
    "baseline_eager",
    "baseline_compiled",
    "baseline_triton",
    "recall_at_k",
    "make_inputs",
]
